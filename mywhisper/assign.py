"""
Speaker assignment pipeline for mywhisper.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Sequence, Tuple

import json
import logging
from collections import defaultdict

import requests
import spacy

from .config import ensure_data_subdir, generate_artefact_key, resolve_data_root
from .models import (
    PipelineEvent,
    PodcastEpisode,
    SpeakerAssignment,
    SpeakerProfile,
    TranscriptSegment,
)

LOGGER = logging.getLogger("mywhisper.assign")


@dataclass(slots=True)
class AssignmentConfig:
    """
    Configuration for speaker assignment pipeline.
    """

    ollama_model: str = "llama3"
    max_iterations: int = 2
    high_confidence_threshold: float = 0.7
    spacy_model: str = "en_core_web_sm"
    sample_utterances_start: int = 3
    sample_utterances_end: int = 3
    output_dir: Path = field(default_factory=lambda: ensure_data_subdir("transcripts"))
    data_root: Path = field(default_factory=resolve_data_root)

    def assignment_path(self, podcast: PodcastEpisode, artefact_key: str) -> Path:
        slug = podcast.artefact_slug()
        dir_path = ensure_data_subdir(f"transcripts/{slug}", self.data_root)
        return dir_path / f"{artefact_key}_with_names.json"


class SpeakerProfileBuilder:
    """
    Construct speaker profiles from transcript segments.
    """

    def build(
        self,
        segments: Sequence[TranscriptSegment],
        sample_start: int,
        sample_end: int,
    ) -> Dict[str, SpeakerProfile]:
        grouped: Dict[str, List[TranscriptSegment]] = defaultdict(list)
        for segment in segments:
            speaker_id = segment.speaker_id or "UNKNOWN"
            grouped[speaker_id].append(segment)

        profiles: Dict[str, SpeakerProfile] = {}
        for speaker_id, speaker_segments in grouped.items():
            profile = SpeakerProfile(speaker_id=speaker_id)
            profile.update_from_segments(
                speaker_segments,
                sample_start=sample_start,
                sample_end=sample_end,
            )
            profiles[speaker_id] = profile
        return profiles


class CandidateRoster:
    """
    Build candidate roster from metadata and NLP hints.
    """

    def __init__(self, config: AssignmentConfig) -> None:
        self.config = config
        self._nlp = None

    def load_spacy_model(self) -> Optional["spacy.language.Language"]:  # type: ignore[name-defined]
        if self._nlp is None:
            try:
                self._nlp = spacy.load(self.config.spacy_model)
            except OSError:
                LOGGER.warning("spaCy model %s not found; skipping person extraction.", self.config.spacy_model)
                self._nlp = None
        return self._nlp

    def compile(
        self,
        podcast: PodcastEpisode,
        additional: Optional[Iterable[str]] = None,
    ) -> List[str]:
        roster = {name.strip() for name in (additional or []) if name}

        metadata = podcast.metadata or {}
        host_roster = metadata.get("host_roster")
        if isinstance(host_roster, (list, tuple, set)):
            roster.update(str(name).strip() for name in host_roster if name)
        elif isinstance(host_roster, str):
            roster.update(part.strip() for part in host_roster.split(",") if part.strip())

        # Fallback names
        roster.update(
            filter(
                None,
                [
                    podcast.show_title.split(" - ")[0],
                    "Unknown Host",
                    "Unknown Guest",
                ],
            )
        )

        candidates_texts = [
            metadata.get("description", ""),
            metadata.get("episode_description", ""),
            metadata.get("summary", ""),
            podcast.episode_title,
        ]
        combined_text = "\n".join(filter(None, candidates_texts))
        nlp = self.load_spacy_model()
        if nlp and combined_text:
            doc = nlp(combined_text)
            roster.update(ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON")

        return sorted(roster)


class LLMClient:
    """
    Base LLM client abstraction.
    """

    def generate(self, prompt: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class OllamaClient(LLMClient):
    """
    Ollama HTTP API client.
    """

    def __init__(self, model_name: str, endpoint: str = "http://localhost:11434/api/generate") -> None:
        self.model_name = model_name
        self.endpoint = endpoint

    def generate(self, prompt: str) -> str:
        payload = {"model": self.model_name, "prompt": prompt, "stream": False}
        response = requests.post(self.endpoint, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")


class SpeakerInferenceEngine:
    """
    Handles prompt building, LLM invocation, and result consolidation.
    """

    def __init__(self, config: AssignmentConfig, client: LLMClient, logger: Optional[logging.Logger] = None) -> None:
        self.config = config
        self.client = client
        self.logger = logger or LOGGER

    def infer(
        self,
        profiles: Dict[str, SpeakerProfile],
        roster: Sequence[str],
        context_summary: str,
        target_speakers: Sequence[str],
    ) -> List[SpeakerAssignment]:
        speaker_blocks = [
            profiles[speaker_id].to_prompt_block()
            for speaker_id in target_speakers
            if speaker_id in profiles
        ]
        prompt = self._build_prompt(speaker_blocks, roster, context_summary)
        self.logger.debug("Invoking LLM with prompt length %d characters", len(prompt))
        response = self.client.generate(prompt)
        return self._parse_assignments(response)

    def critic(self, assignments: Sequence[SpeakerAssignment]) -> Dict[str, bool]:
        consistency: Dict[str, bool] = {}
        seen: Dict[str, str] = {}
        for assignment in assignments:
            name = assignment.proposed_name.strip().lower()
            if name == "unknown":
                consistency[assignment.speaker_id] = assignment.confidence <= self.config.high_confidence_threshold
                continue
            previous = seen.get(name)
            if previous and previous != assignment.speaker_id:
                consistency[assignment.speaker_id] = False
                consistency[previous] = False
            else:
                seen[name] = assignment.speaker_id
                consistency.setdefault(assignment.speaker_id, True)
        return consistency

    def consolidate(
        self,
        prior: Dict[str, SpeakerAssignment],
        new: Iterable[SpeakerAssignment],
    ) -> Dict[str, SpeakerAssignment]:
        merged = dict(prior)
        for assignment in new:
            current = merged.get(assignment.speaker_id)
            if current is None or assignment.confidence > current.confidence:
                merged[assignment.speaker_id] = assignment
        return merged

    def _build_prompt(
        self,
        speaker_blocks: Sequence[str],
        roster: Sequence[str],
        context_summary: str,
    ) -> str:
        candidates = ", ".join(sorted(roster)) if roster else "Unknown"
        speaker_section = "\n\n".join(speaker_blocks)
        instructions = (
            "You are identifying real speaker names for diarized podcast segments.\n"
            "Use the episode context and candidate names to infer who each speaker likely is.\n"
            "Return a JSON array; each element must include speaker_id, proposed_name, confidence (0-1), justification.\n"
            'If unsure, use "UNKNOWN" with confidence <= 0.3.\n'
            "Do not write anything other than JSON."
        )
        return (
            f"{instructions}\n\n"
            f"Episode Context:\n{context_summary}\n\n"
            f"Candidate Names: {candidates}\n\n"
            f"Speaker Evidence:\n{speaker_section}\n"
        )

    def _parse_assignments(self, raw_output: str) -> List[SpeakerAssignment]:
        raw_output = raw_output.strip()
        try:
            data = json.loads(raw_output)
        except json.JSONDecodeError:
            self.logger.error("Failed to parse LLM output: %s", raw_output)
            return []

        assignments: List[SpeakerAssignment] = []
        for item in data:
            try:
                assignments.append(
                    SpeakerAssignment(
                        speaker_id=str(item["speaker_id"]),
                        proposed_name=str(item.get("proposed_name", "UNKNOWN")).strip() or "UNKNOWN",
                        confidence=float(item.get("confidence", 0.0)),
                        justification=str(item.get("justification", "")).strip(),
                    )
                )
            except (TypeError, ValueError, KeyError) as exc:
                self.logger.warning("Skipping malformed assignment %s: %s", item, exc)
        return assignments


class TranscriptAssigner:
    """
    End-to-end pipeline for assigning speaker names to transcript segments.
    """

    def __init__(
        self,
        podcast: PodcastEpisode,
        config: AssignmentConfig,
        inference_engine: SpeakerInferenceEngine,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.podcast = podcast
        self.config = config
        self.inference_engine = inference_engine
        self.logger = logger or LOGGER.getChild(podcast.episode_id)

    @classmethod
    def from_config(
        cls,
        podcast: PodcastEpisode,
        config: AssignmentConfig,
        client: Optional[LLMClient] = None,
    ) -> "TranscriptAssigner":
        engine_client = client or OllamaClient(config.ollama_model)
        inference_engine = SpeakerInferenceEngine(config, engine_client)
        return cls(podcast=podcast, config=config, inference_engine=inference_engine)

    def assign_names(
        self,
        segments: Sequence[TranscriptSegment],
        metadata: Optional[Dict[str, str]] = None,
        yield_progress: bool = False,
    ) -> List[TranscriptSegment] | Generator[PipelineEvent, None, List[TranscriptSegment]]:
        pipeline = self._assign_pipeline(segments, metadata or {})
        if yield_progress:
            return pipeline

        try:
            while True:
                next(pipeline)
        except StopIteration as stop:
            return stop.value

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _assign_pipeline(
        self,
        segments: Sequence[TranscriptSegment],
        metadata: Dict[str, str],
    ) -> Generator[PipelineEvent, None, List[TranscriptSegment]]:
        artefact_key = generate_artefact_key()
        assignment_path = self.config.assignment_path(self.podcast, artefact_key)
        start_time = time.perf_counter()

        yield PipelineEvent(
            stage="start",
            message=f"Assigning speaker names for {self.podcast.episode_title}",
            payload={"artefact_key": artefact_key, "segment_count": len(segments)},
            elapsed=0.0,
        )

        profile_builder = SpeakerProfileBuilder()
        profiles = profile_builder.build(
            segments,
            sample_start=self.config.sample_utterances_start,
            sample_end=self.config.sample_utterances_end,
        )

        elapsed = time.perf_counter() - start_time
        yield PipelineEvent(
            stage="profiles_ready",
            message="Constructed speaker profiles",
            payload={"profiles": len(profiles)},
            elapsed=elapsed,
        )

        roster_builder = CandidateRoster(self.config)
        combined_metadata = dict(self.podcast.metadata or {})
        combined_metadata.update(metadata)
        additional = combined_metadata.get("host_roster")
        if isinstance(additional, str):
            extra_iterable: Iterable[str] = [name.strip() for name in additional.split(",")]
        else:
            extra_iterable = additional or []
        roster = roster_builder.compile(self.podcast, list(extra_iterable))

        context_summary = self._context_summary(combined_metadata)
        target_speakers = list(profiles.keys())
        if not target_speakers:
            return []

        assignments = self.inference_engine.infer(
            profiles=profiles,
            roster=roster,
            context_summary=context_summary,
            target_speakers=target_speakers,
        )
        assignment_map = self.inference_engine.consolidate({}, assignments)
        critic_flags = self.inference_engine.critic(list(assignment_map.values()))

        elapsed = time.perf_counter() - start_time
        yield PipelineEvent(
            stage="inference_round",
            message="Completed initial inference",
            payload={"assignments": len(assignments)},
            elapsed=elapsed,
        )

        unresolved = [
            speaker_id
            for speaker_id, assignment in assignment_map.items()
            if not assignment.is_high_confidence(self.config.high_confidence_threshold)
            or not critic_flags.get(speaker_id, True)
        ]

        if unresolved and self.config.max_iterations > 1:
            refined_assignments = self.inference_engine.infer(
                profiles=profiles,
                roster=roster,
                context_summary=context_summary + "\nFocus on uncertain speakers only.",
                target_speakers=unresolved,
            )
            assignment_map = self.inference_engine.consolidate(assignment_map, refined_assignments)
            critic_flags = self.inference_engine.critic(list(assignment_map.values()))

            elapsed = time.perf_counter() - start_time
            yield PipelineEvent(
                stage="refinement_round",
                message="Completed refinement inference",
                payload={"assignments": len(refined_assignments)},
                elapsed=elapsed,
            )

        final_assignments: Dict[str, SpeakerAssignment] = {}
        for speaker_id, assignment in assignment_map.items():
            if assignment.is_high_confidence(self.config.high_confidence_threshold) and critic_flags.get(
                speaker_id, True
            ):
                final_assignments[speaker_id] = assignment
            else:
                final_assignments[speaker_id] = SpeakerAssignment(
                    speaker_id=speaker_id,
                    proposed_name="UNKNOWN",
                    confidence=assignment.confidence,
                    justification=assignment.justification,
                )

        enriched_segments = self._label_segments(segments, final_assignments)
        self._persist_assignments(assignment_path, enriched_segments)
        self._last_assignment_path = assignment_path
        self._last_artefact_key = artefact_key

        elapsed = time.perf_counter() - start_time
        yield PipelineEvent(
            stage="persisted",
            message="Persisted speaker assignments",
            payload={"path": str(assignment_path), "segments": len(enriched_segments)},
            elapsed=elapsed,
        )

        return enriched_segments

    def _context_summary(self, metadata: Dict[str, str]) -> str:
        description = metadata.get("description") or metadata.get("episode_description") or ""
        published = metadata.get("published_at")
        summary_parts = [
            f"Show: {self.podcast.show_title}",
            f"Episode: {self.podcast.episode_title}",
        ]
        if published:
            summary_parts.append(f"Published: {published}")
        if description:
            summary_parts.append(f"Description: {description}")
        return "\n".join(summary_parts)

    def _label_segments(
        self,
        segments: Sequence[TranscriptSegment],
        assignments: Dict[str, SpeakerAssignment],
    ) -> List[TranscriptSegment]:
        labelled: List[TranscriptSegment] = []
        for seg in segments:
            assignment = assignments.get(seg.speaker_id or "")
            labelled.append(
                TranscriptSegment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    speaker_id=seg.speaker_id,
                    speaker_name=assignment.proposed_name if assignment else (seg.speaker_name or "UNKNOWN"),
                    confidence=assignment.confidence if assignment else seg.confidence,
                    justification=assignment.justification if assignment else seg.justification,
                    metadata=dict(seg.metadata),
                )
            )
        return labelled

    def _persist_assignments(self, path: Path, segments: Sequence[TranscriptSegment]) -> None:
        records = [
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "speaker": seg.speaker_id,
                "speaker_name": seg.speaker_name,
                "confidence": seg.confidence,
                "justification": seg.justification,
                "metadata": seg.metadata,
            }
            for seg in segments
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False, indent=2)

