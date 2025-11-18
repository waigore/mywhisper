"""
Speaker assignment pipeline for mywhisper.
"""

from __future__ import annotations

import time
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Sequence, Tuple

import json
import logging
from collections import defaultdict
import re

import requests
import spacy

from .config import ensure_data_subdir, ensure_episode_subdir, resolve_data_root
from .models import (
    PipelineEvent,
    PodcastEpisode,
    SpeakerAssignment,
    SpeakerNameGuesses,
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
    ollama_json_mode: bool = True
    ollama_use_json_schema: bool = True
    max_iterations: int = 1
    high_confidence_threshold: float = 0.7
    spacy_model: str = "en_core_web_sm"
    sample_utterances_start: int = 50
    sample_utterances_end: int = 50
    output_dir: Path = field(default_factory=lambda: ensure_data_subdir("transcripts"))
    data_root: Path = field(default_factory=resolve_data_root)

    def assignment_path(
        self,
        podcast: PodcastEpisode,
        episode_key: Optional[str] = None,
    ) -> Path:
        key = episode_key or podcast.episode_key
        dir_path = ensure_episode_subdir(key, self.data_root, "transcripts")
        return dir_path / f"{key}_with_names.json"


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
            speaker_id = segment.speaker_id
            if not speaker_id:
                continue
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

    def generate(self, prompt: str, json_schema: Optional[Dict[str, Any]] = None) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class OllamaClient(LLMClient):
    """
    Ollama HTTP API client.
    """

    def __init__(self, model_name: str, endpoint: str = "http://localhost:11434/api/generate", json_mode: bool = True) -> None:
        self.model_name = model_name
        self.endpoint = endpoint
        self.json_mode = json_mode

    def generate(self, prompt: str, json_schema: Optional[Dict[str, Any]] = None) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0},
        }
        if json_schema:
            # Prefer schema-guided JSON if available; keep temperature at 0 for determinism
            # Some Ollama versions accept a structured format object for schema guidance.
            payload["format"] = {"type": "json", "schema": json_schema}
        elif self.json_mode:
            payload["format"] = "json"
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
    ) -> Dict[str, SpeakerNameGuesses]:
        speaker_blocks = [
            profiles[speaker_id].to_prompt_block()
            for speaker_id in target_speakers
            if speaker_id in profiles
        ]
        prompt = self._build_prompt(speaker_blocks, roster, context_summary)
        self.logger.info("LLM prompt (%d chars):\n%s", len(prompt), prompt)
        schema: Optional[Dict[str, Any]] = self._json_schema() if getattr(self.config, "ollama_use_json_schema", True) else None
        # Be resilient to clients that don't accept json_schema kwarg (e.g., test stubs)
        try:
            response = self.client.generate(prompt, json_schema=schema)
        except TypeError:
            response = self.client.generate(prompt)
        self.logger.info("LLM response (%d chars):\n%s", len(response), response)
        return self._parse_assignments(response)

    def critic(self, assignments: Sequence[SpeakerAssignment]) -> Dict[str, bool]:
        """
        Evaluate assignments by enforcing a single highest-confidence speaker per
        normalized name. Additional speakers with the same name but lower confidence are
        rejected. When multiple speakers share the same confidence for the same name, the
        critic signals False for all tied speakers so the caller can trigger a tie-break.
        """

        consistency: Dict[str, bool] = {}
        grouped: Dict[str, List[SpeakerAssignment]] = defaultdict(list)
        for assignment in assignments:
            grouped[assignment.proposed_name.strip().lower()].append(assignment)

        for name, candidates in grouped.items():
            sorted_candidates = sorted(candidates, key=lambda item: item.confidence, reverse=True)
            top_confidence = sorted_candidates[0].confidence
            tied = [item for item in sorted_candidates if abs(item.confidence - top_confidence) < 1e-6]

            if name == "unknown":
                for candidate in sorted_candidates:
                    consistency[candidate.speaker_id] = candidate.confidence <= self.config.high_confidence_threshold
                continue

            if len(tied) == 1:
                consistency[sorted_candidates[0].speaker_id] = True
                for candidate in sorted_candidates[1:]:
                    consistency[candidate.speaker_id] = False
                continue

            for candidate in sorted_candidates:
                consistency[candidate.speaker_id] = False

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

    def select_best(self, guesses: Dict[str, SpeakerNameGuesses]) -> Dict[str, SpeakerAssignment]:
        best: Dict[str, SpeakerAssignment] = {}
        for speaker_id, guess in guesses.items():
            choice = guess.best()
            if choice:
                best[speaker_id] = choice
        return best

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
            "Return ONLY a valid JSON array. No prose, no markdown, no code fences.\n"
            "Each element must be an object: {\"speaker_id\": string, \"proposed_names\": [{\"name\": string, \"confidence\": number, \"justification\": string}]}.\n"
            "Confidence is between 0 and 1. Include multiple proposed_names ordered by confidence.\n"
            "Emit each speaker exactly once. If unsure, include a proposal with name \"UNKNOWN\" and confidence <= 0.3."
        )
        example = (
            '[\n'
            '  {\n'
            '    "speaker_id": "SPEAKER_00",\n'
            '    "proposed_names": [\n'
            '      {"name": "Anthony Pompliano", "confidence": 0.92, "justification": "Matches intro and host context"},\n'
            '      {"name": "UNKNOWN", "confidence": 0.2, "justification": "Low evidence alternative"}\n'
            '    ]\n'
            '  }\n'
            ']'
        )
        return (
            f"{instructions}\n\n"
            f"Output JSON MUST conform to the above and resemble this example (structure only, not content):\n{example}\n\n"
            f"Episode Context:\n{context_summary}\n\n"
            f"Candidate Names: {candidates}\n\n"
            f"Speaker Evidence:\n{speaker_section}\n"
        )

    def _json_schema(self) -> Dict[str, Any]:
        # JSON Schema enforcing the expected output structure
        return {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["speaker_id", "proposed_names"],
                "properties": {
                    "speaker_id": {"type": "string"},
                    "proposed_names": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["name", "confidence", "justification"],
                            "properties": {
                                "name": {"type": "string"},
                                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "justification": {"type": "string"},
                            },
                        },
                    },
                },
            },
        }

    _UNQUOTED_SPEAKER_ID = re.compile(r'("speaker_id"\s*:\s*)([A-Za-z0-9_:-]+)(\s*[,}])')

    def _parse_assignments(self, raw_output: str) -> Dict[str, SpeakerNameGuesses]:
        payload = self._strip_code_fences(raw_output.strip())
        data = self._loads_with_recovery(payload)
        if data is None:
            self.logger.error("Failed to parse LLM output: %s", payload)
            return {}

        if isinstance(data, dict):
            data = [data]

        if not isinstance(data, list):
            self.logger.error("Unexpected LLM payload (expected list): %s", payload)
            return {}

        guesses: Dict[str, SpeakerNameGuesses] = {}
        for item in data:
            try:
                speaker_id = str(item["speaker_id"])
            except (TypeError, ValueError, KeyError) as exc:
                self.logger.warning("Skipping malformed assignment %s: %s", item, exc)
                continue

            guess = guesses.setdefault(speaker_id, SpeakerNameGuesses(speaker_id=speaker_id))
            proposals = item.get("proposed_names")
            if not proposals:
                proposals = [
                    {
                        "name": item.get("proposed_name", "UNKNOWN"),
                        "confidence": item.get("confidence", 0.0),
                        "justification": item.get("justification", ""),
                    }
                ]

            if not isinstance(proposals, list):
                proposals = [proposals]

            for proposal in proposals:
                try:
                    assignment = SpeakerAssignment(
                        speaker_id=speaker_id,
                        proposed_name=str(proposal.get("name", "UNKNOWN")).strip() or "UNKNOWN",
                        confidence=float(proposal.get("confidence", 0.0)),
                        justification=str(proposal.get("justification", "")).strip(),
                    )
                except (TypeError, ValueError) as exc:
                    self.logger.warning("Skipping malformed proposal %s: %s", proposal, exc)
                    continue
                guess.add_proposal(assignment)

        return guesses

    def _strip_code_fences(self, text: str) -> str:
        if not text.startswith("```"):
            return text
        lines = text.splitlines()
        if lines:
            lines.pop(0)
        while lines and lines[-1].strip().startswith("```"):
            lines.pop()
        return "\n".join(lines).strip()

    def _loads_with_recovery(self, payload: str) -> Optional[Any]:
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            repaired = self._repair_unquoted_identifiers(payload)
            if repaired != payload:
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    pass
        return None

    def _repair_unquoted_identifiers(self, payload: str) -> str:
        def repl(match: "re.Match[str]") -> str:  # type: ignore[name-defined]
            prefix, value, suffix = match.groups()
            if value.startswith('"') and value.endswith('"'):
                return match.group(0)
            return f'{prefix}"{value.strip()}"{suffix}'

        return self._UNQUOTED_SPEAKER_ID.sub(repl, payload)


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
        engine_client = client or OllamaClient(config.ollama_model, json_mode=getattr(config, "ollama_json_mode", True))
        inference_engine = SpeakerInferenceEngine(config, engine_client)
        return cls(podcast=podcast, config=config, inference_engine=inference_engine)

    def assign_from_readable(
        self,
        readable_path: Path,
        metadata: Optional[Dict[str, str]] = None,
        yield_progress: bool = False,
    ) -> List[TranscriptSegment] | Generator[PipelineEvent, None, List[TranscriptSegment]]:
        """
        Accept a prettified transcript (readable .txt) and infer real speaker names.
        This parses lines of the form '<label> (<speaker_id>): <text>' to reconstruct
        segments grouped by speaker_id, performs name inference, persists the enriched
        JSON assignment artefact, and updates the readable transcript labels with the
        inferred names.
        """
        pipeline = self._assign_from_readable_pipeline(readable_path, metadata or {})
        if yield_progress:
            return pipeline
        try:
            while True:
                next(pipeline)
        except StopIteration as stop:
            return stop.value

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
        episode_key = self.podcast.episode_key
        assignment_path = self.config.assignment_path(self.podcast, episode_key)
        start_time = time.perf_counter()
        self.logger.info(
            "Speaker assignment started | episode=%s | segments=%d | metadata_keys=%s | config=%s",
            self.podcast.episode_id,
            len(segments),
            sorted(metadata.keys()),
            self._config_snapshot(),
        )

        yield PipelineEvent(
            stage="start",
            step_name="assign",
            episode_id=self.podcast.episode_id,
            message=f"Assigning speaker names for {self.podcast.episode_title}",
            payload={
                "artefact_key": episode_key,
                "episode_key": episode_key,
                "segment_count": len(segments),
                "step": "started",
            },
            artefact_paths={"assignment": assignment_path},
            checkpoint={
                "status": "started",
                "step": "assign",
                "artefact_key": episode_key,
                "episode_key": episode_key,
            },
            elapsed=0.0,
        )

        profile_builder = SpeakerProfileBuilder()
        profiles = profile_builder.build(
            segments,
            sample_start=self.config.sample_utterances_start,
            sample_end=self.config.sample_utterances_end,
        )
        self.logger.info(
            "Constructed %d speaker profiles: %s",
            len(profiles),
            sorted(profiles.keys()),
        )

        elapsed = time.perf_counter() - start_time
        yield PipelineEvent(
            stage="profiles_ready",
            step_name="assign",
            episode_id=self.podcast.episode_id,
            message="Constructed speaker profiles",
            payload={
                "profiles": len(profiles),
                "step": "profiles_ready",
            },
            checkpoint={
                "status": "profiles_ready",
                "step": "assign",
                "artefact_key": episode_key,
                "episode_key": episode_key,
            },
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
        roster_sample = roster[:15]
        roster_log = roster_sample + (["..."] if len(roster) > len(roster_sample) else [])
        self.logger.info(
            "Candidate roster size=%d sample=%s",
            len(roster),
            roster_log,
        )

        context_summary = self._context_summary(combined_metadata)
        target_speakers = list(profiles.keys())
        if not target_speakers:
            return []

        guesses = self.inference_engine.infer(
            profiles=profiles,
            roster=roster,
            context_summary=context_summary,
            target_speakers=target_speakers,
        )
        self.logger.info("Inference guesses: %s", self._guess_snapshot(guesses))

        best_assignments = self.inference_engine.select_best(guesses)
        assignment_map = self.inference_engine.consolidate({}, best_assignments.values())
        assignment_map = self._resolve_name_conflicts(
            assignment_map,
            guesses,
            profiles,
            roster,
            context_summary,
            segments,
        )
        critic_flags = self.inference_engine.critic(list(assignment_map.values()))
        self.logger.info(
            "Inference assignments: %s",
            self._assignment_snapshot(assignment_map.values()),
        )

        elapsed = time.perf_counter() - start_time
        yield PipelineEvent(
            stage="inference_round",
            step_name="assign",
            episode_id=self.podcast.episode_id,
            message="Completed initial inference",
            payload={
                "assignments": len(guesses),
                "step": "inference_round",
            },
            checkpoint={
                "status": "inference_round",
                "step": "assign",
                "artefact_key": episode_key,
                "episode_key": episode_key,
            },
            elapsed=elapsed,
        )

        unresolved = [
            speaker_id
            for speaker_id, assignment in assignment_map.items()
            if not assignment.is_high_confidence(self.config.high_confidence_threshold)
            or not critic_flags.get(speaker_id, True)
        ]

        if unresolved and self.config.max_iterations > 1:
            refined_guesses = self.inference_engine.infer(
                profiles=profiles,
                roster=roster,
                context_summary=context_summary + "\nFocus on uncertain speakers only.",
                target_speakers=unresolved,
            )
            self.logger.info("Refinement guesses: %s", self._guess_snapshot(refined_guesses))
            best_refined = self.inference_engine.select_best(refined_guesses)
            assignment_map = self.inference_engine.consolidate(assignment_map, best_refined.values())
            guesses.update(refined_guesses)
            assignment_map = self._resolve_name_conflicts(
                assignment_map,
                guesses,
                profiles,
                roster,
                context_summary,
                segments,
            )
            critic_flags = self.inference_engine.critic(list(assignment_map.values()))
            self.logger.info(
                "Refinement assignments: %s",
                self._assignment_snapshot(assignment_map.values()),
            )

            elapsed = time.perf_counter() - start_time
            yield PipelineEvent(
                stage="refinement_round",
                step_name="assign",
                episode_id=self.podcast.episode_id,
                message="Completed refinement inference",
                payload={
                "assignments": len(refined_guesses),
                    "step": "refinement_round",
                },
                checkpoint={
                    "status": "refinement_round",
                    "step": "assign",
                    "artefact_key": episode_key,
                    "episode_key": episode_key,
                },
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
        self.logger.info(
            "Final speaker assignments: %s",
            self._assignment_snapshot(final_assignments.values()),
        )

        enriched_segments = self._label_segments(segments, final_assignments)
        self._persist_assignments(assignment_path, enriched_segments)
        self._last_assignment_path = assignment_path
        self._last_episode_key = episode_key
        self._last_artefact_key = episode_key
        named_segments = sum(
            1 for seg in enriched_segments if seg.speaker_name and seg.speaker_name.strip().upper() != "UNKNOWN"
        )
        self.logger.info(
            "Persisted assignments | path=%s | segments=%d | named=%d | unknown=%d",
            assignment_path,
            len(enriched_segments),
            named_segments,
            len(enriched_segments) - named_segments,
        )

        elapsed = time.perf_counter() - start_time
        yield PipelineEvent(
            stage="persisted",
            step_name="assign",
            episode_id=self.podcast.episode_id,
            message="Persisted speaker assignments",
            payload={
                "path": str(assignment_path),
                "segments": len(enriched_segments),
                "artefact_key": episode_key,
                "episode_key": episode_key,
                "step": "completed",
            },
            artefact_paths={"assignment": assignment_path},
            checkpoint={
                "status": "completed",
                "step": "assign",
                "artefact_key": episode_key,
                "episode_key": episode_key,
                "assignment_path": str(assignment_path),
                "segments": len(enriched_segments),
            },
            elapsed=elapsed,
        )

        return enriched_segments

    def _assign_from_readable_pipeline(
        self,
        readable_path: Path,
        metadata: Dict[str, str],
    ) -> Generator[PipelineEvent, None, List[TranscriptSegment]]:
        episode_key = self.podcast.episode_key
        start_time = time.perf_counter()
        self.logger.info(
            "Speaker assignment (from readable) started | episode=%s | readable=%s | config=%s",
            self.podcast.episode_id,
            readable_path,
            self._config_snapshot(),
        )

        yield PipelineEvent(
            stage="start",
            step_name="assign",
            episode_id=self.podcast.episode_id,
            message=f"Parsing readable transcript for {self.podcast.episode_title}",
            payload={
                "episode_key": episode_key,
                "readable_path": str(readable_path),
                "step": "started",
            },
            checkpoint={
                "status": "started",
                "step": "assign",
                "episode_key": episode_key,
                "readable_path": str(readable_path),
            },
        )

        segments = self._segments_from_readable(readable_path)
        # Delegate to the core pipeline for inference + persistence
        generator = self._assign_pipeline(segments, metadata)
        enriched: Optional[List[TranscriptSegment]] = None
        try:
            while True:
                event = next(generator)
                yield event
        except StopIteration as stop:
            enriched = stop.value

        # Update readable labels with inferred names
        id_to_name: Dict[str, str] = {}
        for seg in enriched or []:
            sid = (seg.speaker_id or "").strip()
            name = (seg.speaker_name or "").strip()
            if sid and name:
                id_to_name[sid] = name
        if id_to_name:
            self._rewrite_readable_labels(readable_path, id_to_name)
            self.logger.info("Updated readable transcript labels with assigned names | %s", readable_path)
            elapsed = time.perf_counter() - start_time
            yield PipelineEvent(
                stage="assign",
                step_name="assign",
                episode_id=self.podcast.episode_id,
                message="Updated readable transcript with inferred speaker names",
                payload={
                    "readable_path": str(readable_path),
                    "step": "readable_updated",
                },
                checkpoint={
                    "status": "readable_updated",
                    "step": "assign",
                    "episode_key": episode_key,
                    "readable_path": str(readable_path),
                },
                elapsed=elapsed,
            )

        return enriched or []

    def _segments_from_readable(self, path: Path) -> List[TranscriptSegment]:
        """
        Parse a prettified transcript into approximate segments by speaker_id.
        Each paragraph is treated as a segment; timestamps are not available.
        """
        lines = path.read_text(encoding="utf-8").splitlines()
        segments: List[TranscriptSegment] = []
        t = 0.0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            match = re.match(r"^(.*?)\s*\(([^)]+)\):\s*(.+)$", line)
            if not match:
                continue
            _label, speaker_id, text = match.groups()
            # Create a pseudo-duration to preserve ordering
            start = t
            duration = max(1.0, min(10.0, len(text) / 50.0))
            end = start + duration
            t = end
            segments.append(
                TranscriptSegment(
                    start=start,
                    end=end,
                    text=text.strip(),
                    speaker_id=speaker_id.strip(),
                    speaker_name=speaker_id.strip(),
                    confidence=None,
                    justification=None,
                    metadata={},
                )
            )
        return segments

    def _rewrite_readable_labels(self, path: Path, id_to_name: Dict[str, str]) -> None:
        raw = path.read_text(encoding="utf-8")
        def replace_label(match: "re.Match[str]") -> str:  # type: ignore[name-defined]
            label, speaker_id = match.group(1), match.group(2)
            new_name = id_to_name.get(speaker_id, label).strip() or label
            return f"{new_name} ({speaker_id}):"
        updated = re.sub(r"^(.*?)\s*\(([^)]+)\):", replace_label, raw, flags=re.MULTILINE)
        path.write_text(updated, encoding="utf-8")

    def _config_snapshot(self) -> Dict[str, Any]:
        return {
            "ollama_model": self.config.ollama_model,
            "max_iterations": self.config.max_iterations,
            "high_confidence_threshold": self.config.high_confidence_threshold,
            "spacy_model": self.config.spacy_model,
            "sample_range": (
                self.config.sample_utterances_start,
                self.config.sample_utterances_end,
            ),
            "output_dir": str(self.config.output_dir),
            "data_root": str(self.config.data_root),
        }

    def _assignment_snapshot(
        self,
        assignments: Iterable[SpeakerAssignment],
    ) -> List[Dict[str, Any]]:
        snapshot: List[Dict[str, Any]] = []
        for assignment in assignments:
            snapshot.append(
                {
                    "speaker_id": assignment.speaker_id,
                    "name": assignment.proposed_name,
                    "confidence": round(assignment.confidence, 3),
                }
            )
        return snapshot

    def _guess_snapshot(self, guesses: Dict[str, SpeakerNameGuesses]) -> List[Dict[str, Any]]:
        snapshot: List[Dict[str, Any]] = []
        for speaker_id in sorted(guesses.keys()):
            guess = guesses[speaker_id]
            snapshot.append(
                {
                    "speaker_id": speaker_id,
                    "proposed_names": [
                        {
                            "name": proposal.proposed_name,
                            "confidence": round(proposal.confidence, 3),
                        }
                        for proposal in guess.proposed_names
                    ],
                }
            )
        return snapshot

    def _context_summary(self, metadata: Dict[str, str]) -> str:
        description = (
            metadata.get("description")
            or metadata.get("episode_description")
            or (self.podcast.description or "")
        )
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

    def _resolve_name_conflicts(
        self,
        assignments: Dict[str, SpeakerAssignment],
        guesses: Dict[str, SpeakerNameGuesses],
        profiles: Dict[str, SpeakerProfile],
        roster: Sequence[str],
        context_summary: str,
        segments: Sequence[TranscriptSegment],
    ) -> Dict[str, SpeakerAssignment]:
        if not assignments:
            return assignments

        updated = dict(assignments)
        # Attempt resolution until no further progress is made.
        while True:
            groups = self._group_assignments_by_name(updated)
            progress = False
            for name, speaker_ids in groups.items():
                if len(speaker_ids) < 2 or name == "unknown":
                    continue

                ranked = sorted(
                    (updated[speaker_id] for speaker_id in speaker_ids),
                    key=lambda assignment: assignment.confidence,
                    reverse=True,
                )
                top_conf = ranked[0].confidence
                tied = [assignment for assignment in ranked if abs(assignment.confidence - top_conf) < 1e-6]

                if len(tied) == 1:
                    continue

                tie_ids = [assignment.speaker_id for assignment in tied]
                self.logger.warning(
                    "Name conflict detected for '%s' among %s (confidence %.3f); attempting tie-break.",
                    name,
                    tie_ids,
                    top_conf,
                )
                tie_updates = self._tie_break_with_segments(
                    speaker_ids=tie_ids,
                    segments=segments,
                    roster=roster,
                    context_summary=context_summary,
                    profiles=profiles,
                )
                if tie_updates:
                    for speaker_id, assignment in tie_updates.items():
                        updated[speaker_id] = assignment
                        guesses.setdefault(speaker_id, SpeakerNameGuesses(speaker_id=speaker_id)).add_proposal(assignment)
                    progress = True
                    break

                self._assign_random_unique_names(tie_ids, updated, guesses)
                progress = True
                break

            if not progress:
                break

        return updated

    def _group_assignments_by_name(self, assignments: Dict[str, SpeakerAssignment]) -> Dict[str, List[str]]:
        grouped: Dict[str, List[str]] = defaultdict(list)
        for speaker_id, assignment in assignments.items():
            grouped[assignment.proposed_name.strip().lower()].append(speaker_id)
        return grouped

    def _tie_break_with_segments(
        self,
        speaker_ids: Sequence[str],
        segments: Sequence[TranscriptSegment],
        roster: Sequence[str],
        context_summary: str,
        profiles: Dict[str, SpeakerProfile],
    ) -> Dict[str, SpeakerAssignment]:
        window = self._extract_overlap_segments(segments, speaker_ids)
        if not window:
            return {}

        tie_context = self._format_segments_for_context(window)
        updated_context = f"{context_summary}\nTie-break evidence:\n{tie_context}"
        tie_guesses = self.inference_engine.infer(
            profiles=profiles,
            roster=roster,
            context_summary=updated_context,
            target_speakers=speaker_ids,
        )
        self.logger.info("Tie-break guesses for %s: %s", speaker_ids, self._guess_snapshot(tie_guesses))
        best = self.inference_engine.select_best(tie_guesses)
        if len(best) < len(speaker_ids):
            return {}

        normalized = {assignment.proposed_name.strip().lower() for assignment in best.values()}
        if len(normalized) != len(best):
            return {}

        return best

    def _extract_overlap_segments(
        self,
        segments: Sequence[TranscriptSegment],
        speaker_ids: Sequence[str],
        window: int = 25,
    ) -> List[TranscriptSegment]:
        targets = set(filter(None, speaker_ids))
        if len(targets) <= 1:
            return []

        if not segments:
            return []

        preferred_start = max(0, self.config.sample_utterances_start)
        search_start = preferred_start if preferred_start < len(segments) else 0

        for start in range(search_start, len(segments)):
            end = min(len(segments), start + window)
            window_segments = segments[start:end]
            present = {seg.speaker_id for seg in window_segments if seg.speaker_id in targets}
            if present == targets:
                return window_segments

        return []

    def _format_segments_for_context(self, segments: Sequence[TranscriptSegment]) -> str:
        lines = []
        for seg in segments:
            speaker = seg.speaker_id or "UNKNOWN"
            lines.append(f"[{seg.start:.2f}-{seg.end:.2f}] {speaker}: {seg.text}")
        return "\n".join(lines)

    def _assign_random_unique_names(
        self,
        speaker_ids: Sequence[str],
        assignments: Dict[str, SpeakerAssignment],
        guesses: Dict[str, SpeakerNameGuesses],
    ) -> None:
        rng = random.Random(self.podcast.episode_id)
        shuffled = list(speaker_ids)
        rng.shuffle(shuffled)
        used_names = {
            assignment.proposed_name.strip().lower()
            for assignment in assignments.values()
            if assignment.proposed_name
        }
        conflict_names = {
            assignments[speaker_id].proposed_name.strip().lower()
            for speaker_id in speaker_ids
            if assignments[speaker_id].proposed_name
        }
        for name in conflict_names:
            used_names.discard(name)

        for speaker_id in shuffled:
            guess = guesses.get(speaker_id)
            candidates = guess.proposed_names if guess else [assignments[speaker_id]]
            selected: Optional[SpeakerAssignment] = None
            for candidate in candidates:
                normalized = candidate.proposed_name.strip().lower()
                if normalized and normalized != "unknown" and normalized not in used_names:
                    selected = SpeakerAssignment(
                        speaker_id=speaker_id,
                        proposed_name=candidate.proposed_name,
                        confidence=candidate.confidence,
                        justification=candidate.justification,
                    )
                    used_names.add(normalized)
                    break

            if selected is None:
                fallback_name = f"UNKNOWN_{speaker_id}"
                selected = SpeakerAssignment(
                    speaker_id=speaker_id,
                    proposed_name=fallback_name,
                    confidence=assignments[speaker_id].confidence,
                    justification="Randomized unique fallback due to irreconcilable tie.",
                )
                used_names.add(fallback_name.lower())

            assignments[speaker_id] = selected
            guesses.setdefault(speaker_id, SpeakerNameGuesses(speaker_id=speaker_id)).add_proposal(selected)

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
                "speaker_id": seg.speaker_id,
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

