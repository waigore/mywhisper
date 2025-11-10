from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

from config import (
    ASSIGNED_TRANSCRIPT_PATH,
    HIGH_CONFIDENCE_THRESHOLD,
    INFER_MAX_ITERATIONS,
    INFER_START_OFFSET_SEC,
    MAX_TRANSCRIPT_TOKENS,
    OLLAMA_MODEL,
    PODCAST_EPISODE_DESCRIPTION,
    PODCAST_EPISODE_NAME,
    SPACY_MODEL,
    TRANSCRIPT_SAMPLE_UTTERANCES_END,
    TRANSCRIPT_SAMPLE_UTTERANCES_START,
)


LOGGER = logging.getLogger(__name__)


@dataclass
class Utterance:
    start: float
    end: float
    speaker: str
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class SpeakerProfile:
    speaker_id: str
    total_duration: float = 0.0
    total_turns: int = 0
    first_start: float = float("inf")
    last_end: float = 0.0
    snippets: List[str] = field(default_factory=list)
    sample_quotes: List[str] = field(default_factory=list)

    def to_prompt_block(self) -> str:
        stats = (
            f"total_duration_sec: {self.total_duration:.1f}\n"
            f"turn_count: {self.total_turns}\n"
            f"first_start_sec: {self.first_start:.1f}\n"
            f"last_end_sec: {self.last_end:.1f}\n"
        )
        quotes = "\n".join(f"- \"{quote}\"" for quote in self.sample_quotes)
        return f"Speaker: {self.speaker_id}\n{stats}Quotes:\n{quotes}\n"


@dataclass
class SpeakerAssignment:
    speaker_id: str
    proposed_name: str
    confidence: float
    justification: str

    def is_high_confidence(self) -> bool:
        return self.confidence >= HIGH_CONFIDENCE_THRESHOLD


def load_transcript(path: Path) -> List[Utterance]:
    with path.open("r", encoding="utf-8") as handle:
        items = json.load(handle)
    utterances = [
        Utterance(
            start=float(item["start"]),
            end=float(item["end"]),
            speaker=str(item["speaker"]),
            text=str(item["text"]).strip(),
        )
        for item in items
    ]
    utterances.sort(key=lambda u: u.start)
    return utterances


def build_profiles(
    utterances: Sequence[Utterance],
    sample_start: int,
    sample_end: int,
) -> Dict[str, SpeakerProfile]:
    profiles: Dict[str, SpeakerProfile] = {}
    per_speaker_utts: Dict[str, List[Utterance]] = defaultdict(list)
    for utt in utterances:
        per_speaker_utts[utt.speaker].append(utt)

    for speaker_id, turns in per_speaker_utts.items():
        profile = SpeakerProfile(speaker_id=speaker_id)
        profile.total_turns = len(turns)
        profile.total_duration = sum(utt.duration for utt in turns)
        profile.first_start = min(utt.start for utt in turns)
        profile.last_end = max(utt.end for utt in turns)
        profile.sample_quotes = [
            utt.text for utt in turns[:sample_start]
        ] + [utt.text for utt in turns[-sample_end:]]
        profiles[speaker_id] = profile

    return profiles


def truncate_description(description: str, max_tokens: int) -> str:
    words = description.split()
    if len(words) <= max_tokens:
        return description
    truncated = " ".join(words[:max_tokens])
    LOGGER.warning("Truncated episode description from %d to %d tokens", len(words), max_tokens)
    return truncated


def ensure_spacy_model() -> Optional[Any]:
    try:
        import spacy
    except ImportError:
        LOGGER.warning("spaCy is not installed; person extraction will be skipped.")
        return None

    try:
        return spacy.load(SPACY_MODEL)
    except OSError:
        LOGGER.warning("spaCy model '%s' not found; person extraction will be skipped.", SPACY_MODEL)
        return None


def extract_candidate_people(nlp_model: Optional[Any], text: str) -> List[str]:
    if not nlp_model:
        return []
    doc = nlp_model(text)
    unique = {ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"}
    return sorted(unique)


def default_candidate_roster(extra: Optional[Iterable[str]] = None) -> List[str]:
    roster = set(extra or [])
    roster.update(
        name.strip()
        for name in [
            PODCAST_EPISODE_NAME.split(" - ")[0],
            "Unknown Host",
            "Unknown Guest",
        ]
        if name
    )
    return sorted(roster)


def call_ollama(model: str, prompt: str, options: Optional[Dict[str, Any]] = None) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False}
    if options:
        payload["options"] = options

    response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data.get("response", "")


def build_inference_prompt(
    speaker_blocks: Sequence[str],
    candidate_names: Sequence[str],
    context_summary: str,
) -> str:
    candidates = ", ".join(candidate_names) if candidate_names else "Unknown"
    speaker_section = "\n\n".join(speaker_blocks)
    instructions = (
        "You are identifying real speaker names for anonymized diarized segments from a podcast episode.\n"
        "Use the episode context and candidate names to infer who each speaker likely is.\n"
        "Return a JSON array where each element has keys: speaker_id, proposed_name, confidence (0-1 float), justification.\n"
        "If unsure, set proposed_name to \"UNKNOWN\" and confidence <= 0.3.\n"
        "Do not include any commentary outside JSON.\n"
    )
    return (
        f"{instructions}\n"
        f"Episode Context:\n{context_summary}\n\n"
        f"Candidate Names: {candidates}\n\n"
        f"Speaker Evidence:\n{speaker_section}\n"
    )


def parse_llm_json(raw_output: str) -> List[SpeakerAssignment]:
    raw_output = raw_output.strip()
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        LOGGER.error("Failed to parse LLM output: %s\nOutput: %s", exc, raw_output)
        return []

    assignments: List[SpeakerAssignment] = []
    for item in data:
        try:
            assignments.append(
                SpeakerAssignment(
                    speaker_id=str(item["speaker_id"]),
                    proposed_name=str(item.get("proposed_name", "UNKNOWN")).strip(),
                    confidence=float(item.get("confidence", 0.0)),
                    justification=str(item.get("justification", "")).strip(),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            LOGGER.warning("Skipping malformed assignment %s: %s", item, exc)
    return assignments


def run_inference_cycle(
    profiles: Dict[str, SpeakerProfile],
    candidate_names: Sequence[str],
    context_summary: str,
    target_speakers: Sequence[str],
) -> List[SpeakerAssignment]:
    speaker_blocks = [profiles[speaker_id].to_prompt_block() for speaker_id in target_speakers]
    prompt = build_inference_prompt(speaker_blocks, candidate_names, context_summary)
    response = call_ollama(OLLAMA_MODEL, prompt)
    return parse_llm_json(response)


def critic_pass(assignments: Sequence[SpeakerAssignment]) -> Dict[str, bool]:
    """Return mapping speaker_id->is_consistent according to critic rules."""
    consistency: Dict[str, bool] = {}
    seen_names: Dict[str, str] = {}

    for assignment in assignments:
        if assignment.proposed_name.upper() == "UNKNOWN":
            consistency[assignment.speaker_id] = assignment.confidence <= HIGH_CONFIDENCE_THRESHOLD
            continue
        previous = seen_names.get(assignment.proposed_name.lower())
        if previous and previous != assignment.speaker_id:
            consistency[assignment.speaker_id] = False
            consistency[previous] = False
        else:
            seen_names[assignment.proposed_name.lower()] = assignment.speaker_id
            consistency.setdefault(assignment.speaker_id, True)
    return consistency


def consolidate_assignments(
    prior: Dict[str, SpeakerAssignment],
    new_assignments: Iterable[SpeakerAssignment],
) -> Dict[str, SpeakerAssignment]:
    updated = dict(prior)
    for assignment in new_assignments:
        current = updated.get(assignment.speaker_id)
        if current is None or assignment.confidence > current.confidence:
            updated[assignment.speaker_id] = assignment
    return updated


def prepare_context_summary() -> str:
    description = truncate_description(PODCAST_EPISODE_DESCRIPTION.strip(), MAX_TRANSCRIPT_TOKENS)
    return f"Title: {PODCAST_EPISODE_NAME}\nDescription: {description}"


def label_transcript(
    utterances: Sequence[Utterance],
    assignments: Dict[str, SpeakerAssignment],
) -> List[Dict[str, Any]]:
    enriched = []
    for utt in utterances:
        assignment = assignments.get(utt.speaker)
        enriched.append(
            {
                "start": utt.start + INFER_START_OFFSET_SEC,
                "end": utt.end + INFER_START_OFFSET_SEC,
                "speaker": utt.speaker,
                "text": utt.text,
                "speaker_name": assignment.proposed_name if assignment else "UNKNOWN",
                "confidence": assignment.confidence if assignment else 0.0,
                "justification": assignment.justification if assignment else "",
            }
        )
    return enriched


def save_assignments(output_path: Path, data: List[Dict[str, Any]]) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    LOGGER.info("Wrote %d utterances with inferred names to %s", len(data), output_path)


def inference_pipeline(
    transcript_path: Path,
    output_path: Path,
    host_roster: Optional[Sequence[str]] = None,
) -> Dict[str, SpeakerAssignment]:
    utterances = load_transcript(transcript_path)
    profiles = build_profiles(
        utterances,
        sample_start=TRANSCRIPT_SAMPLE_UTTERANCES_START,
        sample_end=TRANSCRIPT_SAMPLE_UTTERANCES_END,
    )

    nlp = ensure_spacy_model()
    candidate_people = extract_candidate_people(
        nlp,
        PODCAST_EPISODE_NAME + "\n" + PODCAST_EPISODE_DESCRIPTION,
    )
    roster = default_candidate_roster(candidate_people)
    if host_roster:
        roster = sorted(set(roster).union(host_roster))

    context_summary = prepare_context_summary()

    # Initial inference pass
    initial_assignments = run_inference_cycle(
        profiles,
        roster,
        context_summary,
        target_speakers=list(profiles.keys()),
    )
    assignment_map = consolidate_assignments({}, initial_assignments)

    # Critic pass
    consistency_flags = critic_pass(initial_assignments)

    # Single reconciliation cycle per requirements
    unresolved_speakers = [
        speaker_id
        for speaker_id, assignment in assignment_map.items()
        if not assignment.is_high_confidence() or not consistency_flags.get(speaker_id, True)
    ]

    if unresolved_speakers and INFER_MAX_ITERATIONS > 1:
        refined_assignments = run_inference_cycle(
            profiles,
            roster,
            context_summary + "\nFocus on previously uncertain speakers only.",
            target_speakers=unresolved_speakers,
        )
        assignment_map = consolidate_assignments(assignment_map, refined_assignments)

    final_assignments = {
        speaker_id: assignment
        for speaker_id, assignment in assignment_map.items()
    }

    # Force low-confidence entries to UNKNOWN
    for speaker_id, assignment in list(final_assignments.items()):
        if not assignment.is_high_confidence():
            final_assignments[speaker_id] = SpeakerAssignment(
                speaker_id=speaker_id,
                proposed_name="UNKNOWN",
                confidence=assignment.confidence,
                justification=assignment.justification,
            )

    labeled = label_transcript(utterances, final_assignments)
    save_assignments(output_path, labeled)

    return final_assignments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer speaker names using an LLM.")
    parser.add_argument(
        "--input",
        type=Path,
        default=ASSIGNED_TRANSCRIPT_PATH,
        help="Path to diarized transcript JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(ASSIGNED_TRANSCRIPT_PATH).with_name("whisper_diarization_with_names_v5.json"),
        help="Path for the enriched transcript output.",
    )
    parser.add_argument(
        "--host",
        action="append",
        dest="hosts",
        help="Optional host or recurring participant names to seed the roster. Can be repeated.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    assignments = inference_pipeline(
        transcript_path=args.input,
        output_path=args.output,
        host_roster=args.hosts,
    )

    summary = [
        {
            "speaker_id": speaker_id,
            "proposed_name": assignment.proposed_name,
            "confidence": assignment.confidence,
        }
        for speaker_id, assignment in assignments.items()
    ]
    LOGGER.info("Final assignments: %s", summary)


if __name__ == "__main__":
    main()

