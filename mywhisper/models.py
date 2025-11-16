"""
Shared domain models and dataclasses for mywhisper pipelines.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, TYPE_CHECKING

from .config import derive_episode_key, generate_artefact_key

if TYPE_CHECKING:
    import torch

__all__ = [
    "PipelineEvent",
    "PodcastEpisode",
    "TranscriptSegment",
    "SpeakerProfile",
    "SpeakerAssignment",
    "SpeakerNameGuesses",
    "AudioChunk",
    "DiarizedTurn",
]


@dataclass(slots=True)
class PipelineEvent:
    """Structured progress event emitted by generator-driven pipelines."""

    stage: str
    message: str
    step_name: Optional[str] = None
    episode_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    artefact_paths: Dict[str, Path] = field(default_factory=dict)
    checkpoint: Dict[str, Any] = field(default_factory=dict)
    elapsed: Optional[float] = None
    transient: bool = False


@dataclass(slots=True)
class PodcastEpisode:
    """Representation of a podcast episode tracked in the catalog."""

    episode_id: str
    show_title: str
    episode_title: str
    source_path: Path
    description: Optional[str] = None
    duration_sec: Optional[float] = None
    published_at: Optional[_dt.datetime] = None
    author: Optional[str] = None
    guid: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def artefact_slug(self) -> str:
        """Return a filesystem-safe slug derived from show and episode titles."""
        base = f"{self.show_title}__{self.episode_title}".lower()
        slug = "".join(ch if ch.isalnum() else "_" for ch in base)
        return "_".join(filter(None, slug.split("_")))

    @property
    def episode_key(self) -> str:
        """Return the deterministic eight-digit key for this episode."""
        meta = self.metadata
        existing = None
        if isinstance(meta, dict):
            value = meta.get("episode_key")
            if isinstance(value, str) and len(value) == 8 and value.isdigit():
                existing = value
        if existing:
            return existing

        key = derive_episode_key(self.episode_id)
        if isinstance(meta, dict):
            meta["episode_key"] = key
        return key


@dataclass(slots=True)
class TranscriptSegment:
    """One segment of a transcript."""

    start: float
    end: float
    text: str
    speaker_id: Optional[str] = None
    speaker_name: Optional[str] = None
    confidence: Optional[float] = None
    justification: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(slots=True)
class SpeakerProfile:
    """Aggregated statistics about a diarized speaker."""

    speaker_id: str
    total_duration: float = 0.0
    total_turns: int = 0
    first_start: float = float("inf")
    last_end: float = 0.0
    snippets: List[str] = field(default_factory=list)
    sample_quotes: List[str] = field(default_factory=list)

    def update_from_segments(
        self,
        segments: Sequence[TranscriptSegment],
        sample_start: int,
        sample_end: int,
    ) -> None:
        """
        Populate the profile statistics from transcript segments.
        """

        if not segments:
            return

        self.total_turns = len(segments)
        self.total_duration = sum(seg.duration() for seg in segments)
        self.first_start = min(seg.start for seg in segments)
        self.last_end = max(seg.end for seg in segments)
        self.snippets = [seg.text for seg in segments]

        self.sample_quotes = [
            seg.text for seg in segments[:sample_start]
        ] + [seg.text for seg in segments[-sample_end:]]

    def to_prompt_block(self) -> str:
        """
        Render the profile as a text block for LLM prompting.
        """

        stats = (
            f"speaker_id: {self.speaker_id}\n"
            f"total_duration_sec: {self.total_duration:.1f}\n"
            f"turn_count: {self.total_turns}\n"
            f"first_start_sec: {self.first_start:.1f}\n"
            f"last_end_sec: {self.last_end:.1f}\n"
        )
        quotes = "\n".join(f"- \"{quote}\"" for quote in self.sample_quotes if quote)
        return f"{stats}sample_quotes:\n{quotes}\n"


@dataclass(slots=True)
class SpeakerAssignment:
    """Speaker name assignment returned from LLM inference."""

    speaker_id: str
    proposed_name: str
    confidence: float
    justification: str = ""

    def is_high_confidence(self, threshold: float) -> bool:
        return self.confidence >= threshold


@dataclass(slots=True)
class SpeakerNameGuesses:
    """Container for per-speaker name proposals."""

    speaker_id: str
    proposed_names: List[SpeakerAssignment] = field(default_factory=list)

    def add_proposal(self, proposal: SpeakerAssignment) -> None:
        """
        Add or merge a name proposal for this speaker, keeping the highest-confidence
        instance per normalized name and maintaining descending confidence order.
        """

        normalized = proposal.proposed_name.strip().lower()
        existing_index = next(
            (
                idx
                for idx, candidate in enumerate(self.proposed_names)
                if candidate.proposed_name.strip().lower() == normalized
            ),
            None,
        )
        if existing_index is not None:
            if proposal.confidence > self.proposed_names[existing_index].confidence:
                self.proposed_names[existing_index] = proposal
        else:
            self.proposed_names.append(proposal)

        self.proposed_names.sort(key=lambda candidate: candidate.confidence, reverse=True)

    def best(self) -> Optional[SpeakerAssignment]:
        """Return the highest-confidence proposal, if any."""

        return self.proposed_names[0] if self.proposed_names else None


@dataclass(slots=True)
class AudioChunk:
    """Temporary representation of an audio chunk."""

    path: Path
    global_start: float
    global_end: float
    tensor: Optional["torch.Tensor"] = None  # type: ignore[name-defined]
    sample_rate: Optional[int] = None
    artefact_key: str = field(default_factory=generate_artefact_key)

    def duration(self) -> float:
        return max(0.0, self.global_end - self.global_start)


@dataclass(slots=True)
class DiarizedTurn:
    """Diarization result entry."""

    start: float
    end: float
    speaker_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


