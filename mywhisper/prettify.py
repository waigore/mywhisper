from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Sequence

from .config import ensure_episode_subdir, resolve_data_root
from .logging_utils import LoggingBase
from .models import PipelineEvent, PodcastEpisode, TranscriptSegment
from .podcasts import PodcastCatalog

LOGGER = logging.getLogger("mywhisper.prettify")


@dataclass(slots=True)
class PrettifyConfig:
    """
    Configuration for generating readable transcripts.
    """

    data_root: Path = field(default_factory=resolve_data_root)
    collapse_gap_seconds: float = 1.5
    max_block_characters: Optional[int] = None
    output_subdir: str = "transcripts"

    def condensed_path(self, podcast: PodcastEpisode, episode_key: Optional[str] = None) -> Path:
        key = episode_key or podcast.episode_key
        directory = ensure_episode_subdir(key, self.data_root, self.output_subdir)
        return directory / f"{key}_condensed.json"

    def assignment_path(self, podcast: PodcastEpisode, episode_key: Optional[str] = None) -> Path:
        key = episode_key or podcast.episode_key
        directory = ensure_episode_subdir(key, self.data_root, self.output_subdir)
        return directory / f"{key}_with_names.json"

    def readable_path(self, podcast: PodcastEpisode, episode_key: Optional[str] = None) -> Path:
        key = episode_key or podcast.episode_key
        directory = ensure_episode_subdir(key, self.data_root, self.output_subdir)
        return directory / f"{key}_readable.txt"


class TranscriptPrettifier(LoggingBase):
    """
    Collapse diarized segments into readable blocks and persist text artefacts.
    """

    def __init__(
        self,
        podcast: PodcastEpisode,
        config: Optional[PrettifyConfig] = None,
        catalog: Optional[PodcastCatalog] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.podcast = podcast
        self.config = config or PrettifyConfig()
        self.catalog = catalog
        base_logger = logger or LOGGER
        self.logger = base_logger.getChild(podcast.episode_id)
        self._last_assignment_path: Optional[Path] = None
        self._last_readable_path: Optional[Path] = None
        self._last_condensed_path: Optional[Path] = None

    def get_outputs(self) -> Dict[str, Optional[Path]]:
        """
        Get all outputs from prettify execution.
        
        Returns:
            Dictionary with readable_path and condensed_path keys.
        """
        return {
            "readable_path": self._last_readable_path,
            "condensed_path": self._last_condensed_path,
        }

    def prettify(
        self,
        *,
        assignment_path: Optional[Path] = None,
        yield_progress: bool = False,
    ) -> Dict[str, Path] | Generator[PipelineEvent, None, Dict[str, Path]]:
        pipeline = self._pipeline(assignment_path=assignment_path)
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

    def _pipeline(
        self,
        assignment_path: Optional[Path],
    ) -> Generator[PipelineEvent, None, Dict[str, Path]]:
        episode_key = self.podcast.episode_key
        resolved_assignment = assignment_path or self.config.assignment_path(self.podcast, episode_key)
        resolved_assignment = resolved_assignment.resolve()
        readable_path = self.config.readable_path(self.podcast, episode_key).resolve()
        condensed_path = self.config.condensed_path(self.podcast, episode_key).resolve()

        if not resolved_assignment.exists():
            raise FileNotFoundError(f"Assigned transcript not found at {resolved_assignment}")

        start_time = time.perf_counter()
        self.logger.info(
            "Prettify start | episode=%s | assignment_path=%s | output=%s",
            self.podcast.episode_id,
            resolved_assignment,
            readable_path,
        )

        yield PipelineEvent(
            stage="prettify",
            step_name="prettify",
            episode_id=self.podcast.episode_id,
            message="Loading assigned transcript",
            payload={
                "episode_key": episode_key,
                "assignment_path": str(resolved_assignment),
                "step": "load",
            },
            artefact_paths={"assignment": resolved_assignment},
            checkpoint={
                "status": "started",
                "step": "prettify",
                "assignment_path": str(resolved_assignment),
                "episode_key": episode_key,
            },
        )

        segments = self._load_segments(resolved_assignment)
        blocks = self._collapse_segments(segments)

        yield PipelineEvent(
            stage="prettify",
            step_name="prettify",
            episode_id=self.podcast.episode_id,
            message="Collapsed transcript segments",
            payload={
                "blocks": len(blocks),
                "segments": len(segments),
                "step": "collapse",
            },
            checkpoint={
                "status": "collapsed",
                "step": "prettify",
                "segments": len(segments),
                "blocks": len(blocks),
            },
        )

        # Persist condensed JSON (collapsed blocks)
        condensed_records = [
            {
                "start": float(block.get("start", 0.0)),
                "end": float(block.get("end", 0.0)),
                "speaker_id": block.get("speaker_id") or "UNKNOWN",
                "speaker_name": block.get("speaker_name") or (block.get("speaker_id") or "UNKNOWN"),
                "text": " ".join(part.strip() for part in (block.get("texts") or []) if str(part).strip()),
            }
            for block in blocks
            if block.get("texts")
        ]
        condensed_path.parent.mkdir(parents=True, exist_ok=True)
        condensed_path.write_text(json.dumps(condensed_records, ensure_ascii=False, indent=2), encoding="utf-8")
        self.logger.info(
            "Persisted condensed transcript | path=%s | blocks=%d",
            condensed_path,
            len(condensed_records),
        )

        # Persist readable text
        readable_text = self._format_blocks(blocks)
        readable_path.parent.mkdir(parents=True, exist_ok=True)
        readable_path.write_text(readable_text, encoding="utf-8")
        self.logger.info(
            "Persisted readable transcript | path=%s | blocks=%d",
            readable_path,
            len(blocks),
        )

        # Record artefacts (readable first for backward-compat tests)
        artefact_key = f"{episode_key}_readable"
        if self.catalog:
            self.catalog.record_artefact(
                episode_id=self.podcast.episode_id,
                kind="readable_transcript",
                path=readable_path,
                artefact_key=artefact_key,
            )
            self.catalog.record_artefact(
                episode_id=self.podcast.episode_id,
                kind="condensed_transcript",
                path=condensed_path,
                artefact_key=f"{episode_key}_condensed",
            )

        elapsed = time.perf_counter() - start_time
        yield PipelineEvent(
            stage="prettify",
            step_name="prettify",
            episode_id=self.podcast.episode_id,
            message="Persisted condensed and readable transcripts",
            payload={
                "path": str(readable_path),
                "blocks": len(blocks),
                "step": "completed",
            },
            artefact_paths={"readable": readable_path, "condensed": condensed_path},
            checkpoint={
                "status": "completed",
                "step": "prettify",
                "readable_path": str(readable_path),
                "condensed_path": str(condensed_path),
                "blocks": len(blocks),
                "assignment_path": str(resolved_assignment),
                "episode_key": episode_key,
            },
            elapsed=elapsed,
        )

        self._last_assignment_path = resolved_assignment
        self._last_readable_path = readable_path
        self._last_condensed_path = condensed_path
        return {"readable_path": readable_path, "condensed_path": condensed_path}

    def _load_segments(self, path: Path) -> List[TranscriptSegment]:
        records = json.loads(path.read_text(encoding="utf-8"))
        segments: List[TranscriptSegment] = []
        for record in records:
            try:
                start = float(record.get("start", 0.0))
                end = float(record.get("end", start))
                text = str(record.get("text", "")).strip()
            except (TypeError, ValueError):
                continue
            if not text:
                continue

            speaker_id = str(record.get("speaker_id") or record.get("speaker") or "UNKNOWN").strip()
            speaker_name = str(record.get("speaker_name") or speaker_id or "UNKNOWN").strip()
            segments.append(
                TranscriptSegment(
                    start=start,
                    end=end,
                    text=text,
                    speaker_id=speaker_id or "UNKNOWN",
                    speaker_name=speaker_name or speaker_id or "UNKNOWN",
                    confidence=record.get("confidence"),
                    justification=record.get("justification"),
                    metadata=record.get("metadata", {}),
                )
            )
        segments.sort(key=lambda seg: (seg.start, seg.end))
        return segments

    def _collapse_segments(self, segments: Sequence[TranscriptSegment]) -> List[dict]:
        blocks: List[dict] = []
        current: Optional[dict] = None

        for seg in segments:
            speaker_id = (seg.speaker_id or "UNKNOWN").strip() or "UNKNOWN"
            speaker_name = (seg.speaker_name or speaker_id).strip() or speaker_id
            text = seg.text.strip()
            if not text:
                continue

            if current and self._can_merge(current, seg, text):
                previous_count = len(current["texts"])
                current["texts"].append(text)
                current["end"] = seg.end
                current["char_count"] += len(text) + (1 if previous_count else 0)
            else:
                if current:
                    blocks.append(current)
                current = {
                    "speaker_id": speaker_id,
                    "speaker_name": speaker_name,
                    "texts": [text],
                    "start": seg.start,
                    "end": seg.end,
                    "char_count": len(text),
                }

        if current:
            blocks.append(current)
        return blocks

    def _can_merge(self, current: dict, segment: TranscriptSegment, text: str) -> bool:
        segment_id = (segment.speaker_id or "UNKNOWN").strip() or "UNKNOWN"
        if not current["speaker_id"] or current["speaker_id"] != segment_id:
            return False

        gap_limit = self.config.collapse_gap_seconds
        if gap_limit is not None:
            gap = max(0.0, segment.start - float(current["end"]))
            if gap > gap_limit:
                return False

        threshold = self.config.max_block_characters
        if threshold is not None:
            projected = current["char_count"] + len(text)
            if current["texts"]:
                projected += 1
            if projected > threshold:
                return False

        return True

    def _format_blocks(self, blocks: Iterable[dict]) -> str:
        lines: List[str] = []
        for block in blocks:
            speaker_label = block["speaker_name"] or block["speaker_id"] or "UNKNOWN"
            speaker_id = block["speaker_id"] or "UNKNOWN"
            text = " ".join(part.strip() for part in block["texts"]).strip()
            if not text:
                continue
            lines.append(f"{speaker_label} ({speaker_id}): {text}")
        return "\n\n".join(lines)


