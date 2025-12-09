from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Sequence

from .config import ensure_episode_subdir, resolve_data_root
from .logging_utils import LoggingBase
from .models import DiarizedTurn, PipelineEvent, PodcastEpisode, TranscriptSegment
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
        diarization_results: Optional[Any] = None,
        yield_progress: bool = False,
    ) -> Dict[str, Path] | Generator[PipelineEvent, None, Dict[str, Path]]:
        pipeline = self._pipeline(assignment_path=assignment_path, diarization_results=diarization_results)
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
        diarization_results: Optional[Any] = None,
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
        segments = self._merge_segments_by_sentences(segments)
        
        # Apply diarization labels after sentence merging
        # Diarization results are required for prettify step
        if diarization_results is None:
            raise RuntimeError(
                f"Diarization results are required for prettify step. "
                f"Episode {self.podcast.episode_id} has no diarization results. "
                f"Run diarize step first or include it in the pipeline plan."
            )
        
        # If diarization_results is a Path, verify it exists
        if isinstance(diarization_results, Path):
            if not diarization_results.exists():
                raise FileNotFoundError(
                    f"Diarization RTTM file not found: {diarization_results}. "
                    f"Episode {self.podcast.episode_id} requires diarization results. "
                    f"Run diarize step first or include it in the pipeline plan."
                )
        
        self.logger.debug(
            "Applying diarization labels | episode=%s | diarization_results=%s",
            self.podcast.episode_id,
            diarization_results,
        )
        diarized_turns = ensure_diarized_turns(diarization_results)
        self.logger.debug(
            "Loaded diarized turns | episode=%s | turns_count=%d",
            self.podcast.episode_id,
            len(diarized_turns) if diarized_turns else 0,
        )
        
        if not diarized_turns:
            raise RuntimeError(
                f"No diarization turns found in {diarization_results}. "
                f"Episode {self.podcast.episode_id} requires valid diarization results. "
                f"The RTTM file may be empty or malformed. Run diarize step again."
            )
        
        segments = apply_diarization_labels(segments, diarized_turns)
        self.logger.info(
            "Applied diarization labels | episode=%s | segments_count=%d | turns_count=%d",
            self.podcast.episode_id,
            len(segments) if segments else 0,
            len(diarized_turns),
        )
        
        blocks = self._collapse_segments(segments)
        
        # Split blocks that exceed 10 sentences
        blocks = self._split_blocks_by_sentences(blocks, max_sentences=10)

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
        condensed_records = []
        for block in blocks:
            if not block.get("texts"):
                continue
            record = {
                "start": float(block.get("start", 0.0)),
                "end": float(block.get("end", 0.0)),
                "speaker_id": block.get("speaker_id") or "UNKNOWN",
                "speaker_name": block.get("speaker_name") or (block.get("speaker_id") or "UNKNOWN"),
                "text": " ".join(part.strip() for part in (block.get("texts") or []) if str(part).strip()),
            }
            # Only include indeterminate field when True
            if block.get("indeterminate") is True:
                record["indeterminate"] = True
            condensed_records.append(record)
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
            indeterminate = record.get("indeterminate") if record.get("indeterminate") is True else None
            segments.append(
                TranscriptSegment(
                    start=start,
                    end=end,
                    text=text,
                    speaker_id=speaker_id or "UNKNOWN",
                    speaker_name=speaker_name or speaker_id or "UNKNOWN",
                    confidence=record.get("confidence"),
                    justification=record.get("justification"),
                    indeterminate=indeterminate,
                    metadata=record.get("metadata", {}),
                )
            )
        segments.sort(key=lambda seg: (seg.start, seg.end))
        return segments

    def _should_prevent_merge_by_failsafe(self, merge_group: List[TranscriptSegment]) -> bool:
        """
        Check if merging should be prevented by the 40-word failsafe rule.
        
        Args:
            merge_group: List of segments to potentially merge
            
        Returns:
            True if merging should be prevented, False otherwise
        """
        if len(merge_group) <= 1:
            return False
        
        combined_text = " ".join(seg.text.strip() for seg in merge_group)
        words = combined_text.split()
        word_count = len([w for w in words if w.strip()])
        
        if word_count <= 40:
            return False
        
        punctuation_pattern = re.compile(r"[,;.!?]")
        has_punctuation = bool(punctuation_pattern.search(combined_text))
        
        return not has_punctuation

    def _process_merge_group(
        self,
        merge_group: List[TranscriptSegment],
        first_seg: TranscriptSegment,
        merged: List[TranscriptSegment],
    ) -> None:
        """
        Process a merge group by either merging segments or adding them individually.
        
        Args:
            merge_group: List of segments to process
            first_seg: First segment in the group (for metadata)
            merged: List to append results to
        """
        if len(merge_group) == 1:
            # Only one segment, just add it
            merged.append(first_seg)
            return
        
        if self._should_prevent_merge_by_failsafe(merge_group):
            # Don't merge, add segments individually
            merged.extend(merge_group)
            return
        
        # Merge the segments
        merged_seg = self._create_merged_segment(merge_group, first_seg)
        merged.append(merged_seg)

    def _create_merged_segment(
        self, merge_group: List[TranscriptSegment], first_seg: TranscriptSegment
    ) -> TranscriptSegment:
        """
        Create a merged segment from a group of segments.
        
        Args:
            merge_group: List of segments to merge
            first_seg: First segment in the group (for metadata)
            
        Returns:
            Merged TranscriptSegment
        """
        merged_text = " ".join(seg.text.strip() for seg in merge_group)
        return TranscriptSegment(
            start=min(seg.start for seg in merge_group),
            end=max(seg.end for seg in merge_group),
            text=merged_text,
            speaker_id=first_seg.speaker_id,
            speaker_name=first_seg.speaker_name,
            confidence=first_seg.confidence,
            justification=first_seg.justification,
            metadata=dict(first_seg.metadata),
        )

    def _merge_segments_by_sentences(self, segments: Sequence[TranscriptSegment]) -> List[TranscriptSegment]:
        """
        Merge consecutive segments that belong to the same sentence.
        
        Segments that don't end with sentence-ending punctuation (., !, ?) are merged
        with following segments until a sentence boundary is found. A failsafe prevents
        merging when the combined text exceeds 40 words without any punctuation.
        
        This function only depends on segment text content and timing gaps, not on
        speaker assignments. Speaker labels are applied after merging.
        
        Args:
            segments: List of transcript segments to merge
            
        Returns:
            List of merged transcript segments
        """
        if not segments:
            return []
        
        merged: List[TranscriptSegment] = []
        i = 0
        sentence_end_pattern = re.compile(r"[.!?]\s*$")
        
        while i < len(segments):
            current_seg = segments[i]
            merge_group = [current_seg]
            
            # Check if current segment ends with sentence-ending punctuation
            if sentence_end_pattern.search(current_seg.text):
                # Segment is complete, no merging needed
                merged.append(current_seg)
                i += 1
                continue
            
            # Collect following segments until we find a sentence boundary
            # Only merge segments that are contiguous (no gap or minimal gap)
            j = i + 1
            found_boundary = False
            max_gap_seconds = self.config.collapse_gap_seconds or 1.5  # Use collapse gap or default 1.5s
            broke_due_to_constraint = False
            
            while j < len(segments):
                next_seg = segments[j]
                
                # Check constraints: don't merge across large gaps
                prev_seg = merge_group[-1]
                gap = max(0.0, next_seg.start - prev_seg.end)
                
                # Check if gap is too large - segments must be contiguous
                if gap > max_gap_seconds:
                    # Gap too large, stop collecting
                    broke_due_to_constraint = True
                    break
                
                merge_group.append(next_seg)
                
                # Check if this segment ends with sentence-ending punctuation
                if not sentence_end_pattern.search(next_seg.text):
                    # No sentence boundary yet, continue collecting
                    j += 1
                    continue
                
                # Found sentence boundary, check failsafe before merging
                found_boundary = True
                if self._should_prevent_merge_by_failsafe(merge_group):
                    # Don't merge, add segments individually
                    merged.extend(merge_group[:-1])  # Add all except the last
                    i = j  # Move to the last segment, which will be processed in next iteration
                    break
                
                # Merge the segments
                merged_seg = self._create_merged_segment(merge_group, current_seg)
                merged.append(merged_seg)
                i = j + 1
                break
            
            # If we handled a boundary case (merged or failsafe), continue to next iteration
            if found_boundary:
                continue
            
            # If we broke due to gap constraint, handle the merge_group we collected
            if broke_due_to_constraint:
                self._process_merge_group(merge_group, current_seg, merged)
                i = j  # Continue from the segment that caused the break
                continue
            
            # If we reached the end without finding a sentence boundary
            if j >= len(segments):
                self._process_merge_group(merge_group, current_seg, merged)
                break
        
        return merged

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
                # Preserve indeterminate if any segment in the block has it
                if seg.indeterminate is True:
                    current["indeterminate"] = True
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
                # Include indeterminate if segment has it set to True
                if seg.indeterminate is True:
                    current["indeterminate"] = True

        if current:
            blocks.append(current)
        return blocks

    def _can_merge(self, current: dict, segment: TranscriptSegment, text: str) -> bool:
        segment_id = (segment.speaker_id or "UNKNOWN").strip() or "UNKNOWN"
        if not current["speaker_id"] or current["speaker_id"] != segment_id:
            return False

        # Do not merge if indeterminate status differs
        current_indeterminate = current.get("indeterminate") is True
        segment_indeterminate = segment.indeterminate is True
        if current_indeterminate != segment_indeterminate:
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

    def _split_block_by_sentences(self, block: dict, max_sentences: int = 10) -> List[dict]:
        """
        Split a block into multiple blocks if it exceeds max_sentences.
        
        Args:
            block: Block dictionary with 'texts', 'start', 'end', 'speaker_id', 'speaker_name'
            max_sentences: Maximum number of sentences per block (default: 10)
        
        Returns:
            List of block dictionaries (original block if no split needed)
        """
        # Join all text parts into a single string
        full_text = " ".join(part.strip() for part in block.get("texts", []) if str(part).strip())
        if not full_text:
            return [block]
        
        # Find sentence boundaries using regex pattern
        sentence_pattern = re.compile(r"[.!?]\s+")
        sentence_boundaries: List[int] = [0]  # Start position
        
        for match in sentence_pattern.finditer(full_text):
            # Position after punctuation and whitespace
            sentence_boundaries.append(match.end())
        
        # Ensure we have at least one boundary (the end)
        if sentence_boundaries[-1] != len(full_text):
            sentence_boundaries.append(len(full_text))
        
        # Count sentences (boundaries include start and end, so sentences = boundaries - 1)
        num_sentences = len(sentence_boundaries) - 1
        
        # If block has <= max_sentences, return as-is
        if num_sentences <= max_sentences:
            return [block]
        
        # Split the block into multiple blocks
        split_blocks: List[dict] = []
        block_start = float(block.get("start", 0.0))
        block_end = float(block.get("end", block_start))
        block_duration = block_end - block_start
        total_chars = len(full_text)
        
        # Calculate how many blocks we need
        num_blocks = (num_sentences + max_sentences - 1) // max_sentences  # Ceiling division
        
        for block_idx in range(num_blocks):
            # Calculate sentence range for this block
            start_sentence_idx = block_idx * max_sentences
            end_sentence_idx = min(start_sentence_idx + max_sentences, num_sentences)
            
            # Get character positions for this block
            char_start = sentence_boundaries[start_sentence_idx]
            char_end = sentence_boundaries[end_sentence_idx]
            
            # Extract text for this block
            block_text = full_text[char_start:char_end].strip()
            if not block_text:
                continue
            
            # Calculate proportional time range
            if total_chars > 0:
                start_ratio = char_start / total_chars
                end_ratio = char_end / total_chars
            else:
                start_ratio = block_idx / num_blocks
                end_ratio = (block_idx + 1) / num_blocks
            
            split_start = block_start + start_ratio * block_duration
            split_end = block_start + end_ratio * block_duration
            
            # Create new block
            split_block = {
                "speaker_id": block.get("speaker_id") or "UNKNOWN",
                "speaker_name": block.get("speaker_name") or block.get("speaker_id") or "UNKNOWN",
                "texts": [block_text],
                "start": split_start,
                "end": split_end,
                "char_count": len(block_text),
            }
            split_blocks.append(split_block)
        
        return split_blocks if split_blocks else [block]

    def _split_blocks_by_sentences(self, blocks: List[dict], max_sentences: int = 10) -> List[dict]:
        """
        Split blocks that exceed max_sentences into multiple blocks.
        
        Args:
            blocks: List of block dictionaries
            max_sentences: Maximum number of sentences per block (default: 10)
        
        Returns:
            List of block dictionaries with splits applied
        """
        result: List[dict] = []
        for block in blocks:
            split_blocks = self._split_block_by_sentences(block, max_sentences)
            result.extend(split_blocks)
        return result

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


# ------------------------------------------------------------------ #
# Diarization utility functions (used by PrettifyStep)
# ------------------------------------------------------------------ #


def read_rttm_turns(path: Path) -> List[DiarizedTurn]:
    """Read diarization turns from an RTTM file."""
    if not path.exists():
        LOGGER.warning("RTTM file %s not found; cannot load diarization turns.", path)
        return []

    turns: List[DiarizedTurn] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8 or parts[0].upper() != "SPEAKER":
                continue
            try:
                start = float(parts[3])
                duration = float(parts[4])
            except ValueError:
                continue
            speaker = parts[7] if len(parts) > 7 else ""
            turns.append(
                DiarizedTurn(
                    start=start,
                    end=start + duration,
                    speaker_id=str(speaker or f"speaker_{len(turns)}"),
                )
            )
    turns.sort(key=lambda turn: (turn.start, turn.end))
    return turns


def ensure_diarized_turns(diarization_results: Any) -> List[DiarizedTurn]:
    """Convert diarization results to a list of DiarizedTurn objects."""
    if diarization_results is None:
        return []

    if isinstance(diarization_results, list):
        turns: List[DiarizedTurn] = []
        for item in diarization_results:
            if isinstance(item, DiarizedTurn):
                turns.append(item)
            elif isinstance(item, dict):
                try:
                    start = float(item["start"])
                    end = float(item["end"])
                    speaker = str(item.get("speaker") or item.get("speaker_id") or "")
                except (KeyError, TypeError, ValueError):
                    continue
                turns.append(DiarizedTurn(start=start, end=end, speaker_id=speaker or "UNKNOWN"))
        turns.sort(key=lambda turn: (turn.start, turn.end))
        return turns

    # Handle Path objects and strings
    if isinstance(diarization_results, Path):
        return read_rttm_turns(diarization_results)
    
    if isinstance(diarization_results, str):
        return read_rttm_turns(Path(diarization_results))

    return []


def _collect_overlap_parts(
    segment: TranscriptSegment,
    turns: Sequence[DiarizedTurn],
    start_idx: int,
) -> tuple[list[dict[str, float | str]], int]:
    overlaps: list[dict[str, float | str]] = []
    total_turns = len(turns)
    idx = start_idx
    start = segment.start
    end = segment.end

    while idx < total_turns and turns[idx].end <= start:
        idx += 1

    scan = idx
    while scan < total_turns and turns[scan].start < end:
        turn = turns[scan]
        overlap_start = max(start, turn.start)
        overlap_end = min(end, turn.end)
        if overlap_end > overlap_start:
            turn_duration = turn.end - turn.start
            overlaps.append(
                {
                    "start": overlap_start,
                    "end": overlap_end,
                    "speaker_id": turn.speaker_id,
                    "turn_start": turn.start,
                    "turn_end": turn.end,
                    "turn_duration": turn_duration,
                }
            )
        scan += 1

    return overlaps, idx


def _split_text_at_boundaries(text: str, durations: Sequence[float]) -> List[str]:
    """
    Split text at boundaries (sentence or word) based on proportional durations.
    
    First attempts to split at sentence boundaries (. ! ?), then falls back to word boundaries.
    Never splits within words.
    """
    if not durations:
        return []
    
    total_chars = len(text)
    if total_chars == 0:
        return ["" for _ in durations]
    
    total_duration = sum(durations)
    if total_duration <= 0:
        return ["" for _ in durations]
    
    # Calculate target character counts for each duration
    counts: List[int] = []
    for duration in durations:
        if duration <= 0:
            counts.append(0)
            continue
        ratio = duration / total_duration
        count = int(round(ratio * total_chars))
        counts.append(count)
    
    # Ensure at least one character per positive duration
    for idx, duration in enumerate(durations):
        if duration > 0 and counts[idx] == 0:
            counts[idx] = 1
    
    # Balance counts to sum to total_chars
    diff = total_chars - sum(counts)
    positive_indices = [idx for idx, duration in enumerate(durations) if duration > 0]
    if not positive_indices:
        positive_indices = list(range(len(counts)))
    
    idx = 0
    while diff != 0 and positive_indices:
        target = positive_indices[idx % len(positive_indices)]
        if diff > 0:
            counts[target] += 1
            diff -= 1
        elif counts[target] > 0:
            counts[target] -= 1
            diff += 1
        idx += 1
    
    # Try sentence-boundary splitting first
    sentence_pattern = re.compile(r"[.!?]\s+")
    sentence_boundaries: List[int] = [0]  # Start positions including 0
    
    for match in sentence_pattern.finditer(text):
        # Position after punctuation and whitespace
        sentence_boundaries.append(match.end())
    
    # Ensure we have at least one boundary (the end)
    if sentence_boundaries[-1] != len(text):
        sentence_boundaries.append(len(text))
    
    # If we have enough sentence boundaries to split, try sentence-based allocation
    num_parts_needed = len([c for c in counts if c > 0])
    if len(sentence_boundaries) > num_parts_needed:
        parts: List[str] = []
        cumulative_target = 0
        boundary_idx = 0
        
        for part_idx, count in enumerate(counts):
            if count <= 0:
                parts.append("")
                continue
            
            cumulative_target += count
            part_start = sentence_boundaries[boundary_idx]
            part_end = part_start
            
            # Find the best sentence boundary to end this part
            if part_idx == len(counts) - 1:
                # Last part: take everything remaining
                part_end = len(text)
                boundary_idx = len(sentence_boundaries)
            else:
                # Find the sentence boundary closest to target position
                if boundary_idx + 1 >= len(sentence_boundaries):
                    # No more boundaries, take rest of text
                    part_end = len(text)
                    boundary_idx = len(sentence_boundaries)
                else:
                    best_boundary_idx = boundary_idx + 1
                    best_distance = abs(sentence_boundaries[best_boundary_idx] - cumulative_target)
                    
                    for candidate_idx in range(boundary_idx + 2, len(sentence_boundaries)):
                        distance = abs(sentence_boundaries[candidate_idx] - cumulative_target)
                        if distance < best_distance:
                            best_distance = distance
                            best_boundary_idx = candidate_idx
                        else:
                            # Getting further away, stop searching
                            break
                    
                    part_end = sentence_boundaries[best_boundary_idx]
                    boundary_idx = best_boundary_idx
            
            parts.append(text[part_start:part_end])
        
        # If we successfully split into the right number of parts, return them
        if len(parts) == len(counts) and all(parts):
            return [part.strip() for part in parts]
    
    # Fall back to word-boundary splitting
    word_pattern = re.compile(r"\S+\s*")
    words = word_pattern.findall(text)
    if not words:
        words = [text]
    
    word_lengths = [len(word) for word in words]
    parts: List[str] = []
    word_idx = 0
    
    for count in counts:
        if count <= 0 or word_idx >= len(words):
            parts.append("")
            continue
        
        chunk_words: List[str] = []
        chunk_len = 0
        
        while word_idx < len(words) and (chunk_len < count or not chunk_words):
            current = words[word_idx]
            chunk_words.append(current)
            chunk_len += word_lengths[word_idx]
            word_idx += 1
            if chunk_len >= count:
                break
        
        parts.append("".join(chunk_words))
    
    # Add any remaining words to the last part
    if word_idx < len(words):
        remainder = "".join(words[word_idx:])
        if parts:
            parts[-1] += remainder
        else:
            parts.append(remainder)
    
    if len(parts) < len(counts):
        parts.extend([""] * (len(counts) - len(parts)))
    
    return [part.strip() for part in parts]


def _split_text_by_durations(text: str, durations: Sequence[float]) -> List[str]:
    """
    Split text by durations using boundary-aware splitting.
    
    Delegates to _split_text_at_boundaries which attempts sentence-boundary
    splitting first, then falls back to word-boundary splitting.
    """
    return _split_text_at_boundaries(text, durations)


def _calculate_sentence_time_range(
    text: str, segment_start: float, segment_end: float
) -> List[tuple[str, float, float]]:
    """
    Map sentence character positions to time positions within segment.
    
    Args:
        text: The segment text
        segment_start: Start time of the segment
        segment_end: End time of the segment
        
    Returns:
        List of tuples: (sentence_text, sentence_start_time, sentence_end_time)
        Returns empty list if no sentences detected (or only one sentence)
    """
    sentence_pattern = re.compile(r"[.!?]\s+")
    sentence_boundaries: List[int] = [0]  # Start positions including 0
    
    for match in sentence_pattern.finditer(text):
        # Position after punctuation and whitespace
        sentence_boundaries.append(match.end())
    
    # Ensure we have at least one boundary (the end)
    if sentence_boundaries[-1] != len(text):
        sentence_boundaries.append(len(text))
    
    # Need at least 2 sentences to split (boundaries include start and end)
    if len(sentence_boundaries) <= 2:
        return []
    
    total_chars = len(text)
    segment_duration = segment_end - segment_start
    sentences: List[tuple[str, float, float]] = []
    
    for i in range(len(sentence_boundaries) - 1):
        char_start = sentence_boundaries[i]
        char_end = sentence_boundaries[i + 1]
        sentence_text = text[char_start:char_end].strip()
        
        if not sentence_text:
            continue
        
        # Proportional mapping
        start_ratio = char_start / total_chars if total_chars > 0 else 0.0
        end_ratio = char_end / total_chars if total_chars > 0 else 1.0
        
        sentence_start_time = segment_start + start_ratio * segment_duration
        sentence_end_time = segment_start + end_ratio * segment_duration
        
        sentences.append((sentence_text, sentence_start_time, sentence_end_time))
    
    return sentences


def _assign_sentences_to_speakers(
    segment: TranscriptSegment,
    filtered_overlaps: list[dict[str, float | str]],
) -> Optional[List[TranscriptSegment]]:
    """
    Assign entire sentences to speakers based on percentage overlap.
    
    Args:
        segment: The transcript segment to assign
        filtered_overlaps: List of overlap dictionaries with turn duration info
        
    Returns:
        List of TranscriptSegments with assigned speakers, or None if sentences
        can't be detected or assignment fails (triggers fallback)
    """
    sentences = _calculate_sentence_time_range(segment.text, segment.start, segment.end)
    
    # If no sentences detected (or only one), return None to trigger fallback
    if not sentences:
        return None
    
    result_segments: List[TranscriptSegment] = []
    
    for sentence_text, sentence_start, sentence_end in sentences:
        # Find all overlapping turns for this sentence
        sentence_overlaps: List[dict[str, float | str]] = []
        
        for overlap in filtered_overlaps:
            overlap_start = float(overlap["start"])
            overlap_end = float(overlap["end"])
            
            # Check if overlap intersects with sentence time range
            intersection_start = max(sentence_start, overlap_start)
            intersection_end = min(sentence_end, overlap_end)
            
            if intersection_end > intersection_start:
                # Calculate overlap duration with sentence
                sentence_overlap_duration = intersection_end - intersection_start
                turn_duration = float(overlap.get("turn_duration", 0.0))
                
                # Calculate percentage: overlap_duration / turn_duration
                percentage = (
                    sentence_overlap_duration / turn_duration
                    if turn_duration > 0
                    else 0.0
                )
                
                sentence_overlaps.append(
                    {
                        **overlap,
                        "sentence_overlap_duration": sentence_overlap_duration,
                        "percentage": percentage,
                    }
                )
        
        if not sentence_overlaps:
            # No overlaps for this sentence, assign as UNKNOWN
            result_segments.append(
                TranscriptSegment(
                    start=sentence_start,
                    end=sentence_end,
                    text=sentence_text,
                    speaker_id="UNKNOWN",
                    speaker_name=segment.speaker_name,
                    confidence=segment.confidence,
                    justification=segment.justification,
                    metadata=dict(segment.metadata),
                )
            )
            continue
        
        # Find speaker with highest percentage overlap
        best_overlap = max(
            sentence_overlaps,
            key=lambda item: float(item.get("percentage", 0.0)),
        )
        
        speaker_id = str(best_overlap.get("speaker_id") or "UNKNOWN")
        result_segments.append(
            TranscriptSegment(
                start=sentence_start,
                end=sentence_end,
                text=sentence_text,
                speaker_id=speaker_id,
                speaker_name=_get_speaker_name(speaker_id, segment.speaker_name),
                confidence=segment.confidence,
                justification=segment.justification,
                metadata=dict(segment.metadata),
            )
        )
    
    return result_segments if result_segments else None


def _split_segment_by_parts(
    segment: TranscriptSegment, parts: Sequence[dict[str, float | str]]
) -> List[TranscriptSegment]:
    filtered_parts = [part for part in parts if float(part["end"]) - float(part["start"]) > 0.0]
    if len(filtered_parts) <= 1:
        return []

    filtered_parts[0]["start"] = min(float(filtered_parts[0]["start"]), segment.start)
    filtered_parts[-1]["end"] = max(float(filtered_parts[-1]["end"]), segment.end)

    durations = [
        float(part["end"]) - float(part["start"]) for part in filtered_parts
    ]
    seg_duration = max(segment.end - segment.start, 0.0)
    if seg_duration > 0 and durations:
        duration_sum = sum(durations)
        diff = seg_duration - duration_sum
        if abs(diff) > 1e-9:
            durations[-1] = max(0.0, durations[-1] + diff)

    text_parts = _split_text_by_durations(segment.text, durations)
    results: List[TranscriptSegment] = []
    for idx, part in enumerate(filtered_parts):
        text_piece = text_parts[idx] if idx < len(text_parts) else ""
        speaker_id = str(part.get("speaker_id") or "UNKNOWN")
        results.append(
            TranscriptSegment(
                start=float(part["start"]),
                end=float(part["end"]),
                text=text_piece,
                speaker_id=speaker_id,
                speaker_name=_get_speaker_name(speaker_id, segment.speaker_name),
                confidence=segment.confidence,
                justification=segment.justification,
                metadata=dict(segment.metadata),
            )
        )

    return results


def _get_speaker_name(speaker_id: str, original_speaker_name: Optional[str]) -> str:
    """Get speaker_name from speaker_id, falling back to original if speaker_id is UNKNOWN."""
    if not speaker_id or speaker_id.strip().upper() == "UNKNOWN":
        return original_speaker_name or "UNKNOWN"
    return speaker_id


def apply_diarization_labels(
    segments: Sequence[TranscriptSegment],
    turns: Sequence[DiarizedTurn],
    min_overlap_threshold: float = 0.3,
) -> List[TranscriptSegment]:
    """Apply diarization speaker IDs to transcript segments based on time overlap.
    
    Args:
        segments: Transcript segments to assign speaker IDs to
        turns: Diarized turns with speaker IDs
        min_overlap_threshold: Minimum overlap duration in seconds to consider (default: 0.3)
    """
    if not segments:
        return []
    if not turns:
        return list(segments)

    sorted_turns = sorted(turns, key=lambda turn: (turn.start, turn.end))
    updated_segments: List[TranscriptSegment] = []
    leading_index = 0
    total_turns = len(sorted_turns)

    for seg in segments:
        # Only skip if speaker_id is already assigned and not UNKNOWN
        # Process segments with None, empty string, or "UNKNOWN" speaker_id
        if seg.speaker_id and seg.speaker_id.strip() and seg.speaker_id.strip().upper() != "UNKNOWN":
            updated_segments.append(seg)
            continue

        overlaps, next_index = _collect_overlap_parts(seg, sorted_turns, leading_index)
        leading_index = next_index

        if not overlaps:
            updated_segments.append(seg)
            continue

        # Filter overlaps below threshold
        filtered_overlaps = [
            overlap
            for overlap in overlaps
            if float(overlap["end"]) - float(overlap["start"]) >= min_overlap_threshold
        ]

        if not filtered_overlaps:
            # All overlaps are below threshold - assign largest overlap with indeterminate=True
            if overlaps:
                best_overlap = max(
                    overlaps,
                    key=lambda item: float(item["end"]) - float(item["start"]),
                )
                speaker_id = str(best_overlap.get("speaker_id") or "UNKNOWN")
                updated_segments.append(
                    TranscriptSegment(
                        start=seg.start,
                        end=seg.end,
                        text=seg.text,
                        speaker_id=speaker_id,
                        speaker_name=_get_speaker_name(speaker_id, seg.speaker_name),
                        confidence=seg.confidence,
                        justification=seg.justification,
                        indeterminate=True,
                        metadata=dict(seg.metadata),
                    )
                )
            else:
                updated_segments.append(seg)
            continue

        # Check if all filtered overlaps have the same speaker
        unique_speakers = {str(overlap.get("speaker_id") or "UNKNOWN") for overlap in filtered_overlaps}
        if len(unique_speakers) == 1:
            # All overlaps are from the same speaker - assign without splitting
            speaker_id = next(iter(unique_speakers))
            updated_segments.append(
                TranscriptSegment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    speaker_id=speaker_id,
                    speaker_name=_get_speaker_name(speaker_id, seg.speaker_name),
                    confidence=seg.confidence,
                    justification=seg.justification,
                    metadata=dict(seg.metadata),
                )
            )
            continue

        if len(filtered_overlaps) == 1:
            overlap = filtered_overlaps[0]
            speaker_id = str(overlap.get("speaker_id") or "UNKNOWN")
            updated_segments.append(
                TranscriptSegment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    speaker_id=speaker_id,
                    speaker_name=_get_speaker_name(speaker_id, seg.speaker_name),
                    confidence=seg.confidence,
                    justification=seg.justification,
                    metadata=dict(seg.metadata),
                )
            )
            continue

        # Try sentence-based assignment first when multiple speakers detected
        sentence_segments = _assign_sentences_to_speakers(seg, filtered_overlaps)
        if sentence_segments:
            updated_segments.extend(sentence_segments)
            continue

        # For single sentence with multiple speakers, use percentage-based selection
        # Check if there's at least one sentence boundary (even if only one sentence)
        # Pattern matches punctuation followed by whitespace OR punctuation at end of string
        sentence_pattern = re.compile(r"[.!?](\s+|$)")
        has_sentence_boundary = bool(sentence_pattern.search(seg.text))
        
        if has_sentence_boundary and len(filtered_overlaps) > 1:
            # Single sentence with multiple speakers - use percentage-based selection
            segment_duration = seg.end - seg.start
            best_overlap = None
            best_percentage = -1.0
            
            for overlap in filtered_overlaps:
                overlap_duration = float(overlap["end"]) - float(overlap["start"])
                turn_duration = float(overlap.get("turn_duration", 0.0))
                
                if turn_duration > 0:
                    percentage = overlap_duration / turn_duration
                    if percentage > best_percentage:
                        best_percentage = percentage
                        best_overlap = overlap
            
            if best_overlap:
                speaker_id = str(best_overlap.get("speaker_id") or "UNKNOWN")
                updated_segments.append(
                    TranscriptSegment(
                        start=seg.start,
                        end=seg.end,
                        text=seg.text,
                        speaker_id=speaker_id,
                        speaker_name=_get_speaker_name(speaker_id, seg.speaker_name),
                        confidence=seg.confidence,
                        justification=seg.justification,
                        metadata=dict(seg.metadata),
                    )
                )
                continue

        # Fall back to current proportional splitting when sentences can't be detected
        split_segments = _split_segment_by_parts(seg, filtered_overlaps)
        if split_segments:
            updated_segments.extend(split_segments)
            continue

        best_overlap = max(
            filtered_overlaps,
            key=lambda item: float(item["end"]) - float(item["start"]),
        )
        speaker_id = str(best_overlap.get("speaker_id") or "UNKNOWN")
        updated_segments.append(
            TranscriptSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text,
                speaker_id=speaker_id,
                speaker_name=_get_speaker_name(speaker_id, seg.speaker_name),
                confidence=seg.confidence,
                justification=seg.justification,
                metadata=dict(seg.metadata),
            )
        )

    return updated_segments


