import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from config import (
    ASSIGN_OUTPUT_PATH,
    ASSIGN_RTTM_PATH,
    ASSIGN_TRANSCRIPT_PATH,
    ASSIGN_TRANSCRIPT_TIME_FACTOR,
    ASSIGN_UNKNOWN_LABEL,
)


@dataclass(frozen=True)
class DiarizationSegment:
    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str


def _load_transcript(path: Path, time_factor: float) -> List[TranscriptSegment]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    segments: List[TranscriptSegment] = []
    for entry in payload:
        start = float(entry["start"]) / time_factor
        end = float(entry["end"]) / time_factor
        text = entry["text"]
        segments.append(TranscriptSegment(start=start, end=end, text=text))
    return segments


def _parse_rttm_line(line: str) -> DiarizationSegment:
    """
    Parse a single RTTM line and return a diarization segment.

    Expected format:
        SPEAKER <uri> <chan> <start> <duration> <ortho> <stype> <name> <conf> <slat>
    """
    parts = line.strip().split()
    if len(parts) < 9 or parts[0].upper() != "SPEAKER":
        raise ValueError(f"Invalid RTTM line: {line.strip()}")

    start = float(parts[3])
    duration = float(parts[4])
    speaker = parts[7]
    end = start + duration
    return DiarizationSegment(start=start, end=end, speaker=speaker)


def _load_diarization(path: Path) -> List[DiarizationSegment]:
    segments: List[DiarizationSegment] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            segments.append(_parse_rttm_line(stripped))
    return segments


def _assign_speakers(
    transcript: Sequence[TranscriptSegment],
    diarization: Sequence[DiarizationSegment],
    unknown_label: str = "UNKNOWN",
) -> List[dict]:
    diarized: List[dict] = []
    for seg in transcript:
        overlaps = []
        for diar_seg in diarization:
            overlap_start = max(seg.start, diar_seg.start)
            overlap_end = min(seg.end, diar_seg.end)
            if overlap_end > overlap_start:
                overlaps.append((overlap_end - overlap_start, diar_seg.speaker))

        if overlaps:
            _, speaker = max(overlaps, key=lambda item: item[0])
        else:
            speaker = unknown_label

        diarized.append(
            {
                "start": seg.start,
                "end": seg.end,
                "speaker": speaker,
                "text": seg.text,
            }
        )
    return diarized


def _write_output(path: Path, segments: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(list(segments), f, indent=4)

def main() -> None:
    transcript_segments = _load_transcript(
        ASSIGN_TRANSCRIPT_PATH,
        ASSIGN_TRANSCRIPT_TIME_FACTOR,
    )
    diarization_segments = _load_diarization(ASSIGN_RTTM_PATH)

    if not diarization_segments:
        raise ValueError(f"No diarization segments found in {ASSIGN_RTTM_PATH}")

    diarized_transcript = _assign_speakers(
        transcript_segments,
        diarization_segments,
        unknown_label=ASSIGN_UNKNOWN_LABEL,
    )
    _write_output(ASSIGN_OUTPUT_PATH, diarized_transcript)


if __name__ == "__main__":
    main()

