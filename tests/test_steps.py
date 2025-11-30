from __future__ import annotations

from pathlib import Path
import json

import pytest

from mywhisper.checkpoints.models import PipelineCheckpoint
from mywhisper.models import DiarizedTurn, TranscriptSegment
from mywhisper.myw.config import MywConfig
from mywhisper.myw.services.steps import (
    STEP_ORDER,
    ensure_placeholder_assignment,
    load_artefact_path,
    read_transcript,
    validate_assignment_availability,
    validate_classified_availability,
    validate_condensed_availability,
    validate_diarization_availability,
    validate_themes_availability,
    validate_transcript_availability,
    validate_vocative_availability,
)
from mywhisper.prettify import (
    apply_diarization_labels,
    ensure_diarized_turns,
    read_rttm_turns,
)


class InMemoryCheckpointStore:
    def __init__(self, rows: dict[tuple[str, str], PipelineCheckpoint] | None = None) -> None:
        self._rows = rows or {}

    def get_step(self, episode_id: str, step: str):
        return self._rows.get((episode_id, step))


def test_load_artefact_path(tmp_path):
    """Test generic artefact path loader."""
    checkpoints = InMemoryCheckpointStore()
    episode_id = "ep1"
    step = "transcribe"
    
    # No checkpoint
    assert load_artefact_path(checkpoints, episode_id, step, ("transcript_path", "path")) is None
    
    # With checkpoint - path in details
    transcript_file = tmp_path / "transcript.json"
    transcript_file.write_text('[]')
    checkpoints._rows[(episode_id, step)] = PipelineCheckpoint(
        episode_id=episode_id,
        step=step,
        status="completed",
        stage="persisted",
        message="done",
        details={"transcript_path": str(transcript_file)},
    )
    result = load_artefact_path(checkpoints, episode_id, step, ("transcript_path", "path"))
    assert result == transcript_file
    
    # With checkpoint - path in payload (fallback)
    checkpoints._rows[(episode_id, step)] = PipelineCheckpoint(
        episode_id=episode_id,
        step=step,
        status="completed",
        stage="persisted",
        message="done",
        payload={"path": str(transcript_file)},
    )
    result = load_artefact_path(checkpoints, episode_id, step, ("transcript_path", "path"))
    assert result == transcript_file


def test_read_transcript(tmp_path):
    """Test reading transcript from JSON file."""
    transcript_path = tmp_path / "transcript.json"
    payload = [
        {
            "start": 0.0,
            "end": 1.0,
            "text": "Hello",
            "speaker_id": "S0",
            "speaker_name": "Host",
        }
    ]
    transcript_path.write_text(json.dumps(payload))
    
    segments = read_transcript(transcript_path)
    assert segments
    assert len(segments) == 1
    assert segments[0].text == "Hello"
    assert segments[0].speaker_id == "S0"
    
    # Missing file
    assert read_transcript(tmp_path / "missing.json") is None


def test_read_rttm_turns(tmp_path):
    """Test reading RTTM file."""
    rttm = tmp_path / "turns.rttm"
    # SPEAKER <uri> <chan> <start> <dur> <ortho> <stype> <name>
    rttm.write_text("SPEAKER test 1 0.00 0.80 <NA> <NA> SPK0\n")
    
    turns = read_rttm_turns(rttm)
    assert turns
    assert len(turns) == 1
    assert turns[0].speaker_id == "SPK0"
    assert turns[0].start == 0.0
    assert turns[0].end == 0.8
    
    # Missing file
    assert read_rttm_turns(tmp_path / "missing.rttm") == []


def test_ensure_diarized_turns():
    """Test converting diarization results to turns."""
    # From list of DiarizedTurn
    turns_list = [DiarizedTurn(start=0.0, end=1.0, speaker_id="S0")]
    result = ensure_diarized_turns(turns_list)
    assert result == turns_list
    
    # From list of dicts
    dict_list = [{"start": 0.0, "end": 1.0, "speaker": "S0"}]
    result = ensure_diarized_turns(dict_list)
    assert len(result) == 1
    assert result[0].speaker_id == "S0"
    
    # From Path (will call read_rttm_turns)
    # This is tested separately
    
    # None
    assert ensure_diarized_turns(None) == []


def test_apply_diarization_labels(tmp_path):
    """Test applying diarization labels to segments."""
    segments = [
        TranscriptSegment(start=0.0, end=1.0, text="intro"),
        TranscriptSegment(start=1.0, end=2.0, text="guest"),
    ]
    turns = [
        DiarizedTurn(start=0.0, end=1.2, speaker_id="SPK0"),
        DiarizedTurn(start=1.2, end=2.5, speaker_id="SPK1"),
    ]
    
    labelled = apply_diarization_labels(segments, turns)
    # With threshold filtering (0.3s default), the tiny 0.2s overlap on segment 2 is filtered
    # So segment 2 gets assigned to SPK1 without splitting
    assert len(labelled) == 2
    assert labelled[0].speaker_id == "SPK0"
    assert labelled[0].start == pytest.approx(0.0)
    assert labelled[0].end == pytest.approx(1.0)
    assert labelled[1].speaker_id == "SPK1"
    assert labelled[1].start == pytest.approx(1.0)
    assert labelled[1].end == pytest.approx(2.0)
    
    # With lower threshold, should split segment 2
    labelled_low_threshold = apply_diarization_labels(segments, turns, min_overlap_threshold=0.1)
    assert len(labelled_low_threshold) == 3
    assert labelled_low_threshold[0].speaker_id == "SPK0"
    assert labelled_low_threshold[-1].speaker_id == "SPK1"
    assert any(seg.speaker_id == "SPK0" and seg.start >= 1.0 for seg in labelled_low_threshold[1:])
    assert any(seg.speaker_id == "SPK1" and seg.start >= 1.2 for seg in labelled_low_threshold)
    
    # Empty segments
    assert apply_diarization_labels([], turns) == []
    
    # Empty turns
    assert apply_diarization_labels(segments, []) == list(segments)


def test_apply_diarization_labels_splits_overlapping_segment():
    """Segments spanning multiple turns should split by diarization."""
    segments = [
        TranscriptSegment(start=0.0, end=2.0, text="hello world", speaker_id=None),
    ]
    turns = [
        DiarizedTurn(start=0.0, end=1.0, speaker_id="SPK0"),
        DiarizedTurn(start=1.0, end=2.0, speaker_id="SPK1"),
    ]
    labelled = apply_diarization_labels(segments, turns)
    assert len(labelled) == 2
    assert labelled[0].speaker_id == "SPK0"
    assert labelled[1].speaker_id == "SPK1"
    assert labelled[0].start == pytest.approx(0.0)
    assert labelled[0].end == pytest.approx(1.0)
    assert labelled[1].start == pytest.approx(1.0)
    assert labelled[1].end == pytest.approx(2.0)
    assert labelled[0].text
    assert labelled[1].text


def test_apply_diarization_labels_same_speaker_no_split():
    """Segments with multiple overlaps from same speaker should not split."""
    segments = [
        TranscriptSegment(
            start=161.54, 
            end=168.32, 
            text="Larry McDonald, welcome to Hidden Forces.",
            speaker_id=None
        ),
    ]
    turns = [
        DiarizedTurn(start=98.868, end=161.694, speaker_id="SPEAKER_00"),
        DiarizedTurn(start=164.950, end=168.578, speaker_id="SPEAKER_00"),
    ]
    labelled = apply_diarization_labels(segments, turns)
    # Should not split since both overlaps are from same speaker
    assert len(labelled) == 1
    assert labelled[0].speaker_id == "SPEAKER_00"
    assert labelled[0].start == pytest.approx(161.54)
    assert labelled[0].end == pytest.approx(168.32)
    assert labelled[0].text == "Larry McDonald, welcome to Hidden Forces."


def test_apply_diarization_labels_filters_tiny_overlaps():
    """Tiny overlaps below threshold should be filtered out."""
    segments = [
        TranscriptSegment(
            start=168.32,
            end=172.9,
            text="It's great to be with you guys. I've heard a lot about the platform.",
            speaker_id=None
        ),
    ]
    turns = [
        DiarizedTurn(start=168.32, end=168.578, speaker_id="SPEAKER_00"),  # 0.258s overlap
        DiarizedTurn(start=169.203, end=173.489, speaker_id="SPEAKER_01"),  # ~3.7s overlap
    ]
    # With threshold of 0.5s, first overlap (0.258s) should be filtered
    labelled = apply_diarization_labels(segments, turns, min_overlap_threshold=0.5)
    # Should only have one overlap after filtering (SPEAKER_01)
    assert len(labelled) == 1
    assert labelled[0].speaker_id == "SPEAKER_01"
    
    # With lower threshold, should split
    labelled_low_threshold = apply_diarization_labels(segments, turns, min_overlap_threshold=0.1)
    assert len(labelled_low_threshold) >= 2


def test_apply_diarization_labels_boundary_aware_splitting():
    """Text should be split at sentence boundaries when possible."""
    segments = [
        TranscriptSegment(
            start=0.0,
            end=10.0,
            text="Hello world. This is a test. Another sentence here.",
            speaker_id=None
        ),
    ]
    turns = [
        DiarizedTurn(start=0.0, end=5.0, speaker_id="SPK0"),
        DiarizedTurn(start=5.0, end=10.0, speaker_id="SPK1"),
    ]
    labelled = apply_diarization_labels(segments, turns)
    # With sentence-based assignment, should split into 3 segments (one per sentence)
    assert len(labelled) == 3
    assert labelled[0].speaker_id == "SPK0"
    assert "Hello world" in labelled[0].text
    assert labelled[0].text.endswith(".")
    # Second sentence should be assigned based on overlap
    assert labelled[1].text.endswith(".")
    # Third sentence should be assigned to SPK1
    assert labelled[2].speaker_id == "SPK1"
    assert "Another sentence here" in labelled[2].text
    assert labelled[2].text.endswith(".")


def test_apply_diarization_labels_sentence_based_assignment():
    """Multiple sentences should be assigned to speakers without splitting sentences."""
    segments = [
        TranscriptSegment(
            start=0.0,
            end=10.0,
            text="First sentence here. Second sentence here. Third sentence here.",
            speaker_id=None
        ),
    ]
    turns = [
        DiarizedTurn(start=0.0, end=3.5, speaker_id="SPK0"),  # Overlaps with first sentence
        DiarizedTurn(start=3.5, end=6.5, speaker_id="SPK1"),  # Overlaps with second sentence
        DiarizedTurn(start=6.5, end=10.0, speaker_id="SPK2"),  # Overlaps with third sentence
    ]
    labelled = apply_diarization_labels(segments, turns)
    # Should have 3 segments, one per sentence
    assert len(labelled) == 3
    assert labelled[0].speaker_id == "SPK0"
    assert "First sentence here" in labelled[0].text
    assert labelled[1].speaker_id == "SPK1"
    assert "Second sentence here" in labelled[1].text
    assert labelled[2].speaker_id == "SPK2"
    assert "Third sentence here" in labelled[2].text
    # Verify sentences are complete (not split)
    assert labelled[0].text.endswith(".")
    assert labelled[1].text.endswith(".")
    assert labelled[2].text.endswith(".")


def test_apply_diarization_labels_percentage_based_selection():
    """When multiple speakers overlap same sentence, assign based on percentage overlap."""
    segments = [
        TranscriptSegment(
            start=0.0,
            end=5.0,
            text="This is a single sentence that overlaps with multiple speakers.",
            speaker_id=None
        ),
    ]
    # Both speakers overlap 1.5s with the sentence
    # SPK0 has a 10s turn, SPK1 has a 1.8s turn
    # SPK0 percentage: 1.5/10 = 0.15 (15%)
    # SPK1 percentage: 1.5/1.8 = 0.833 (83.3%)
    # Should assign to SPK1 (higher percentage)
    turns = [
        DiarizedTurn(start=0.0, end=10.0, speaker_id="SPK0"),  # 10s turn, overlaps 1.5s
        DiarizedTurn(start=1.0, end=2.8, speaker_id="SPK1"),  # 1.8s turn, overlaps 1.5s
    ]
    labelled = apply_diarization_labels(segments, turns)
    # Should assign entire sentence to SPK1 (higher percentage)
    assert len(labelled) == 1
    assert labelled[0].speaker_id == "SPK1"
    assert labelled[0].text == "This is a single sentence that overlaps with multiple speakers."


def test_apply_diarization_labels_sentences_never_split():
    """Sentences should never be split across multiple speakers."""
    segments = [
        TranscriptSegment(
            start=0.0,
            end=10.0,
            text="First sentence. Second sentence. Third sentence.",
            speaker_id=None
        ),
    ]
    # Overlapping turns that would cause proportional splitting
    turns = [
        DiarizedTurn(start=0.0, end=3.5, speaker_id="SPK0"),
        DiarizedTurn(start=2.0, end=6.5, speaker_id="SPK1"),
        DiarizedTurn(start=5.0, end=10.0, speaker_id="SPK2"),
    ]
    labelled = apply_diarization_labels(segments, turns)
    # Each sentence should be assigned to one speaker (not split)
    assert len(labelled) == 3
    # Verify each segment is a complete sentence
    for seg in labelled:
        assert seg.text.endswith(".")
        # Verify sentence is not split (should contain complete sentence text)
        assert "sentence" in seg.text.lower()


def test_ensure_placeholder_assignment(tmp_path):
    """Test creating placeholder assignment file."""
    config = MywConfig(
        data_dir=tmp_path,
        db_path=tmp_path / "db.sqlite",
        podcast_cache_path=tmp_path,
        podcast_db_path=tmp_path / "podcasts.db",
        log_level="INFO",
        whisper_model=str(tmp_path / "model.bin"),
        device=None,
        ollama_model="llama3",
        spacy_model="en_core_web_sm",
        hf_token=None,
    )
    
    from mywhisper.models import PodcastEpisode
    
    episode = PodcastEpisode(
        episode_id="ep1",
        show_title="Show",
        episode_title="Episode",
        source_path=tmp_path / "audio.wav",
        metadata={"episode_key": "12345678"},
    )
    
    segments = [
        TranscriptSegment(start=0.0, end=1.0, text="Hello", speaker_id="S0"),
    ]
    
    path = ensure_placeholder_assignment(config, episode, segments)
    assert path.exists()
    
    data = json.loads(path.read_text())
    assert len(data) == 1
    assert data[0]["speaker_id"] == "S0"
    assert data[0]["speaker_name"] == "S0"


def test_validation_functions(tmp_path):
    """Test validation functions."""
    # validate_transcript_availability
    with pytest.raises(RuntimeError):
        validate_transcript_availability(("diarize",), None)
    validate_transcript_availability(("transcribe", "diarize"), None)
    
    # validate_diarization_availability
    with pytest.raises(RuntimeError):
        validate_diarization_availability(("prettify",), {}, None)
    validate_diarization_availability(("prettify", "diarize"), {}, None)
    validate_diarization_availability(("assign",), {"diarize": "completed"}, None)
    
    # validate_assignment_availability
    with pytest.raises(RuntimeError):
        validate_assignment_availability(("assign",), None)
    # Create a real file for the test
    readable_file = tmp_path / "readable.txt"
    readable_file.write_text("test")
    validate_assignment_availability(("assign",), readable_file)
    
    # validate_condensed_availability
    with pytest.raises(RuntimeError):
        validate_condensed_availability(("thematize",), None)
    
    # validate_themes_availability
    with pytest.raises(RuntimeError):
        validate_themes_availability(("classify",), None)
    
    # validate_classified_availability (soft check)
    validate_classified_availability(("vocative",), {})
    validate_classified_availability(("vocative",), {"classify": "completed"})
    
    # validate_vocative_availability (no-op)
    validate_vocative_availability(("assign",), None)


def test_step_order():
    """Test that STEP_ORDER is defined correctly."""
    assert "transcribe" in STEP_ORDER
    assert "diarize" in STEP_ORDER
    assert "prettify" in STEP_ORDER
    assert "thematize" in STEP_ORDER
    assert "classify" in STEP_ORDER
    assert "vocative" in STEP_ORDER
    assert "assign" in STEP_ORDER
    assert len(STEP_ORDER) == 7

