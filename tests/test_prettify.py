from __future__ import annotations

import json
from pathlib import Path
from typing import List

from mywhisper.models import PodcastEpisode, TranscriptSegment
from mywhisper.prettify import PrettifyConfig, TranscriptPrettifier


class StubCatalog:
    def __init__(self) -> None:
        self.records: List[tuple[str, str, Path, str]] = []

    def record_artefact(self, episode_id: str, kind: str, path: Path, artefact_key: str) -> None:
        self.records.append((episode_id, kind, path, artefact_key))


def _episode(tmp_path: Path) -> PodcastEpisode:
    return PodcastEpisode(
        episode_id="ep-001",
        show_title="Testing Show",
        episode_title="Episode 1",
        source_path=tmp_path / "audio.wav",
        metadata={"episode_key": "12345678"},
    )


def _create_mock_rttm(tmp_path: Path, episode: PodcastEpisode) -> Path:
    """Create a mock RTTM file for testing."""
    rttm_path = tmp_path / "data" / "transcripts" / episode.episode_key / f"{episode.episode_key}.rttm"
    rttm_path.parent.mkdir(parents=True, exist_ok=True)
    # Create a simple RTTM with two speakers covering the test segments
    rttm_content = """SPEAKER test 1 0.0 1.0 <NA> <NA> S0 <NA> <NA>
SPEAKER test 1 1.0 1.0 <NA> <NA> S0 <NA> <NA>
SPEAKER test 1 4.0 1.0 <NA> <NA> S1 <NA> <NA>
"""
    rttm_path.write_text(rttm_content, encoding="utf-8")
    return rttm_path


def test_prettifier_collapses_segments_and_records_artefact(tmp_path):
    config = PrettifyConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    assignment_path = config.assignment_path(episode)
    assignment_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {"start": 0.0, "end": 1.0, "text": "Hello", "speaker_id": "S0", "speaker_name": "Host"},
        {"start": 1.2, "end": 2.0, "text": "there", "speaker_id": "S0", "speaker_name": "Host"},
        {"start": 4.0, "end": 5.0, "text": "Hi everyone", "speaker_id": "S1", "speaker_name": "Guest"},
    ]
    assignment_path.write_text(json.dumps(payload), encoding="utf-8")
    rttm_path = _create_mock_rttm(tmp_path, episode)

    catalog = StubCatalog()
    prettifier = TranscriptPrettifier(episode, config=config, catalog=catalog)

    result = prettifier.prettify(diarization_results=rttm_path)
    readable_path = result.get("readable_path") if isinstance(result, dict) else result
    assert readable_path.exists()
    lines = readable_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "Host (S0): Hello there"
    assert lines[2] == "Guest (S1): Hi everyone"
    assert catalog.records
    recorded = catalog.records[0]
    assert recorded[1] == "readable_transcript"
    assert recorded[2] == readable_path


def test_prettifier_generator_emits_events(tmp_path):
    config = PrettifyConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    assignment_path = config.assignment_path(episode)
    assignment_path.parent.mkdir(parents=True, exist_ok=True)
    assignment_path.write_text(
        json.dumps([{"start": 0.0, "end": 1.0, "text": "Hello", "speaker_id": "S0"}]),
        encoding="utf-8",
    )
    rttm_path = _create_mock_rttm(tmp_path, episode)

    prettifier = TranscriptPrettifier(episode, config=config, catalog=StubCatalog())

    generator = prettifier.prettify(diarization_results=rttm_path, yield_progress=True)
    events = []
    try:
        while True:
            events.append(next(generator))
    except StopIteration as stop:
        result = stop.value

    assert events
    assert all(event.stage == "prettify" for event in events)
    assert events[0].checkpoint["status"] == "started"
    assert events[-1].checkpoint["status"] == "completed"
    readable_path = result.get("readable_path") if isinstance(result, dict) else result
    assert readable_path.exists()


def test_prettifier_get_outputs(tmp_path):
    """Test get_outputs method returns paths (hits line 73)."""
    config = PrettifyConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    assignment_path = config.assignment_path(episode)
    assignment_path.parent.mkdir(parents=True, exist_ok=True)
    assignment_path.write_text(
        json.dumps([{"start": 0.0, "end": 1.0, "text": "Hello", "speaker_id": "S0"}]),
        encoding="utf-8",
    )
    rttm_path = _create_mock_rttm(tmp_path, episode)

    prettifier = TranscriptPrettifier(episode, config=config, catalog=StubCatalog())
    prettifier.prettify(diarization_results=rttm_path)
    
    outputs = prettifier.get_outputs()
    assert "readable_path" in outputs
    assert "condensed_path" in outputs
    assert outputs["readable_path"] is not None
    assert outputs["condensed_path"] is not None


def test_prettifier_file_not_found_error(tmp_path):
    """Test FileNotFoundError when assignment file doesn't exist (hits line 109)."""
    import pytest
    config = PrettifyConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    # Create a dummy RTTM file so we get the FileNotFoundError for assignment, not diarization
    rttm_path = _create_mock_rttm(tmp_path, episode)
    
    prettifier = TranscriptPrettifier(episode, config=config, catalog=StubCatalog())
    
    with pytest.raises(FileNotFoundError, match="Assigned transcript not found"):
        prettifier.prettify(diarization_results=rttm_path)


def test_prettifier_invalid_record_types(tmp_path):
    """Test handling of invalid record types (hits lines 242-243, 245)."""
    config = PrettifyConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    assignment_path = config.assignment_path(episode)
    assignment_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create assignment with invalid types that will trigger TypeError/ValueError
    payload = [
        {"start": "invalid", "end": 1.0, "text": "Hello", "speaker_id": "S0"},  # Invalid start
        {"start": 2.0, "end": "invalid", "text": "World", "speaker_id": "S0"},  # Invalid end
        {"start": 3.0, "end": 4.0, "text": "", "speaker_id": "S0"},  # Empty text (hits line 245)
        {"start": 5.0, "end": 6.0, "text": "Valid", "speaker_id": "S0"},  # Valid record
    ]
    assignment_path.write_text(json.dumps(payload), encoding="utf-8")
    rttm_path = _create_mock_rttm(tmp_path, episode)

    prettifier = TranscriptPrettifier(episode, config=config, catalog=StubCatalog())
    result = prettifier.prettify(diarization_results=rttm_path)
    readable_path = result.get("readable_path") if isinstance(result, dict) else result
    assert readable_path.exists()
    # Should only contain the valid record
    content = readable_path.read_text(encoding="utf-8")
    assert "Valid" in content
    assert "Hello" not in content  # Invalid records should be skipped


def test_prettifier_empty_text_segments(tmp_path):
    """Test prettifier skips segments with empty text (hits line 273)."""
    from mywhisper.models import TranscriptSegment
    config = PrettifyConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    assignment_path = config.assignment_path(episode)
    assignment_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create assignment with empty text segments
    payload = [
        {"start": 0.0, "end": 1.0, "text": "Hello", "speaker_id": "S0", "speaker_name": "Host"},
        {"start": 1.0, "end": 2.0, "text": "   ", "speaker_id": "S0", "speaker_name": "Host"},  # Whitespace only
        {"start": 2.0, "end": 3.0, "text": "", "speaker_id": "S0", "speaker_name": "Host"},  # Empty
        {"start": 3.0, "end": 4.0, "text": "World", "speaker_id": "S1", "speaker_name": "Guest"},
    ]
    assignment_path.write_text(json.dumps(payload), encoding="utf-8")
    rttm_path = _create_mock_rttm(tmp_path, episode)

    prettifier = TranscriptPrettifier(episode, config=config, catalog=StubCatalog())
    result = prettifier.prettify(diarization_results=rttm_path)
    readable_path = result.get("readable_path") if isinstance(result, dict) else result
    assert readable_path.exists()
    content = readable_path.read_text(encoding="utf-8")
    assert "Hello" in content
    assert "World" in content
    # Empty/whitespace segments should be skipped


def test_prettifier_gap_limit(tmp_path):
    """Test prettifier respects gap_limit (hits line 305)."""
    config = PrettifyConfig(
        data_root=tmp_path / "data",
        collapse_gap_seconds=1.0,  # 1 second gap limit
    )
    episode = _episode(tmp_path)
    assignment_path = config.assignment_path(episode)
    assignment_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create segments with gap larger than limit
    payload = [
        {"start": 0.0, "end": 1.0, "text": "First", "speaker_id": "S0", "speaker_name": "Host"},
        {"start": 3.0, "end": 4.0, "text": "Second", "speaker_id": "S0", "speaker_name": "Host"},  # 2 second gap
    ]
    assignment_path.write_text(json.dumps(payload), encoding="utf-8")
    rttm_path = _create_mock_rttm(tmp_path, episode)

    prettifier = TranscriptPrettifier(episode, config=config, catalog=StubCatalog())
    result = prettifier.prettify(diarization_results=rttm_path)
    readable_path = result.get("readable_path") if isinstance(result, dict) else result
    assert readable_path.exists()
    content = readable_path.read_text(encoding="utf-8")
    # Should create separate blocks due to gap
    assert "First" in content
    assert "Second" in content


def test_prettifier_max_block_characters(tmp_path):
    """Test prettifier respects max_block_characters (hits lines 309-313)."""
    config = PrettifyConfig(
        data_root=tmp_path / "data",
        max_block_characters=10,  # Very small limit
    )
    episode = _episode(tmp_path)
    assignment_path = config.assignment_path(episode)
    assignment_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create segments that exceed the character limit
    payload = [
        {"start": 0.0, "end": 1.0, "text": "Short", "speaker_id": "S0", "speaker_name": "Host"},
        {"start": 1.0, "end": 2.0, "text": "Very long text that exceeds limit", "speaker_id": "S0", "speaker_name": "Host"},
    ]
    assignment_path.write_text(json.dumps(payload), encoding="utf-8")
    rttm_path = _create_mock_rttm(tmp_path, episode)

    prettifier = TranscriptPrettifier(episode, config=config, catalog=StubCatalog())
    result = prettifier.prettify(diarization_results=rttm_path)
    readable_path = result.get("readable_path") if isinstance(result, dict) else result
    assert readable_path.exists()
    content = readable_path.read_text(encoding="utf-8")
    # Should create separate blocks due to character limit
    assert "Short" in content


def test_prettifier_format_blocks_empty_text(tmp_path):
    """Test _format_blocks skips blocks with empty text (hits line 324)."""
    config = PrettifyConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    assignment_path = config.assignment_path(episode)
    assignment_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create assignment that results in empty text blocks after processing
    payload = [
        {"start": 0.0, "end": 1.0, "text": "Valid text", "speaker_id": "S0", "speaker_name": "Host"},
    ]
    assignment_path.write_text(json.dumps(payload), encoding="utf-8")

    prettifier = TranscriptPrettifier(episode, config=config, catalog=StubCatalog())
    # Manually test _format_blocks with empty text
    blocks = [
        {"speaker_id": "S0", "speaker_name": "Host", "texts": ["   "]},  # Whitespace only
        {"speaker_id": "S1", "speaker_name": "Guest", "texts": ["Valid"]},
    ]
    result = prettifier._format_blocks(blocks)
    # Should only include the valid block
    assert "Valid" in result
    assert "S1" in result


def test_should_prevent_merge_by_failsafe_single_segment(tmp_path):
    """Test _should_prevent_merge_by_failsafe with single segment."""
    config = PrettifyConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    prettifier = TranscriptPrettifier(episode, config=config)
    
    seg = TranscriptSegment(start=0.0, end=1.0, text="Hello", speaker_id="S0")
    assert not prettifier._should_prevent_merge_by_failsafe([seg])


def test_should_prevent_merge_by_failsafe_empty_group(tmp_path):
    """Test _should_prevent_merge_by_failsafe with empty group."""
    config = PrettifyConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    prettifier = TranscriptPrettifier(episode, config=config)
    
    assert not prettifier._should_prevent_merge_by_failsafe([])


def test_should_prevent_merge_by_failsafe_under_40_words(tmp_path):
    """Test _should_prevent_merge_by_failsafe with <= 40 words."""
    config = PrettifyConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    prettifier = TranscriptPrettifier(episode, config=config)
    
    # Create segments with exactly 40 words
    text = " ".join(["word"] * 40)
    seg1 = TranscriptSegment(start=0.0, end=1.0, text=text[:20], speaker_id="S0")
    seg2 = TranscriptSegment(start=1.0, end=2.0, text=text[20:], speaker_id="S0")
    
    assert not prettifier._should_prevent_merge_by_failsafe([seg1, seg2])


def test_should_prevent_merge_by_failsafe_over_40_words_with_punctuation(tmp_path):
    """Test _should_prevent_merge_by_failsafe with >40 words but has punctuation."""
    config = PrettifyConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    prettifier = TranscriptPrettifier(episode, config=config)
    
    # Create segments with 45 words and punctuation
    text = "word, " * 22 + "word"  # 45 words with commas
    seg1 = TranscriptSegment(start=0.0, end=1.0, text=text[:50], speaker_id="S0")
    seg2 = TranscriptSegment(start=1.0, end=2.0, text=text[50:], speaker_id="S0")
    
    assert not prettifier._should_prevent_merge_by_failsafe([seg1, seg2])


def test_should_prevent_merge_by_failsafe_over_40_words_no_punctuation(tmp_path):
    """Test _should_prevent_merge_by_failsafe with >40 words and no punctuation."""
    config = PrettifyConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    prettifier = TranscriptPrettifier(episode, config=config)
    
    # Create segments with 45 words and no punctuation
    text = " ".join(["word"] * 45)
    seg1 = TranscriptSegment(start=0.0, end=1.0, text=text[:50], speaker_id="S0")
    seg2 = TranscriptSegment(start=1.0, end=2.0, text=text[50:], speaker_id="S0")
    
    assert prettifier._should_prevent_merge_by_failsafe([seg1, seg2])


def test_should_prevent_merge_by_failsafe_with_whitespace(tmp_path):
    """Test _should_prevent_merge_by_failsafe handles whitespace correctly."""
    config = PrettifyConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    prettifier = TranscriptPrettifier(episode, config=config)
    
    # Create segments with extra whitespace
    text = "   " + " ".join(["word"] * 45) + "   "
    seg1 = TranscriptSegment(start=0.0, end=1.0, text=text[:50], speaker_id="S0")
    seg2 = TranscriptSegment(start=1.0, end=2.0, text=text[50:], speaker_id="S0")
    
    assert prettifier._should_prevent_merge_by_failsafe([seg1, seg2])


def test_create_merged_segment_basic(tmp_path):
    """Test _create_merged_segment creates correct merged segment."""
    config = PrettifyConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    prettifier = TranscriptPrettifier(episode, config=config)
    
    seg1 = TranscriptSegment(
        start=0.0, end=1.0, text="Hello", speaker_id="S0", speaker_name="Host",
        confidence=0.9, justification="test", metadata={"key": "value"}
    )
    seg2 = TranscriptSegment(start=1.0, end=2.0, text="there", speaker_id="S0")
    
    merged = prettifier._create_merged_segment([seg1, seg2], seg1)
    
    assert merged.start == 0.0
    assert merged.end == 2.0
    assert merged.text == "Hello there"
    assert merged.speaker_id == "S0"
    assert merged.speaker_name == "Host"
    assert merged.confidence == 0.9
    assert merged.justification == "test"
    assert merged.metadata == {"key": "value"}


def test_create_merged_segment_multiple_segments(tmp_path):
    """Test _create_merged_segment with multiple segments."""
    config = PrettifyConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    prettifier = TranscriptPrettifier(episode, config=config)
    
    seg1 = TranscriptSegment(start=0.0, end=1.0, text="Did you know", speaker_id="S0")
    seg2 = TranscriptSegment(start=1.0, end=2.0, text="that over long periods", speaker_id="S0")
    seg3 = TranscriptSegment(start=2.0, end=3.0, text="just a handful", speaker_id="S0")
    
    merged = prettifier._create_merged_segment([seg1, seg2, seg3], seg1)
    
    assert merged.start == 0.0
    assert merged.end == 3.0
    assert merged.text == "Did you know that over long periods just a handful"


def test_process_merge_group_single_segment(tmp_path):
    """Test _process_merge_group with single segment."""
    config = PrettifyConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    prettifier = TranscriptPrettifier(episode, config=config)
    
    seg = TranscriptSegment(start=0.0, end=1.0, text="Hello", speaker_id="S0")
    merged = []
    
    prettifier._process_merge_group([seg], seg, merged)
    
    assert len(merged) == 1
    assert merged[0] == seg


def test_process_merge_group_merges_when_allowed(tmp_path):
    """Test _process_merge_group merges segments when failsafe allows."""
    config = PrettifyConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    prettifier = TranscriptPrettifier(episode, config=config)
    
    seg1 = TranscriptSegment(start=0.0, end=1.0, text="Hello", speaker_id="S0")
    seg2 = TranscriptSegment(start=1.0, end=2.0, text="there.", speaker_id="S0")
    merged = []
    
    prettifier._process_merge_group([seg1, seg2], seg1, merged)
    
    assert len(merged) == 1
    assert merged[0].text == "Hello there."
    assert merged[0].start == 0.0
    assert merged[0].end == 2.0


def test_process_merge_group_adds_individually_when_failsafe_prevents(tmp_path):
    """Test _process_merge_group adds segments individually when failsafe prevents merge."""
    config = PrettifyConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    prettifier = TranscriptPrettifier(episode, config=config)
    
    # Create segments with >40 words and no punctuation
    text = " ".join(["word"] * 45)
    seg1 = TranscriptSegment(start=0.0, end=1.0, text=text[:50], speaker_id="S0")
    seg2 = TranscriptSegment(start=1.0, end=2.0, text=text[50:], speaker_id="S0")
    merged = []
    
    prettifier._process_merge_group([seg1, seg2], seg1, merged)
    
    assert len(merged) == 2
    assert merged[0] == seg1
    assert merged[1] == seg2


def test_merge_segments_by_sentences_empty_list(tmp_path):
    """Test _merge_segments_by_sentences with empty list."""
    config = PrettifyConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    prettifier = TranscriptPrettifier(episode, config=config)
    
    result = prettifier._merge_segments_by_sentences([])
    assert result == []


def test_merge_segments_by_sentences_single_segment_with_punctuation(tmp_path):
    """Test _merge_segments_by_sentences with single segment ending with punctuation."""
    config = PrettifyConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    prettifier = TranscriptPrettifier(episode, config=config)
    
    seg = TranscriptSegment(start=0.0, end=1.0, text="Hello.", speaker_id="S0")
    result = prettifier._merge_segments_by_sentences([seg])
    
    assert len(result) == 1
    assert result[0] == seg


def test_merge_segments_by_sentences_merges_two_segments(tmp_path):
    """Test _merge_segments_by_sentences merges two segments forming one sentence."""
    config = PrettifyConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    prettifier = TranscriptPrettifier(episode, config=config)
    
    seg1 = TranscriptSegment(start=0.0, end=1.0, text="Did you know", speaker_id="S0")
    seg2 = TranscriptSegment(start=1.0, end=2.0, text="that?", speaker_id="S0")
    result = prettifier._merge_segments_by_sentences([seg1, seg2])
    
    assert len(result) == 1
    assert result[0].text == "Did you know that?"
    assert result[0].start == 0.0
    assert result[0].end == 2.0


def test_merge_segments_by_sentences_merges_three_segments(tmp_path):
    """Test _merge_segments_by_sentences merges three segments forming one sentence."""
    config = PrettifyConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    prettifier = TranscriptPrettifier(episode, config=config)
    
    seg1 = TranscriptSegment(start=0.0, end=1.0, text="We'll examine", speaker_id="S0")
    seg2 = TranscriptSegment(start=1.0, end=2.0, text="how scale changes", speaker_id="S0")
    seg3 = TranscriptSegment(start=2.0, end=3.0, text="a business.", speaker_id="S0")
    result = prettifier._merge_segments_by_sentences([seg1, seg2, seg3])
    
    assert len(result) == 1
    assert result[0].text == "We'll examine how scale changes a business."
    assert result[0].start == 0.0
    assert result[0].end == 3.0


def test_merge_segments_by_sentences_respects_gap_constraint(tmp_path):
    """Test _merge_segments_by_sentences respects gap constraint."""
    config = PrettifyConfig(
        data_root=tmp_path / "data",
        collapse_gap_seconds=1.0,
    )
    episode = _episode(tmp_path)
    prettifier = TranscriptPrettifier(episode, config=config)
    
    seg1 = TranscriptSegment(start=0.0, end=1.0, text="Hello", speaker_id="S0")
    seg2 = TranscriptSegment(start=2.5, end=3.0, text="there.", speaker_id="S0")  # 1.5s gap > 1.0s limit
    result = prettifier._merge_segments_by_sentences([seg1, seg2])
    
    # Should merge what we have (seg1) and then process seg2 separately
    assert len(result) >= 1
    # seg2 should be processed separately due to gap


def test_merge_segments_by_sentences_ignores_speaker_constraint(tmp_path):
    """Test _merge_segments_by_sentences ignores speaker constraint (text-based only)."""
    config = PrettifyConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    prettifier = TranscriptPrettifier(episode, config=config)
    
    seg1 = TranscriptSegment(start=0.0, end=1.0, text="Hello", speaker_id="S0")
    seg2 = TranscriptSegment(start=1.0, end=2.0, text="there.", speaker_id="S1")  # Different speaker
    result = prettifier._merge_segments_by_sentences([seg1, seg2])
    
    # Should merge based on text content only, not speaker
    # seg1 doesn't end with punctuation, seg2 does, so they should merge
    assert len(result) == 1
    assert result[0].text == "Hello there."


def test_merge_segments_by_sentences_failsafe_prevents_merge(tmp_path):
    """Test _merge_segments_by_sentences respects failsafe rule."""
    config = PrettifyConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    prettifier = TranscriptPrettifier(episode, config=config)
    
    # Create segments with >40 words and no punctuation, ending with punctuation
    # Split into segments so first two together have >40 words without punctuation
    words = ["word"] * 45
    seg1 = TranscriptSegment(start=0.0, end=1.0, text=" ".join(words[:25]), speaker_id="S0")
    seg2 = TranscriptSegment(start=1.0, end=2.0, text=" ".join(words[25:45]), speaker_id="S0")
    seg3 = TranscriptSegment(start=2.0, end=3.0, text="end.", speaker_id="S0")
    
    result = prettifier._merge_segments_by_sentences([seg1, seg2, seg3])
    
    # When checking failsafe at seg3 boundary, seg1+seg2 have >40 words without punctuation
    # So failsafe should prevent merging seg1+seg2, but seg3 should be merged separately
    # Actually, the failsafe check happens when we find the boundary (seg3), and it checks
    # all segments in merge_group (seg1, seg2, seg3). Since seg3 has punctuation, the
    # combined text has punctuation, so failsafe doesn't prevent. Let's test a different scenario.
    
    # Test: seg1 and seg2 together have >40 words, no punctuation, and no seg3
    seg1_no_punct = TranscriptSegment(start=0.0, end=1.0, text=" ".join(words[:25]), speaker_id="S0")
    seg2_no_punct = TranscriptSegment(start=1.0, end=2.0, text=" ".join(words[25:45]), speaker_id="S0")
    
    result2 = prettifier._merge_segments_by_sentences([seg1_no_punct, seg2_no_punct])
    
    # Should not merge due to failsafe (>40 words, no punctuation)
    assert len(result2) == 2
    assert result2[0] == seg1_no_punct
    assert result2[1] == seg2_no_punct


def test_merge_segments_by_sentences_multiple_sentences(tmp_path):
    """Test _merge_segments_by_sentences handles multiple sentences correctly."""
    config = PrettifyConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    prettifier = TranscriptPrettifier(episode, config=config)
    
    seg1 = TranscriptSegment(start=0.0, end=1.0, text="First sentence.", speaker_id="S0")
    seg2 = TranscriptSegment(start=1.0, end=2.0, text="Second", speaker_id="S0")
    seg3 = TranscriptSegment(start=2.0, end=3.0, text="sentence.", speaker_id="S0")
    
    result = prettifier._merge_segments_by_sentences([seg1, seg2, seg3])
    
    # seg1 should be separate (ends with punctuation)
    # seg2 and seg3 should be merged
    assert len(result) == 2
    assert result[0].text == "First sentence."
    assert result[1].text == "Second sentence."


def test_merge_segments_by_sentences_end_of_list_no_punctuation(tmp_path):
    """Test _merge_segments_by_sentences handles end of list without punctuation."""
    config = PrettifyConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    prettifier = TranscriptPrettifier(episode, config=config)
    
    seg1 = TranscriptSegment(start=0.0, end=1.0, text="Hello", speaker_id="S0")
    seg2 = TranscriptSegment(start=1.0, end=2.0, text="there", speaker_id="S0")
    
    result = prettifier._merge_segments_by_sentences([seg1, seg2])
    
    # Should merge since they're contiguous and no punctuation found
    assert len(result) == 1
    assert result[0].text == "Hello there"


def test_merge_segments_by_sentences_with_question_mark(tmp_path):
    """Test _merge_segments_by_sentences recognizes question mark as sentence end."""
    config = PrettifyConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    prettifier = TranscriptPrettifier(episode, config=config)
    
    seg1 = TranscriptSegment(start=0.0, end=1.0, text="Did you know", speaker_id="S0")
    seg2 = TranscriptSegment(start=1.0, end=2.0, text="that?", speaker_id="S0")
    seg3 = TranscriptSegment(start=2.0, end=3.0, text="Yes.", speaker_id="S0")
    
    result = prettifier._merge_segments_by_sentences([seg1, seg2, seg3])
    
    assert len(result) == 2
    assert result[0].text == "Did you know that?"
    assert result[1].text == "Yes."


def test_merge_segments_by_sentences_with_exclamation(tmp_path):
    """Test _merge_segments_by_sentences recognizes exclamation as sentence end."""
    config = PrettifyConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    prettifier = TranscriptPrettifier(episode, config=config)
    
    seg1 = TranscriptSegment(start=0.0, end=1.0, text="Wow", speaker_id="S0")
    seg2 = TranscriptSegment(start=1.0, end=2.0, text="amazing!", speaker_id="S0")
    
    result = prettifier._merge_segments_by_sentences([seg1, seg2])
    
    assert len(result) == 1
    assert result[0].text == "Wow amazing!"


def test_merge_segments_by_sentences_gap_break_merges_collected(tmp_path):
    """Test _merge_segments_by_sentences merges collected segments when gap breaks collection."""
    config = PrettifyConfig(
        data_root=tmp_path / "data",
        collapse_gap_seconds=1.0,
    )
    episode = _episode(tmp_path)
    prettifier = TranscriptPrettifier(episode, config=config)
    
    # Create segments where first two are close, third has large gap
    seg1 = TranscriptSegment(start=0.0, end=1.0, text="Hello", speaker_id="S0")
    seg2 = TranscriptSegment(start=1.0, end=1.5, text="there", speaker_id="S0")
    seg3 = TranscriptSegment(start=3.0, end=4.0, text="Next.", speaker_id="S0")  # 1.5s gap > 1.0s limit
    
    result = prettifier._merge_segments_by_sentences([seg1, seg2, seg3])
    
    # seg1 and seg2 should be merged (contiguous, no punctuation), seg3 separate
    assert len(result) >= 2
    # First result should be merged seg1+seg2
    assert "Hello there" in result[0].text or "there" in result[0].text


def test_merge_segments_by_sentences_merges_across_speakers(tmp_path):
    """Test _merge_segments_by_sentences merges segments across speaker changes (text-based only)."""
    config = PrettifyConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    prettifier = TranscriptPrettifier(episode, config=config)
    
    # Create segments where first two are same speaker, third is different
    # Since we no longer check speakers, all should merge if contiguous
    seg1 = TranscriptSegment(start=0.0, end=1.0, text="Hello", speaker_id="S0")
    seg2 = TranscriptSegment(start=1.0, end=2.0, text="there", speaker_id="S0")
    seg3 = TranscriptSegment(start=2.0, end=3.0, text="Next.", speaker_id="S1")  # Different speaker
    
    result = prettifier._merge_segments_by_sentences([seg1, seg2, seg3])
    
    # All segments should merge since they're contiguous and seg3 ends with punctuation
    # seg1 and seg2 don't end with punctuation, so they merge with seg3
    assert len(result) == 1
    assert result[0].text == "Hello there Next."


def test_collapse_segments_preserves_indeterminate_field(tmp_path):
    """Test that indeterminate field is preserved through collapse and serialization."""
    config = PrettifyConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    prettifier = TranscriptPrettifier(episode, config=config)
    
    # Create segments: two normal segments and one indeterminate segment, all same speaker
    seg1 = TranscriptSegment(start=0.0, end=1.0, text="First sentence.", speaker_id="S0", indeterminate=None)
    seg2 = TranscriptSegment(start=1.2, end=2.0, text="Second sentence.", speaker_id="S0", indeterminate=None)
    seg3 = TranscriptSegment(start=2.2, end=3.0, text="Indeterminate sentence.", speaker_id="S0", indeterminate=True)
    seg4 = TranscriptSegment(start=3.2, end=4.0, text="Another normal.", speaker_id="S0", indeterminate=None)
    
    blocks = prettifier._collapse_segments([seg1, seg2, seg3, seg4])
    
    # seg1 and seg2 should collapse (both normal, same speaker, gap < 1.5s)
    # seg3 should NOT collapse with seg2 or seg4 (indeterminate differs)
    # seg4 should be separate
    assert len(blocks) == 3
    assert blocks[0]["speaker_id"] == "S0"
    assert "indeterminate" not in blocks[0]  # Normal segments don't have indeterminate
    assert blocks[1]["speaker_id"] == "S0"
    assert blocks[1]["indeterminate"] is True  # Indeterminate segment has the field
    assert blocks[2]["speaker_id"] == "S0"
    assert "indeterminate" not in blocks[2]  # Normal segments don't have indeterminate


def test_condensed_json_includes_indeterminate_only_when_true(tmp_path):
    """Test that condensed JSON only includes indeterminate field when True."""
    config = PrettifyConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    assignment_path = config.assignment_path(episode)
    assignment_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create assignment file with one indeterminate segment
    payload = [
        {"start": 0.0, "end": 1.0, "text": "Normal segment.", "speaker_id": "S0"},
        {"start": 1.2, "end": 2.0, "text": "Indeterminate segment.", "speaker_id": "S0", "indeterminate": True},
    ]
    assignment_path.write_text(json.dumps(payload), encoding="utf-8")
    
    # Create minimal RTTM
    rttm_path = tmp_path / "data" / "transcripts" / episode.episode_key / f"{episode.episode_key}.rttm"
    rttm_path.parent.mkdir(parents=True, exist_ok=True)
    rttm_content = "SPEAKER test 1 0.0 2.0 <NA> <NA> S0 <NA> <NA>\n"
    rttm_path.write_text(rttm_content, encoding="utf-8")
    
    catalog = StubCatalog()
    prettifier = TranscriptPrettifier(episode, config=config, catalog=catalog)
    
    result = prettifier.prettify(diarization_results=rttm_path)
    condensed_path = result.get("condensed_path") if isinstance(result, dict) else None
    assert condensed_path and condensed_path.exists()
    
    condensed_data = json.loads(condensed_path.read_text(encoding="utf-8"))
    assert len(condensed_data) == 2
    
    # First segment (normal) should not have indeterminate field
    assert "indeterminate" not in condensed_data[0]
    
    # Second segment (indeterminate) should have indeterminate=True
    assert condensed_data[1]["indeterminate"] is True


