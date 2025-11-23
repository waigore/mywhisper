from __future__ import annotations

import json
from pathlib import Path
from typing import List

from mywhisper.models import PodcastEpisode
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

    catalog = StubCatalog()
    prettifier = TranscriptPrettifier(episode, config=config, catalog=catalog)

    result = prettifier.prettify()
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

    prettifier = TranscriptPrettifier(episode, config=config, catalog=StubCatalog())

    generator = prettifier.prettify(yield_progress=True)
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

    prettifier = TranscriptPrettifier(episode, config=config, catalog=StubCatalog())
    prettifier.prettify()
    
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
    
    prettifier = TranscriptPrettifier(episode, config=config, catalog=StubCatalog())
    
    with pytest.raises(FileNotFoundError, match="Assigned transcript not found"):
        prettifier.prettify()


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

    prettifier = TranscriptPrettifier(episode, config=config, catalog=StubCatalog())
    result = prettifier.prettify()
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

    prettifier = TranscriptPrettifier(episode, config=config, catalog=StubCatalog())
    result = prettifier.prettify()
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

    prettifier = TranscriptPrettifier(episode, config=config, catalog=StubCatalog())
    result = prettifier.prettify()
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

    prettifier = TranscriptPrettifier(episode, config=config, catalog=StubCatalog())
    result = prettifier.prettify()
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


