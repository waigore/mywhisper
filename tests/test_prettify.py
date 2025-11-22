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


