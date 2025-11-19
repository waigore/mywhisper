from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from mywhisper.models import PipelineEvent, PodcastEpisode
from mywhisper.thematize import EpisodeThematizer, ThematizeConfig


class StubCatalog:
    def __init__(self) -> None:
        self.records: List[tuple[str, str, Path, str]] = []

    def record_artefact(self, episode_id: str, kind: str, path: Path, artefact_key: str) -> None:
        self.records.append((episode_id, kind, path, artefact_key))


def _episode(tmp_path: Path) -> PodcastEpisode:
    return PodcastEpisode(
        episode_id="ep-theme",
        show_title="Theme Show",
        episode_title="Theme Episode",
        source_path=tmp_path / "audio.wav",
        metadata={"episode_key": "34567890"},
    )


class DummyClient:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        if self.calls == 1:
            return '[{"theme": "Intro", "summary": "Opening discussion."}]'
        return '[{"theme": "Intro", "summary": "Additional intro."}, {"theme": "Tech", "summary": "Tech talk."}]'


class FailingClient:
    def generate(self, prompt: str) -> str:
        raise RuntimeError("LLM offline")


def test_thematizer_generates_per_segment_summaries(tmp_path, monkeypatch):
    config = ThematizeConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    condensed_path = config.condensed_path(episode)
    condensed_path.parent.mkdir(parents=True, exist_ok=True)
    condensed_path.write_text(
        json.dumps(
            [
                {"start": 0.0, "end": 1.0, "speaker_id": "S0", "speaker_name": "Host", "text": "Welcome to the show."},
                {"start": 1.0, "end": 2.0, "speaker_id": "S1", "speaker_name": "Guest", "text": "Tech is great."},
            ]
        ),
        encoding="utf-8",
    )

    catalog = StubCatalog()
    thematizer = EpisodeThematizer(
        podcast=episode,
        config=config,
        catalog=catalog,
        client=DummyClient(),  # type: ignore[arg-type]
    )

    themes_path = thematizer.thematize(condensed_path=condensed_path)
    assert themes_path.exists()
    payload = json.loads(themes_path.read_text(encoding="utf-8"))
    assert len(payload) == 2
    assert any(item.get("theme") for item in payload)
    assert catalog.records and catalog.records[0][1] == "with_themes"


def test_thematizer_fallback_on_failure(tmp_path):
    config = ThematizeConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    condensed_path = config.condensed_path(episode)
    condensed_path.parent.mkdir(parents=True, exist_ok=True)
    condensed_path.write_text(json.dumps([{"start": 0.0, "end": 1.0, "text": "Nothing works today."}]), encoding="utf-8")

    thematizer = EpisodeThematizer(
        podcast=episode,
        config=config,
        catalog=StubCatalog(),
        client=FailingClient(),  # type: ignore[arg-type]
    )

    themes_path = thematizer.thematize(condensed_path=condensed_path)
    payload = json.loads(themes_path.read_text(encoding="utf-8"))
    assert payload[0]["theme"] == config.fallback_theme
    assert "LLM fallback reason" in payload[0]["summary"]


def test_thematizer_yield_progress(tmp_path):
    """Test that yield_progress=True returns a generator."""
    config = ThematizeConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    condensed_path = config.condensed_path(episode)
    condensed_path.parent.mkdir(parents=True, exist_ok=True)
    condensed_path.write_text(
        json.dumps(
            [
                {"start": 0.0, "end": 1.0, "speaker_id": "S0", "speaker_name": "Host", "text": "Welcome to the show."}
            ]
        ),
        encoding="utf-8",
    )

    thematizer = EpisodeThematizer(
        podcast=episode,
        config=config,
        catalog=StubCatalog(),
        client=DummyClient(),  # type: ignore[arg-type]
    )

    # Test yield_progress=True returns generator
    result = thematizer.thematize(condensed_path=condensed_path, yield_progress=True)
    assert hasattr(result, "__iter__")
    assert hasattr(result, "__next__")

    # Consume the generator
    events = []
    try:
        while True:
            event = next(result)
            events.append(event)
    except StopIteration as stop:
        final_path = stop.value
        assert isinstance(final_path, Path)
        assert final_path.exists()

    assert len(events) > 0
    assert all(isinstance(e, PipelineEvent) for e in events)


def test_thematizer_missing_condensed_file(tmp_path):
    """Test that FileNotFoundError is raised when condensed file doesn't exist."""
    config = ThematizeConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    condensed_path = config.condensed_path(episode)
    # Don't create the file

    thematizer = EpisodeThematizer(
        podcast=episode,
        config=config,
        catalog=StubCatalog(),
        client=DummyClient(),  # type: ignore[arg-type]
    )

    with pytest.raises(FileNotFoundError, match="Condensed transcript not found"):
        thematizer.thematize(condensed_path=condensed_path)


def test_thematizer_word_cap_enforcement(tmp_path):
    """Test that summary word count is capped at 50 words."""
    config = ThematizeConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    condensed_path = config.condensed_path(episode)
    condensed_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a response with a very long summary
    long_summary = " ".join(["word"] * 100)  # 100 words
    condensed_path.write_text(
        json.dumps(
            [
                {"start": 0.0, "end": 1.0, "speaker_id": "S0", "speaker_name": "Host", "text": "Test text."}
            ]
        ),
        encoding="utf-8",
    )

    class LongSummaryClient:
        def generate(self, prompt: str) -> str:
            return json.dumps({"theme": "Test", "summary": long_summary})

    thematizer = EpisodeThematizer(
        podcast=episode,
        config=config,
        catalog=StubCatalog(),
        client=LongSummaryClient(),  # type: ignore[arg-type]
    )

    themes_path = thematizer.thematize(condensed_path=condensed_path)
    payload = json.loads(themes_path.read_text(encoding="utf-8"))
    summary = payload[0]["summary"]
    word_count = len(summary.split())
    assert word_count <= 50


def test_thematizer_list_response(tmp_path):
    """Test handling of list response from LLM."""
    config = ThematizeConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    condensed_path = config.condensed_path(episode)
    condensed_path.parent.mkdir(parents=True, exist_ok=True)
    condensed_path.write_text(
        json.dumps(
            [
                {"start": 0.0, "end": 1.0, "speaker_id": "S0", "speaker_name": "Host", "text": "Test text."}
            ]
        ),
        encoding="utf-8",
    )

    class ListResponseClient:
        def generate(self, prompt: str) -> str:
            return '[{"theme": "From List", "summary": "List response summary"}]'

    thematizer = EpisodeThematizer(
        podcast=episode,
        config=config,
        catalog=StubCatalog(),
        client=ListResponseClient(),  # type: ignore[arg-type]
    )

    themes_path = thematizer.thematize(condensed_path=condensed_path)
    payload = json.loads(themes_path.read_text(encoding="utf-8"))
    assert payload[0]["theme"] == "From List"
    assert payload[0]["summary"] == "List response summary"


def test_thematizer_empty_list_response(tmp_path):
    """Test handling of empty list response from LLM."""
    config = ThematizeConfig(data_root=tmp_path / "data")
    episode = _episode(tmp_path)
    condensed_path = config.condensed_path(episode)
    condensed_path.parent.mkdir(parents=True, exist_ok=True)
    condensed_path.write_text(
        json.dumps(
            [
                {"start": 0.0, "end": 1.0, "speaker_id": "S0", "speaker_name": "Host", "text": "Test text."}
            ]
        ),
        encoding="utf-8",
    )

    class EmptyListClient:
        def generate(self, prompt: str) -> str:
            return "[]"

    thematizer = EpisodeThematizer(
        podcast=episode,
        config=config,
        catalog=StubCatalog(),
        client=EmptyListClient(),  # type: ignore[arg-type]
    )

    themes_path = thematizer.thematize(condensed_path=condensed_path)
    payload = json.loads(themes_path.read_text(encoding="utf-8"))
    # Should use fallback theme name
    assert "Segment" in payload[0]["theme"] or payload[0]["theme"] == config.fallback_theme


