from __future__ import annotations

import json
from pathlib import Path
from typing import List

from mywhisper.models import PodcastEpisode
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


