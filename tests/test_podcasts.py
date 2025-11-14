from __future__ import annotations

import plistlib
from datetime import datetime
from pathlib import Path

from mywhisper.models import PodcastEpisode
from mywhisper.podcasts import ApplePodcastsImporter, PodcastCatalog


def create_cache_entry(cache_root: Path) -> Path:
    entry = cache_root / "EpisodeCache"
    entry.mkdir(parents=True)
    audio_path = entry / "episode.m4a"
    audio_path.write_bytes(b"AUDIO")
    metadata = {
        "podcastTitle": "Sample Show",
        "episodeTitle": "Great Episode",
        "author": "Host Name",
        "episodeGuid": "guid-123",
        "releaseDate": datetime(2024, 1, 1).timestamp(),
        "episodeDescription": "An episode about testing.",
    }
    with (entry / "metadata.plist").open("wb") as handle:
        plistlib.dump(metadata, handle)
    return entry


def test_importer_registers_episode(tmp_path):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    create_cache_entry(cache_root)

    data_root = tmp_path / "data"
    catalog = PodcastCatalog(data_root=data_root)
    output_dir = tmp_path / "output"

    importer = ApplePodcastsImporter(
        cache_root=cache_root,
        catalog=catalog,
        output_dir=output_dir,
        move=False,
    )

    episodes = list(importer.register_in_catalog())
    assert len(episodes) == 1
    episode = episodes[0]
    assert episode.show_title == "Sample Show"

    stored = catalog.get_episode("guid-123")
    assert stored is not None
    assert stored.episode_title == "Great Episode"
    assert stored.source_path.exists()
    assert stored.description == "An episode about testing."
    assert stored.metadata.get("description") == "An episode about testing."


def test_catalog_lists_by_show(tmp_path):
    catalog = PodcastCatalog(data_root=tmp_path / "data")
    episode = PodcastEpisode(
        episode_id="id-1",
        show_title="Another Show",
        episode_title="Episode A",
        description="Sample description.",
        source_path=tmp_path / "audio.m4a",
        published_at=datetime(2024, 2, 1),
    )
    catalog.upsert_episode(episode)

    results = list(catalog.list_episodes(show_title="Another Show"))
    assert len(results) == 1
    assert results[0].description == "Sample description."

