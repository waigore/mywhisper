from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from mywhisper.models import PodcastEpisode
from mywhisper.myw.config import MywConfig
from mywhisper.myw.services.catalog import CatalogService
from mywhisper.podcasts import PodcastCatalog


def test_catalog_service_init_with_catalog(tmp_path):
    """Test CatalogService initialization with provided catalog."""
    config = MywConfig(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "db.sqlite",
        podcast_cache_path=tmp_path / "cache",
        podcast_db_path=tmp_path / "podcasts.db",
        log_level="INFO",
    )
    catalog = PodcastCatalog(data_root=tmp_path / "data")
    
    service = CatalogService(config, catalog=catalog)
    assert service.config == config
    assert service.catalog == catalog


def test_catalog_service_init_without_catalog(tmp_path):
    """Test CatalogService initialization without catalog (creates default)."""
    config = MywConfig(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "db.sqlite",
        podcast_cache_path=tmp_path / "cache",
        podcast_db_path=tmp_path / "podcasts.db",
        log_level="INFO",
    )
    
    service = CatalogService(config)
    assert service.config == config
    assert service.catalog is not None
    assert isinstance(service.catalog, PodcastCatalog)


def test_sync_from_cache_with_episodes(tmp_path):
    """Test sync_from_cache when importer finds episodes."""
    config = MywConfig(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "db.sqlite",
        podcast_cache_path=tmp_path / "cache",
        podcast_db_path=tmp_path / "podcasts.db",
        log_level="INFO",
    )
    catalog = PodcastCatalog(data_root=tmp_path / "data")
    service = CatalogService(config, catalog=catalog)
    
    # Create a test episode
    audio_path = tmp_path / "episode.m4a"
    audio_path.write_bytes(b"audio data")
    episode = PodcastEpisode(
        episode_id="test-ep-1",
        show_title="Test Show",
        episode_title="Test Episode",
        source_path=audio_path,
        description="Test description",
        duration_sec=3600.0,
    )
    catalog.upsert_episode(episode)
    
    # Mock the importer to return episodes, preserving the static method
    from mywhisper.podcasts import ApplePodcastsImporter
    mock_episodes = [episode]
    with patch("mywhisper.myw.services.catalog.ApplePodcastsImporter") as mock_importer_class:
        mock_importer = Mock()
        mock_importer.register_in_catalog.return_value = mock_episodes
        # Preserve the static method _first_non_empty
        mock_importer_class._first_non_empty = ApplePodcastsImporter._first_non_empty
        mock_importer_class.return_value = mock_importer
        
        view_states = service.sync_from_cache()
        
        assert len(view_states) == 1
        assert view_states[0].episode_id == "test-ep-1"
        assert view_states[0].show_title == "Test Show"
        assert view_states[0].episode_title == "Test Episode"


def test_sync_from_cache_without_episodes(tmp_path):
    """Test sync_from_cache when importer finds no episodes (falls back to catalog)."""
    config = MywConfig(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "db.sqlite",
        podcast_cache_path=tmp_path / "cache",
        podcast_db_path=tmp_path / "podcasts.db",
        log_level="INFO",
    )
    catalog = PodcastCatalog(data_root=tmp_path / "data")
    service = CatalogService(config, catalog=catalog)
    
    # Create a test episode in catalog
    audio_path = tmp_path / "episode.m4a"
    audio_path.write_bytes(b"audio data")
    episode = PodcastEpisode(
        episode_id="test-ep-2",
        show_title="Test Show 2",
        episode_title="Test Episode 2",
        source_path=audio_path,
    )
    catalog.upsert_episode(episode)
    
    # Mock the importer to return no episodes
    with patch("mywhisper.myw.services.catalog.ApplePodcastsImporter") as mock_importer_class:
        mock_importer = Mock()
        mock_importer.register_in_catalog.return_value = []
        mock_importer_class.return_value = mock_importer
        
        view_states = service.sync_from_cache()
        
        assert len(view_states) == 1
        assert view_states[0].episode_id == "test-ep-2"


def test_list_view_states(tmp_path):
    """Test list_view_states returns episodes from catalog."""
    config = MywConfig(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "db.sqlite",
        podcast_cache_path=tmp_path / "cache",
        podcast_db_path=tmp_path / "podcasts.db",
        log_level="INFO",
    )
    catalog = PodcastCatalog(data_root=tmp_path / "data")
    service = CatalogService(config, catalog=catalog)
    
    # Create multiple episodes with different timestamps
    audio_path1 = tmp_path / "episode1.m4a"
    audio_path1.write_bytes(b"audio data")
    audio_path1.touch()
    
    audio_path2 = tmp_path / "episode2.m4a"
    audio_path2.write_bytes(b"audio data")
    audio_path2.touch()
    
    episode1 = PodcastEpisode(
        episode_id="test-ep-1",
        show_title="Show A",
        episode_title="Episode 1",
        source_path=audio_path1,
    )
    episode2 = PodcastEpisode(
        episode_id="test-ep-2",
        show_title="Show B",
        episode_title="Episode 2",
        source_path=audio_path2,
    )
    catalog.upsert_episode(episode1)
    catalog.upsert_episode(episode2)
    
    view_states = service.list_view_states()
    assert len(view_states) == 2
    # Should be sorted by downloaded_at descending (newest first)
    assert all(vs.episode_id in ["test-ep-1", "test-ep-2"] for vs in view_states)


def test_to_view_states_sorts_by_downloaded_at(tmp_path):
    """Test _to_view_states sorts episodes by downloaded_at descending."""
    config = MywConfig(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "db.sqlite",
        podcast_cache_path=tmp_path / "cache",
        podcast_db_path=tmp_path / "podcasts.db",
        log_level="INFO",
    )
    service = CatalogService(config)
    
    # Create episodes with different timestamps
    audio_path1 = tmp_path / "episode1.m4a"
    audio_path1.write_bytes(b"audio data")
    audio_path1.touch()
    
    audio_path2 = tmp_path / "episode2.m4a"
    audio_path2.write_bytes(b"audio data")
    audio_path2.touch()
    
    # Set different mtimes
    old_time = datetime(2020, 1, 1).timestamp()
    new_time = datetime(2024, 1, 1).timestamp()
    
    import os
    os.utime(audio_path1, (old_time, old_time))
    os.utime(audio_path2, (new_time, new_time))
    
    episode1 = PodcastEpisode(
        episode_id="old-ep",
        show_title="Show",
        episode_title="Old Episode",
        source_path=audio_path1,
    )
    episode2 = PodcastEpisode(
        episode_id="new-ep",
        show_title="Show",
        episode_title="New Episode",
        source_path=audio_path2,
    )
    
    view_states = service._to_view_states([episode1, episode2])
    assert len(view_states) == 2
    # Newer episode should be first
    assert view_states[0].episode_id == "new-ep"
    assert view_states[1].episode_id == "old-ep"


def test_to_view_states_sorts_with_none_downloaded_at(tmp_path):
    """Test _to_view_states handles episodes with None downloaded_at."""
    config = MywConfig(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "db.sqlite",
        podcast_cache_path=tmp_path / "cache",
        podcast_db_path=tmp_path / "podcasts.db",
        log_level="INFO",
    )
    service = CatalogService(config)
    
    # Create episode with non-existent path (will have None downloaded_at)
    non_existent_path = tmp_path / "nonexistent.m4a"
    episode1 = PodcastEpisode(
        episode_id="no-date-ep",
        show_title="Show",
        episode_title="No Date Episode",
        source_path=non_existent_path,
    )
    
    # Create episode with valid path
    audio_path2 = tmp_path / "episode2.m4a"
    audio_path2.write_bytes(b"audio data")
    audio_path2.touch()
    episode2 = PodcastEpisode(
        episode_id="dated-ep",
        show_title="Show",
        episode_title="Dated Episode",
        source_path=audio_path2,
    )
    
    view_states = service._to_view_states([episode1, episode2])
    assert len(view_states) == 2
    # Episode with date should come first
    assert view_states[0].episode_id == "dated-ep"
    assert view_states[1].episode_id == "no-date-ep"


def test_to_view_state_with_metadata(tmp_path):
    """Test _to_view_state extracts metadata correctly."""
    config = MywConfig(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "db.sqlite",
        podcast_cache_path=tmp_path / "cache",
        podcast_db_path=tmp_path / "podcasts.db",
        log_level="INFO",
    )
    service = CatalogService(config)
    
    audio_path = tmp_path / "episode.m4a"
    audio_path.write_bytes(b"audio data")
    audio_path.touch()
    
    episode = PodcastEpisode(
        episode_id="test-ep",
        show_title="Default Show",
        episode_title="Default Episode",
        source_path=audio_path,
        description="Episode description",
        duration_sec=1800.0,
        metadata={
            "podcastTitle": "Metadata Show",
            "episodeTitle": "Metadata Episode",
            "podcasts_db": {
                "podcast_title": "DB Show",
                "episode_title": "DB Episode",
            },
            "mdls": {
                "kMDItemAlbum": "MDLS Show",
                "kMDItemTitle": "MDLS Episode",
            },
        },
    )
    
    view_state = service._to_view_state(episode)
    assert view_state.episode_id == "test-ep"
    assert view_state.show_title == "Metadata Show"  # First non-empty from metadata
    assert view_state.episode_title == "Metadata Episode"
    assert view_state.description == "Episode description"
    assert view_state.duration_sec == 1800.0
    assert view_state.downloaded_at is not None
    assert view_state.file_size == len(b"audio data")
    assert view_state.audio_path == audio_path


def test_to_view_state_with_db_meta_only(tmp_path):
    """Test _to_view_state uses db_meta when main metadata missing."""
    config = MywConfig(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "db.sqlite",
        podcast_cache_path=tmp_path / "cache",
        podcast_db_path=tmp_path / "podcasts.db",
        log_level="INFO",
    )
    service = CatalogService(config)
    
    audio_path = tmp_path / "episode.m4a"
    audio_path.write_bytes(b"audio data")
    audio_path.touch()
    
    episode = PodcastEpisode(
        episode_id="test-ep",
        show_title="Default Show",
        episode_title="Default Episode",
        source_path=audio_path,
        metadata={
            "podcasts_db": {
                "podcast_title": "DB Show",
                "episode_title": "DB Episode",
            },
        },
    )
    
    view_state = service._to_view_state(episode)
    assert view_state.show_title == "DB Show"
    assert view_state.episode_title == "DB Episode"


def test_to_view_state_with_mdls_meta_only(tmp_path):
    """Test _to_view_state uses mdls_meta when other metadata missing."""
    config = MywConfig(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "db.sqlite",
        podcast_cache_path=tmp_path / "cache",
        podcast_db_path=tmp_path / "podcasts.db",
        log_level="INFO",
    )
    service = CatalogService(config)
    
    audio_path = tmp_path / "episode.m4a"
    audio_path.write_bytes(b"audio data")
    audio_path.touch()
    
    episode = PodcastEpisode(
        episode_id="test-ep",
        show_title="Default Show",
        episode_title="Default Episode",
        source_path=audio_path,
        metadata={
            "mdls": {
                "kMDItemAlbum": "MDLS Show",
                "kMDItemTitle": "MDLS Episode",
            },
        },
    )
    
    view_state = service._to_view_state(episode)
    assert view_state.show_title == "MDLS Show"
    assert view_state.episode_title == "MDLS Episode"


def test_to_view_state_falls_back_to_episode_fields(tmp_path):
    """Test _to_view_state falls back to episode fields when metadata missing."""
    config = MywConfig(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "db.sqlite",
        podcast_cache_path=tmp_path / "cache",
        podcast_db_path=tmp_path / "podcasts.db",
        log_level="INFO",
    )
    service = CatalogService(config)
    
    audio_path = tmp_path / "episode.m4a"
    audio_path.write_bytes(b"audio data")
    audio_path.touch()
    
    episode = PodcastEpisode(
        episode_id="test-ep",
        show_title="Episode Show",
        episode_title="Episode Title",
        source_path=audio_path,
        metadata={},
    )
    
    view_state = service._to_view_state(episode)
    assert view_state.show_title == "Episode Show"
    assert view_state.episode_title == "Episode Title"


def test_to_view_state_unknown_show_fallback(tmp_path):
    """Test _to_view_state uses 'Unknown Show' when show_title is empty."""
    config = MywConfig(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "db.sqlite",
        podcast_cache_path=tmp_path / "cache",
        podcast_db_path=tmp_path / "podcasts.db",
        log_level="INFO",
    )
    service = CatalogService(config)
    
    audio_path = tmp_path / "episode.m4a"
    audio_path.write_bytes(b"audio data")
    audio_path.touch()
    
    episode = PodcastEpisode(
        episode_id="test-ep",
        show_title="",
        episode_title="Episode Title",
        source_path=audio_path,
        metadata={},
    )
    
    view_state = service._to_view_state(episode)
    assert view_state.show_title == "Unknown Show"
    assert view_state.episode_title == "Episode Title"


def test_to_view_state_episode_title_fallback_to_stem(tmp_path):
    """Test _to_view_state falls back to source_path.stem for episode_title."""
    config = MywConfig(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "db.sqlite",
        podcast_cache_path=tmp_path / "cache",
        podcast_db_path=tmp_path / "podcasts.db",
        log_level="INFO",
    )
    service = CatalogService(config)
    
    audio_path = tmp_path / "my_episode_file.m4a"
    audio_path.write_bytes(b"audio data")
    audio_path.touch()
    
    episode = PodcastEpisode(
        episode_id="test-ep",
        show_title="Show",
        episode_title="",
        source_path=audio_path,
        metadata={},
    )
    
    view_state = service._to_view_state(episode)
    assert view_state.episode_title == "my_episode_file"


def test_to_view_state_with_invalid_metadata_types(tmp_path):
    """Test _to_view_state handles invalid metadata types gracefully."""
    config = MywConfig(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "db.sqlite",
        podcast_cache_path=tmp_path / "cache",
        podcast_db_path=tmp_path / "podcasts.db",
        log_level="INFO",
    )
    service = CatalogService(config)
    
    audio_path = tmp_path / "episode.m4a"
    audio_path.write_bytes(b"audio data")
    audio_path.touch()
    
    episode = PodcastEpisode(
        episode_id="test-ep",
        show_title="Show",
        episode_title="Episode",
        source_path=audio_path,
        metadata={
            "podcasts_db": "not a dict",  # Invalid type
            "mdls": 123,  # Invalid type
        },
    )
    
    view_state = service._to_view_state(episode)
    # Should fall back to episode fields
    assert view_state.show_title == "Show"
    assert view_state.episode_title == "Episode"


def test_to_view_state_with_stat_failure(tmp_path):
    """Test _to_view_state handles file stat failures."""
    config = MywConfig(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "db.sqlite",
        podcast_cache_path=tmp_path / "cache",
        podcast_db_path=tmp_path / "podcasts.db",
        log_level="INFO",
    )
    service = CatalogService(config)
    
    # Non-existent path
    non_existent_path = tmp_path / "nonexistent.m4a"
    episode = PodcastEpisode(
        episode_id="test-ep",
        show_title="Show",
        episode_title="Episode",
        source_path=non_existent_path,
    )
    
    view_state = service._to_view_state(episode)
    assert view_state.downloaded_at is None
    assert view_state.file_size is None


def test_to_view_state_with_imported_at_timestamp(tmp_path):
    """Test _to_view_state uses imported_at when file stat fails."""
    config = MywConfig(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "db.sqlite",
        podcast_cache_path=tmp_path / "cache",
        podcast_db_path=tmp_path / "podcasts.db",
        log_level="INFO",
    )
    service = CatalogService(config)
    
    non_existent_path = tmp_path / "nonexistent.m4a"
    imported_time = datetime(2023, 6, 15, 10, 30, 0)
    episode = PodcastEpisode(
        episode_id="test-ep",
        show_title="Show",
        episode_title="Episode",
        source_path=non_existent_path,
        metadata={
            "imported_at": imported_time.isoformat(),
        },
    )
    
    view_state = service._to_view_state(episode)
    assert view_state.downloaded_at == imported_time


def test_to_view_state_with_invalid_imported_at(tmp_path):
    """Test _to_view_state handles invalid imported_at timestamp."""
    config = MywConfig(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "db.sqlite",
        podcast_cache_path=tmp_path / "cache",
        podcast_db_path=tmp_path / "podcasts.db",
        log_level="INFO",
    )
    service = CatalogService(config)
    
    non_existent_path = tmp_path / "nonexistent.m4a"
    episode = PodcastEpisode(
        episode_id="test-ep",
        show_title="Show",
        episode_title="Episode",
        source_path=non_existent_path,
        metadata={
            "imported_at": "not a valid timestamp",
        },
    )
    
    view_state = service._to_view_state(episode)
    assert view_state.downloaded_at is None


def test_to_view_state_with_non_string_imported_at(tmp_path):
    """Test _to_view_state handles non-string imported_at."""
    config = MywConfig(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "db.sqlite",
        podcast_cache_path=tmp_path / "cache",
        podcast_db_path=tmp_path / "podcasts.db",
        log_level="INFO",
    )
    service = CatalogService(config)
    
    non_existent_path = tmp_path / "nonexistent.m4a"
    episode = PodcastEpisode(
        episode_id="test-ep",
        show_title="Show",
        episode_title="Episode",
        source_path=non_existent_path,
        metadata={
            "imported_at": 12345,  # Not a string
        },
    )
    
    view_state = service._to_view_state(episode)
    # Should not use non-string imported_at
    assert view_state.downloaded_at is None


def test_to_view_state_file_size_stat_failure(tmp_path):
    """Test _to_view_state handles file size stat failure separately."""
    config = MywConfig(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "db.sqlite",
        podcast_cache_path=tmp_path / "cache",
        podcast_db_path=tmp_path / "podcasts.db",
        log_level="INFO",
    )
    service = CatalogService(config)
    
    # Create a mock Path object that fails on second stat call
    audio_path = tmp_path / "episode.m4a"
    audio_path.write_bytes(b"audio data")
    audio_path.touch()
    
    # Create a mock stat result
    from os import stat_result
    stat_result_obj = audio_path.stat()
    
    call_count = {"count": 0}
    
    def mock_stat():
        call_count["count"] += 1
        if call_count["count"] == 2:  # Second call for file_size
            raise OSError("Permission denied")
        return stat_result_obj
    
    # Create a mock Path with stat method
    mock_path = MagicMock(spec=Path)
    mock_path.stat = mock_stat
    mock_path.stem = audio_path.stem
    
    episode = PodcastEpisode(
        episode_id="test-ep",
        show_title="Show",
        episode_title="Episode",
        source_path=mock_path,
    )
    
    view_state = service._to_view_state(episode)
    # downloaded_at should work (first stat call), but file_size should be None (second stat call fails)
    assert view_state.downloaded_at is not None
    assert view_state.file_size is None


def test_to_view_state_episode_key(tmp_path):
    """Test _to_view_state includes episode_key."""
    config = MywConfig(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "db.sqlite",
        podcast_cache_path=tmp_path / "cache",
        podcast_db_path=tmp_path / "podcasts.db",
        log_level="INFO",
    )
    service = CatalogService(config)
    
    audio_path = tmp_path / "episode.m4a"
    audio_path.write_bytes(b"audio data")
    audio_path.touch()
    
    episode = PodcastEpisode(
        episode_id="test-ep",
        show_title="Show",
        episode_title="Episode",
        source_path=audio_path,
    )
    
    view_state = service._to_view_state(episode)
    assert view_state.episode_key is not None
    assert len(view_state.episode_key) == 8
    assert view_state.episode_key.isdigit()


def test_to_view_state_all_fields(tmp_path):
    """Test _to_view_state sets all EpisodeViewState fields correctly."""
    config = MywConfig(
        data_dir=tmp_path / "data",
        db_path=tmp_path / "db.sqlite",
        podcast_cache_path=tmp_path / "cache",
        podcast_db_path=tmp_path / "podcasts.db",
        log_level="INFO",
    )
    service = CatalogService(config)
    
    audio_path = tmp_path / "episode.m4a"
    audio_path.write_bytes(b"audio data")
    audio_path.touch()
    
    episode = PodcastEpisode(
        episode_id="test-ep-123",
        show_title="My Show",
        episode_title="My Episode",
        source_path=audio_path,
        description="A great episode",
        duration_sec=3600.5,
    )
    
    view_state = service._to_view_state(episode)
    
    assert view_state.episode_id == "test-ep-123"
    assert view_state.episode_key is not None
    assert view_state.show_title == "My Show"
    assert view_state.episode_title == "My Episode"
    assert view_state.downloaded_at is not None
    assert view_state.status == "Downloaded"
    assert view_state.remarks == ""
    assert view_state.description == "A great episode"
    assert view_state.duration_sec == 3600.5
    assert view_state.file_size == len(b"audio data")
    assert view_state.audio_path == audio_path

