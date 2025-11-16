from __future__ import annotations

import logging
from datetime import datetime
from typing import Iterable, List, Optional

from ...models import PodcastEpisode
from ...podcasts import ApplePodcastsImporter, PodcastCatalog
from ..config import MywConfig
from ..models import EpisodeViewState

LOGGER = logging.getLogger("mywhisper.myw.catalog")


class CatalogService:
    """
    Synchronise the podcast catalog with the local Apple Podcasts cache.
    """

    def __init__(
        self,
        config: MywConfig,
        catalog: Optional[PodcastCatalog] = None,
    ) -> None:
        self.config = config
        self.catalog = catalog or PodcastCatalog(data_root=config.data_dir)

    def sync_from_cache(self) -> List[EpisodeViewState]:
        importer = ApplePodcastsImporter(
            cache_root=self.config.podcast_cache_path,
            catalog=self.catalog,
            db_path=self.config.podcast_db_path,
        )
        episodes: List[PodcastEpisode] = []
        for episode in importer.register_in_catalog():
            LOGGER.info("Registered episode %s", episode.episode_id)
            episodes.append(episode)

        if not episodes:
            episodes = list(self.catalog.list_episodes())

        return self._to_view_states(episodes)

    def list_view_states(self) -> List[EpisodeViewState]:
        episodes = list(self.catalog.list_episodes())
        return self._to_view_states(episodes)

    def _to_view_states(self, episodes: Iterable[PodcastEpisode]) -> List[EpisodeViewState]:
        rows = [self._to_view_state(episode) for episode in episodes]
        rows.sort(key=lambda row: row.downloaded_at or datetime.min, reverse=True)
        return rows

    def _to_view_state(self, episode: PodcastEpisode) -> EpisodeViewState:
        metadata = episode.metadata or {}
        db_meta = metadata.get("podcasts_db")
        if not isinstance(db_meta, dict):
            db_meta = {}
        mdls_meta = metadata.get("mdls")
        if not isinstance(mdls_meta, dict):
            mdls_meta = {}

        show_title = ApplePodcastsImporter._first_non_empty(  # type: ignore[attr-defined]
            metadata.get("podcastTitle"),
            db_meta.get("podcast_title"),
            mdls_meta.get("kMDItemAlbum"),
            episode.show_title,
        )
        if not show_title:
            show_title = "Unknown Show"

        episode_title = ApplePodcastsImporter._first_non_empty(  # type: ignore[attr-defined]
            metadata.get("episodeTitle"),
            db_meta.get("episode_title"),
            mdls_meta.get("kMDItemTitle"),
            episode.episode_title,
            episode.source_path.stem if episode.source_path else None,
        )
        downloaded_at: Optional[datetime] = None
        try:
            stat_result = episode.source_path.stat()
            downloaded_at = datetime.fromtimestamp(stat_result.st_mtime)
        except OSError:
            LOGGER.debug("Unable to stat %s", episode.source_path)
        if downloaded_at is None:
            imported = metadata.get("imported_at")
            if isinstance(imported, str):
                try:
                    downloaded_at = datetime.fromisoformat(imported)
                except ValueError:
                    LOGGER.debug("Invalid imported_at timestamp %s", imported)
        file_size = None
        try:
            file_size = episode.source_path.stat().st_size
        except OSError:
            LOGGER.debug("Unable to read size for %s", episode.source_path)

        episode_key = episode.episode_key

        return EpisodeViewState(
            episode_id=episode.episode_id,
            episode_key=episode_key,
            show_title=show_title,
            episode_title=episode_title,
            downloaded_at=downloaded_at,
            status="Downloaded",
            remarks="",
            description=episode.description,
            duration_sec=episode.duration_sec,
            file_size=file_size,
            audio_path=episode.source_path,
        )

