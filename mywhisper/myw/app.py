from __future__ import annotations

import contextlib
import logging
from typing import Dict, List, Optional

from textual.app import App
from textual.screen import Screen

from ..checkpoints import CheckpointStore
from ..config import DEFAULT_DATA_ROOT
from ..podcasts import PodcastCatalog
from .config import ConfigError, MywConfig, load_config
from .logging import setup_logging
from .messages import PipelineCompleted, PipelineProgress, PipelineStopped
from .models import EpisodeViewState
from .screens.listing import PodcastListingScreen
from .screens.view import PodcastViewScreen
from .services.catalog import CatalogService
from .services.pipeline import PipelineRunner
from .services.queue import QueueController


class MywApp(App[None]):
    """
    Textual application entrypoint for the myw frontend.
    """

    CSS_PATH = None

    def __init__(self, config: Optional[MywConfig] = None) -> None:
        super().__init__()
        self.config = config or load_config()
        log_dir = self.config.data_dir / "logs"
        self.logger = setup_logging(self.config.log_level, log_dir)
        self.logger.info("Starting myw with data directory %s", self.config.data_dir)

        self.catalog = PodcastCatalog(data_root=self.config.data_dir)
        self.catalog_service = CatalogService(self.config, self.catalog)
        self.queue = QueueController(enqueue_delay=3.0)
        self.checkpoints = CheckpointStore(self.config.db_path)
        self.pipeline_runner = PipelineRunner(
            config=self.config,
            catalog=self.catalog,
            queue=self.queue,
            checkpoints=self.checkpoints,
            callback=self._handle_pipeline_message,
        )
        self._episodes: Dict[str, EpisodeViewState] = {}

    async def on_mount(self) -> None:
        listing = PodcastListingScreen(
            catalog_service=self.catalog_service,
            queue=self.queue,
            view_callback=self.show_episode,
            enqueue_callback=self.enqueue_episode,
        )
        self.install_screen(listing, name="listing")
        await self.push_screen("listing")
        self.pipeline_runner.start()

    async def on_shutdown_request(self) -> None:
        self.pipeline_runner.shutdown()

    # ------------------------------------------------------------------ #
    # Episode state management
    # ------------------------------------------------------------------ #

    def register_episodes(self, episodes: List[EpisodeViewState]) -> None:
        for episode in episodes:
            self._episodes[episode.episode_id] = episode
            status, remarks = self.queue.get_status(episode.episode_id)
            episode.status = status
            episode.remarks = remarks
        self.logger.info("Registered %d episodes in view state", len(episodes))

    def enqueue_episode(self, episode_id: str) -> None:
        self.queue.enqueue(episode_id)
        self.queue.set_status(episode_id, "In progress", "Queued")
        episode = self._episodes.get(episode_id)
        if episode:
            episode.status = "In progress"
            episode.remarks = "Queued"
        self.logger.info("Episode %s enqueued for processing", episode_id)

    async def show_episode(self, episode_id: str) -> None:
        episode = self._episodes.get(episode_id)
        if not episode:
            return
        self.logger.info("Opening episode view for %s", episode_id)
        screen = PodcastViewScreen(episode=episode, enqueue_callback=self.enqueue_episode)
        name = f"episode-{episode_id}"
        self.install_screen(screen, name=name)
        await self.push_screen(name)
        
    # ------------------------------------------------------------------ #
    # Pipeline event handling
    # ------------------------------------------------------------------ #

    def _handle_pipeline_message(self, message) -> None:
        """
        Callback executed from the pipeline thread to post messages to the UI.
        """

        def _post() -> None:
            self.post_message(message)

        self.call_from_thread(_post)

    def on_pipeline_progress(self, message: PipelineProgress) -> None:
        episode = self._episodes.get(message.payload.episode_id)
        if episode:
            episode.status = "In progress"
            episode.remarks = message.payload.remarks

    def on_pipeline_stopped(self, message: PipelineStopped) -> None:
        episode = self._episodes.get(message.payload.episode_id)
        if episode:
            episode.status = "Stopped"
            episode.remarks = message.payload.remarks

    def on_pipeline_completed(self, message: PipelineCompleted) -> None:
        episode = self._episodes.get(message.payload.episode_id)
        if episode:
            episode.status = "Completed"
            episode.remarks = message.payload.remarks

