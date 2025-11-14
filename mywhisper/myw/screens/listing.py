from __future__ import annotations

import logging
import asyncio
from typing import TYPE_CHECKING, Awaitable, Callable, Optional, cast

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Footer, Header

from ..messages import CatalogSynced, PipelineCompleted, PipelineProgress, PipelineStopped
from ..models import EpisodeViewState, PipelineStatus
from ..services.catalog import CatalogService
from ..services.queue import QueueController

if TYPE_CHECKING:
    from ..app import MywApp

LOGGER = logging.getLogger("mywhisper.myw.screen.listing")
from ..widgets.episode_table import EpisodeTable
from ..widgets.progress_bar import PipelineProgressBar


class PodcastListingScreen(Screen):
    """
    Default screen showing catalog of downloaded podcast episodes.
    """

    BINDINGS = [
        Binding("r", "refresh", "Refresh", show=True),
        Binding("v", "view", "View Episode", show=True),
        Binding("enter", "view", "View Episode", show=False),
        Binding("s", "toggle_stop", "Stop / Resume", show=True),
        Binding("q", "quit", "Quit", show=True),
    ]

    def __init__(
        self,
        catalog_service: CatalogService,
        queue: QueueController,
        view_callback: Callable[[str], "Awaitable[None]"],
        enqueue_callback: Callable[[str], None],
    ) -> None:
        super().__init__()
        self.catalog_service = catalog_service
        self.queue = queue
        self._view_callback = view_callback
        self._enqueue_callback = enqueue_callback
        self._episodes: list[EpisodeViewState] = []

    def compose(self) -> ComposeResult:
        yield Header()
        self.table = EpisodeTable()
        yield self.table
        self.progress = PipelineProgressBar()
        self.progress.visible = False
        yield self.progress
        yield Footer()

    async def on_mount(self) -> None:
        await self.refresh_catalog()

    async def action_refresh(self) -> None:
        LOGGER.info("Refresh triggered from listing screen")
        await self.refresh_catalog()

    async def action_view(self) -> None:
        episode_id = self.table.selected_episode_id()
        if episode_id:
            LOGGER.info("Viewing episode %s", episode_id)
            await self._view_callback(episode_id)

    async def action_toggle_stop(self) -> None:
        current = self.queue.current_episode_id()
        if current:
            LOGGER.info("Stop requested for current episode %s", current)
            self.queue.stop_current()
            return

        stopped = [
            episode_id
            for episode_id, (status, _) in self.queue.snapshot_status().items()
            if status == "Stopped"
        ]
        if stopped:
            LOGGER.info("Resuming stopped episode %s", stopped[0])
            self.queue.resume_episode(stopped[0])
        else:
            episode_id = self.table.selected_episode_id()
            if not episode_id:
                return
            # Enqueue selected episode if nothing processing.
            LOGGER.info(
                "No active pipeline; enqueueing selected episode %s",
                episode_id,
            )
            self._enqueue_callback(episode_id)

    async def action_quit(self) -> None:
        LOGGER.info("Quit requested from listing screen")
        await self.app.action_quit()

    async def refresh_catalog(self) -> None:
        episodes = await asyncio.to_thread(self.catalog_service.sync_from_cache)
        self._episodes = episodes
        app = cast("MywApp", self.app)
        app.register_episodes(episodes)
        self.post_message(CatalogSynced(episodes))

    def _render_table(self) -> None:
        statuses = self.queue.snapshot_status()
        self.table.load(self._episodes, statuses)

    @on(CatalogSynced)
    def handle_catalog_synced(self, message: CatalogSynced) -> None:
        self._episodes = message.episodes
        self._render_table()

    @on(PipelineProgress)
    def handle_pipeline_progress(self, message: PipelineProgress) -> None:
        payload = message.payload
        self.table.update_status(payload.episode_id, payload.status.step or "In progress", payload.remarks)
        self._show_progress(payload.status, payload.remarks)

    @on(PipelineStopped)
    def handle_pipeline_stopped(self, message: PipelineStopped) -> None:
        payload = message.payload
        self.table.update_status(payload.episode_id, "Stopped", payload.remarks)
        self.progress.hide()

    @on(PipelineCompleted)
    def handle_pipeline_completed(self, message: PipelineCompleted) -> None:
        payload = message.payload
        self.table.update_status(payload.episode_id, "Completed", payload.remarks)
        self.progress.hide()

    def _show_progress(self, status: PipelineStatus, remarks: str) -> None:
        if not status.active:
            self.progress.hide()
            return
        step = status.step or "In progress"
        self.progress.show(step=step, progress=status.progress or 0.0, message=remarks or status.message)

