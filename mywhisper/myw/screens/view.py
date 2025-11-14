from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from rich.table import Table

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Static

from ..models import EpisodeViewState

LOGGER = logging.getLogger("mywhisper.myw.screen.view")


class PodcastViewScreen(Screen):
    """
    Detail view for a single podcast episode.
    """

    BINDINGS = [
        Binding("b", "back", "Back", show=True),
        Binding("e", "enqueue", "Enqueue", show=True),
    ]

    def __init__(
        self,
        episode: EpisodeViewState,
        enqueue_callback: Callable[[str], None],
    ) -> None:
        super().__init__()
        self.episode = episode
        self._enqueue_callback = enqueue_callback

    def compose(self) -> ComposeResult:
        yield Header()
        self.detail = Static()
        yield self.detail
        yield Footer()

    async def on_mount(self) -> None:
        self._render_detail()

    async def action_enqueue(self) -> None:
        self._enqueue_callback(self.episode.episode_id)
        LOGGER.info("Episode %s enqueued from view screen", self.episode.episode_id)
        await self.app.pop_screen()

    async def action_back(self) -> None:
        LOGGER.info("Returning to listing from view screen for episode %s", self.episode.episode_id)
        await self.app.pop_screen()

    def _render_detail(self) -> None:
        table = Table.grid(padding=1)
        table.add_row("Podcast", self.episode.show_title)
        table.add_row("Episode", self.episode.episode_title)
        table.add_row("Status", self.episode.status)
        table.add_row("Remarks", self.episode.remarks or "—")
        table.add_row("Description", self.episode.description or "N/A")
        if self.episode.duration_sec is not None:
            table.add_row("Length", f"{self.episode.duration_sec / 60:.1f} min")
        if self.episode.file_size is not None:
            table.add_row("File Size", f"{self.episode.file_size / (1024 * 1024):.2f} MB")
        if self.episode.audio_path:
            table.add_row("Audio Path", str(Path(self.episode.audio_path)))
        table.add_row("Episode ID", self.episode.episode_id)
        table.add_row("Episode Key", self.episode.episode_key)
        self.detail.update(table)

