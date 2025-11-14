from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable, Tuple

from textual.widgets import DataTable

from ..models import EpisodeViewState


class EpisodeTable(DataTable):
    """
    Sortable table displaying downloaded podcast episodes.
    """

    _EPISODE_TITLE_LIMIT = 50

    def __init__(self) -> None:
        super().__init__()
        self.cursor_type = "row"
        self._selected_row_key: str | None = None
        self._row_keys: list[str] = []
        self.add_columns("Episode", "Podcast", "Downloaded At", "Status", "Remarks")

    def load(
        self,
        episodes: Iterable[EpisodeViewState],
        statuses: Dict[str, Tuple[str, str]],
    ) -> None:
        previous_selection = self._selected_row_key
        self.clear()
        self._row_keys = []
        self._selected_row_key = None
        for episode in episodes:
            status, remarks = statuses.get(episode.episode_id, (episode.status, episode.remarks))
            downloaded = self._format_datetime(episode.downloaded_at)
            self.add_row(
                self._format_episode_title(episode.episode_title),
                episode.show_title,
                downloaded,
                status,
                remarks,
                key=episode.episode_id,
            )
            self._row_keys.append(str(episode.episode_id))

        if self._row_keys:
            if previous_selection and previous_selection in self._row_keys:
                self._selected_row_key = previous_selection
                target_index = self._row_keys.index(previous_selection)
            else:
                self._selected_row_key = self._row_keys[0]
                target_index = 0
            try:
                self.cursor_coordinate = (target_index, 0)
            except Exception:
                pass

    def update_status(self, episode_id: str, status: str, remarks: str) -> None:
        try:
            self.update_cell(episode_id, "Status", status)
            self.update_cell(episode_id, "Remarks", remarks)
        except KeyError:
            pass

    def selected_episode_id(self) -> str | None:
        if self._selected_row_key is not None:
            return self._selected_row_key
        if self._row_keys:
            return self._row_keys[0]
        return None

    def _set_selected_row_key(self, row_key: DataTable.RowKey | str | None) -> None:
        normalized = self._normalize_row_key(row_key)
        if normalized is not None:
            self._selected_row_key = normalized

    def _normalize_row_key(self, row_key: DataTable.RowKey | str | None) -> str | None:
        if row_key is None:
            return None
        value = getattr(row_key, "value", row_key)
        return str(value)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self._set_selected_row_key(event.row_key)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self._set_selected_row_key(event.row_key)

    @staticmethod
    def _format_datetime(value: datetime | None) -> str:
        if not value:
            return ""
        return value.strftime("%Y-%m-%d %H:%M")

    @classmethod
    def _format_episode_title(cls, title: str) -> str:
        limit = cls._EPISODE_TITLE_LIMIT
        if len(title) <= limit:
            return title
        truncated = title[: limit - 3].rstrip()
        if not truncated:
            truncated = title[: limit - 3]
        return f"{truncated}..."

