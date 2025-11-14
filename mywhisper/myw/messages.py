from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from textual.message import Message

from .models import EpisodeViewState, PipelineStatus


class CatalogSynced(Message):
    def __init__(self, episodes: list[EpisodeViewState]) -> None:
        self.episodes = episodes
        super().__init__()


@dataclass(slots=True)
class PipelineEventPayload:
    episode_id: str
    status: PipelineStatus
    remarks: str


class PipelineProgress(Message):
    def __init__(self, payload: PipelineEventPayload) -> None:
        self.payload = payload
        super().__init__()


class PipelineStopped(Message):
    def __init__(self, payload: PipelineEventPayload) -> None:
        self.payload = payload
        super().__init__()


class PipelineCompleted(Message):
    def __init__(self, payload: PipelineEventPayload) -> None:
        self.payload = payload
        super().__init__()

