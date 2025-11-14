from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional

from ..models import PipelineStatus

StatusMap = Dict[str, tuple[str, str]]
LOGGER = logging.getLogger("mywhisper.myw.queue")


@dataclass(slots=True)
class QueueItem:
    episode_id: str
    resume: bool = False


class QueueController:
    """
    Manage the pipeline execution queue and current processing state.
    """

    def __init__(self) -> None:
        self._queue: Deque[QueueItem] = deque()
        self._statuses: StatusMap = {}
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._current: Optional[QueueItem] = None
        self._stop_requested = False
        self._shutdown = False

    def snapshot_status(self) -> StatusMap:
        with self._lock:
            return dict(self._statuses)

    def enqueue(self, episode_id: str, resume: bool = False) -> None:
        with self._condition:
            if any(item.episode_id == episode_id for item in self._queue):
                LOGGER.debug("Episode %s already queued", episode_id)
                return
            if self._current and self._current.episode_id == episode_id:
                LOGGER.debug("Episode %s already in progress", episode_id)
                return
            self._queue.append(QueueItem(episode_id=episode_id, resume=resume))
            self._statuses.setdefault(episode_id, ("Downloaded", ""))
            LOGGER.info("Enqueued episode %s (resume=%s)", episode_id, resume)
            self._condition.notify()

    def stop_current(self) -> None:
        with self._lock:
            if self._current:
                self._stop_requested = True
                LOGGER.info("Stop requested for episode %s", self._current.episode_id)

    def resume_episode(self, episode_id: str) -> None:
        with self._condition:
            if any(item.episode_id == episode_id for item in self._queue):
                return
            self._queue.appendleft(QueueItem(episode_id=episode_id, resume=True))
            self._statuses[episode_id] = ("In progress", "Resuming")
            self._stop_requested = False
            LOGGER.info("Resuming episode %s from checkpoint", episode_id)
            self._condition.notify()

    def next_item(self) -> Optional[QueueItem]:
        with self._condition:
            while not self._queue and not self._shutdown:
                self._condition.wait()
            if self._shutdown:
                return None
            item = self._queue.popleft()
            self._current = item
            self._stop_requested = False
            self._statuses[item.episode_id] = ("In progress", "")
            LOGGER.info("Starting pipeline for episode %s", item.episode_id)
            return item

    def release_current(self) -> None:
        with self._lock:
            self._current = None
            self._stop_requested = False
            LOGGER.debug("Released current episode lock")

    def request_shutdown(self) -> None:
        with self._condition:
            self._shutdown = True
            self._condition.notify_all()
            LOGGER.info("Queue controller shutdown requested")

    def should_stop(self) -> bool:
        with self._lock:
            return self._stop_requested

    def set_status(self, episode_id: str, status: str, remarks: str = "") -> None:
        with self._lock:
            self._statuses[episode_id] = (status, remarks)
            LOGGER.debug("Status updated for %s: %s - %s", episode_id, status, remarks)

    def get_status(self, episode_id: str) -> tuple[str, str]:
        with self._lock:
            return self._statuses.get(episode_id, ("Downloaded", ""))

    def current_status(self) -> PipelineStatus:
        with self._lock:
            if not self._current:
                return PipelineStatus(active=False)
            status, remarks = self._statuses.get(self._current.episode_id, ("In progress", ""))
            return PipelineStatus(
                active=True,
                episode_id=self._current.episode_id,
                step=status,
                progress=0.0,
                message=remarks,
            )

    def current_episode_id(self) -> Optional[str]:
        with self._lock:
            return self._current.episode_id if self._current else None

    def has_pending(self) -> bool:
        with self._lock:
            return bool(self._queue)

