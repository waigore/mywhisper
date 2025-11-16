from __future__ import annotations

from typing import Optional

from ..models import PipelineEvent
from .models import PipelineCheckpoint
from .store import CheckpointStore


class PipelineEventAdapter:
    """
    Consume PipelineEvent objects, derive checkpoint data, and persist them.
    """

    def __init__(
        self,
        store: CheckpointStore,
        episode_id: Optional[str] = None,
        pipeline_id: Optional[str] = None,
    ) -> None:
        self.store = store
        self._episode_id = episode_id
        self._pipeline_id = pipeline_id

    def process(self, event: PipelineEvent) -> PipelineCheckpoint:
        episode_id = event.episode_id or self._episode_id
        if not episode_id:
            raise ValueError("episode_id must be provided on the event or adapter.")

        step = event.step_name or event.stage
        status = (event.checkpoint or {}).get("status") or event.stage

        checkpoint = PipelineCheckpoint(
            pipeline_id=self._pipeline_id,
            episode_id=episode_id,
            step=step,
            status=status,
            stage=event.stage,
            message=event.message,
            payload=dict(event.payload),
            details=dict(event.checkpoint),
            artefact_paths={name: str(path) for name, path in event.artefact_paths.items()},
            elapsed=event.elapsed,
        )

        self.store.upsert(checkpoint)
        return checkpoint

