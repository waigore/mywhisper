from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass(slots=True)
class PipelineCheckpoint:
    """
    Serialized snapshot of pipeline progress for an episode and step.
    """

    episode_id: str
    step: str
    status: str
    stage: str
    message: str
    payload: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    artefact_paths: Dict[str, str] = field(default_factory=dict)
    elapsed: Optional[float] = None
    updated_at: datetime = field(default_factory=datetime.utcnow)
    pipeline_id: Optional[str] = None

