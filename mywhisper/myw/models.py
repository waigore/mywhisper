from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class EpisodeViewState:
    episode_id: str
    episode_key: str
    show_title: str
    episode_title: str
    downloaded_at: Optional[datetime]
    status: str
    remarks: str = ""
    description: Optional[str] = None
    duration_sec: Optional[float] = None
    file_size: Optional[int] = None
    audio_path: Optional[Path] = None


@dataclass(slots=True)
class PipelineStatus:
    active: bool = False
    episode_id: Optional[str] = None
    step: Optional[str] = None
    progress: float = 0.0
    message: str = ""

