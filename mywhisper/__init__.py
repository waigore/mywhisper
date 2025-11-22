"""
mywhisper package

Provides transcription, diarization, speaker assignment, and podcast catalog
tooling for long-form podcast audio pipelines.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

from .checkpoints import CheckpointStore, PipelineEventAdapter, PipelineCheckpoint
from .config import (
    DEFAULT_DATA_ROOT,
    derive_episode_key,
    ensure_data_subdir,
    generate_artefact_key,
)
from .logging_utils import LoggingBase, LoggingMeta, SerializationMode, log_function, serialize_input, serialize_output
from .models import (
    PipelineEvent,
    PodcastEpisode,
    TranscriptSegment,
    SpeakerAssignment,
    SpeakerProfile,
)

__all__ = [
    "configure_logging",
    "DEFAULT_DATA_ROOT",
    "ensure_data_subdir",
    "generate_artefact_key",
    "derive_episode_key",
    "PipelineCheckpoint",
    "CheckpointStore",
    "PipelineEventAdapter",
    "PipelineEvent",
    "PodcastEpisode",
    "TranscriptSegment",
    "SpeakerAssignment",
    "SpeakerProfile",
    "LoggingBase",
    "LoggingMeta",
    "log_function",
    "SerializationMode",
    "serialize_input",
    "serialize_output",
]

ROOT_LOGGER_NAME = "mywhisper"


def configure_logging(
    level: int | str = logging.INFO,
    handlers: Optional[Iterable[logging.Handler]] = None,
) -> logging.Logger:
    """
    Configure the root mywhisper logger.

    Parameters
    ----------
    level:
        Logging level or name. Defaults to logging.INFO.
    handlers:
        Optional iterable of handlers to attach to the root logger. When omitted,
        a StreamHandler with basic formatting is added if the logger has no
        handlers yet.
    """

    logger = logging.getLogger(ROOT_LOGGER_NAME)
    numeric_level = logging.getLevelName(level) if isinstance(level, str) else level
    logger.setLevel(numeric_level)

    if handlers:
        for handler in handlers:
            logger.addHandler(handler)
    elif not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.debug("Configured mywhisper root logger at level %s", numeric_level)
    return logger


