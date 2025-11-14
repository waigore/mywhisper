from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

from .. import configure_logging


def setup_logging(
    level: str,
    log_dir: Path,
    handlers: Optional[Iterable[logging.Handler]] = None,
) -> logging.Logger:
    """
    Configure myw application logging, bridging to the core mywhisper logger.
    """

    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "myw.log")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )

    all_handlers = list(handlers or [])
    all_handlers.append(file_handler)

    logger = configure_logging(level=level, handlers=all_handlers)
    logger.info("Initialized myw logging at level %s", level)
    return logger

