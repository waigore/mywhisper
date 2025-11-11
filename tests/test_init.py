from __future__ import annotations

import logging

from mywhisper import configure_logging


def test_configure_logging_sets_level():
    logger = configure_logging(level="DEBUG")
    assert logger.level == logging.DEBUG
    assert logger.name == "mywhisper"

