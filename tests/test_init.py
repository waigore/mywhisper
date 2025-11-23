from __future__ import annotations

import logging

from mywhisper import configure_logging


def test_configure_logging_sets_level():
    logger = configure_logging(level="DEBUG")
    assert logger.level == logging.DEBUG
    assert logger.name == "mywhisper"


def test_configure_logging_with_handlers():
    """Test configure_logging with custom handlers"""
    handler = logging.NullHandler()
    logger = configure_logging(level="INFO", handlers=[handler])
    assert logger.level == logging.INFO
    assert handler in logger.handlers


def test_configure_logging_with_numeric_level():
    """Test configure_logging with numeric level"""
    logger = configure_logging(level=logging.WARNING)
    assert logger.level == logging.WARNING

