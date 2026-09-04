"""Structured logging configuration for the Play Store Audit Report Generator."""

from __future__ import annotations

import logging
import sys

# Every logger handed out by get_logger(), so set_global_level() can
# retarget all of them at once (e.g. for a CLI --verbose flag) without
# each module needing to know about that flag.
_created_loggers: list[logging.Logger] = []


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a configured module-level logger.

    Uses a single stream handler with a consistent structured format
    (timestamp, level, logger name, message). Safe to call repeatedly
    for the same name -- handlers are not duplicated.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        _created_loggers.append(logger)

    logger.setLevel(level)
    return logger


def set_global_level(level: int) -> None:
    """
    Update the level of every logger created so far via get_logger().

    Used by main.py's --verbose flag to reveal DEBUG-level detail
    (scraper request attempts, retry backoff, etc.) across every
    module at once. Safe to call any time after the modules that
    matter have already been imported -- which, in this project, is
    always true by the time main() runs, since all services are
    imported at module load time.
    """
    for logger in _created_loggers:
        logger.setLevel(level)
