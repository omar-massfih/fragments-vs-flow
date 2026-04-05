"""Shared logging helpers for the pipeline CLI."""

from __future__ import annotations

import json
import logging
import sys
from typing import Mapping


LOGGER_NAME = "pipeline"


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure the shared pipeline logger.

    Args:
        level: Logging level to apply to the shared logger.

    Returns:
        The configured shared logger.
    """
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the shared logger or one of its children."""
    parent_logger = configure_logging()
    if not name:
        return parent_logger
    return parent_logger.getChild(name)


def emit_json(payload: Mapping[str, object]) -> None:
    """Write one JSON payload to stdout with stable formatting."""
    sys.stdout.write(json.dumps(payload, indent=2) + "\n")
