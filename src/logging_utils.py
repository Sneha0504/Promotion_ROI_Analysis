"""Shared logging configuration.

Every module obtains its logger through :func:`get_logger` so output is consistent
and uses the standard :mod:`logging` facility rather than ``print``.
"""

from __future__ import annotations

import logging

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        level: Logging level for the logger and its stream handler.

    Returns:
        A :class:`logging.Logger` with a single stream handler attached. Repeated
        calls for the same ``name`` reuse the existing handler (no duplicate lines).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
    return logger
