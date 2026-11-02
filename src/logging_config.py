"""
Logging configuration for REQTRACE AI.

This module configures console and rotating-file logging without recording
environment-file contents, secrets, or complete engineering documents.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOGGER_NAME = "reqtrace_ai"

_LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | "
    "%(module)s.%(funcName)s:%(lineno)d | %(message)s"
)

_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    log_level: str,
    log_file_path: Path,
) -> logging.Logger:
    """
    Configure and return the REQTRACE AI root logger.

    Repeated calls do not add duplicate handlers.
    """

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.propagate = False

    if logger.handlers:
        return logger

    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(
        getattr(logging, log_level.upper(), logging.INFO)
    )
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        filename=log_file_path,
        maxBytes=2 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(
        getattr(logging, log_level.upper(), logging.INFO)
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def get_logger(module_name: str) -> logging.Logger:
    """Return a child logger under the REQTRACE AI logger."""

    return logging.getLogger(f"{LOGGER_NAME}.{module_name}")
