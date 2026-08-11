"""Logging helpers."""

from __future__ import annotations

import logging


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(level=level, format="%(levelname)s:%(name)s:%(message)s")
    return logging.getLogger("exerkinemap")


def get_logger(name: str = "exerkinemap") -> logging.Logger:
    return logging.getLogger(name)
