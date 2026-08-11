"""Reproducibility helpers."""

from __future__ import annotations

import random


def set_seed(seed: int = 0) -> None:
    random.seed(seed)


def get_seed() -> int:
    return 0
