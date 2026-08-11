"""Filtering scaffolding for exerkine candidates."""

from __future__ import annotations

from typing import Iterable, List, TypeVar

T = TypeVar("T")


class ExerkineFilter:
    """Simple placeholder for filtering candidate exerkines."""

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def filter(self, values: Iterable[float]) -> List[float]:
        return [value for value in values if value >= self.threshold]


def filter_exerkines(values: Iterable[float], threshold: float = 0.0) -> List[float]:
    return ExerkineFilter(threshold=threshold).filter(values)
