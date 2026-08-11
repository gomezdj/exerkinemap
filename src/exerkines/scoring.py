"""Exerkine scoring scaffolding."""

from __future__ import annotations

from typing import Iterable, List


class ExerkineScorer:
    """Simple placeholder for scoring exerkine candidates."""

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def score(self, values: Iterable[float]) -> List[float]:
        return [value * self.weight for value in values]


def score_exerkines(values: Iterable[float], weight: float = 1.0) -> List[float]:
    return ExerkineScorer(weight=weight).score(values)
