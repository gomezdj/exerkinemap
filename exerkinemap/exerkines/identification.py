"""Exerkine identification scaffolding."""

from __future__ import annotations

from typing import Iterable, List


class ExerkineIdentifier:
    """Simple placeholder for identifying candidate exerkines."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def identify(self, scores: Iterable[float]) -> List[float]:
        return [score for score in scores if score >= self.threshold]


def identify_exerkines(scores: Iterable[float], threshold: float = 0.5) -> List[float]:
    return ExerkineIdentifier(threshold=threshold).identify(scores)
