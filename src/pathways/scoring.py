"""Pathway scoring scaffolding."""

from __future__ import annotations

from typing import Iterable, List


class PathwayScorer:
    """Simple placeholder for pathway scoring."""

    def score(self, values: Iterable[float]) -> List[float]:
        return [float(value) for value in values]


def score_pathways(values: Iterable[float]) -> List[float]:
    return PathwayScorer().score(values)
