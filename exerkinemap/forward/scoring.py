"""Forward scoring scaffolding."""

from __future__ import annotations

from typing import Iterable, List


class ForwardScorer:
    """Simple placeholder for scoring forward predictions."""

    def score(self, values: Iterable[float]) -> List[float]:
        return [float(value) for value in values]


def score_forward_results(values: Iterable[float]) -> List[float]:
    return ForwardScorer().score(values)
