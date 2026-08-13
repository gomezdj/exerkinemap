"""Differential response analysis scaffolding."""

from __future__ import annotations

from typing import Iterable, List, Tuple


class DifferentialResponseAnalyzer:
    """Simple placeholder for differential-response analysis."""

    def analyze(self, values_a: Iterable[float], values_b: Iterable[float]) -> List[Tuple[float, float]]:
        a = list(values_a)
        b = list(values_b)
        return list(zip(a, b))


def differential_response(values_a: Iterable[float], values_b: Iterable[float]) -> List[Tuple[float, float]]:
    return DifferentialResponseAnalyzer().analyze(values_a, values_b)
