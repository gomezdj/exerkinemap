"""Differential expression scaffolding."""

from __future__ import annotations

from typing import Iterable, List, Tuple


class DifferentialExpressionAnalyzer:
    """Simple placeholder for differential-expression analysis."""

    def analyze(self, group_a: Iterable[float], group_b: Iterable[float]) -> List[Tuple[float, float]]:
        return list(zip(group_a, group_b))


def analyze_differential_expression(group_a: Iterable[float], group_b: Iterable[float]) -> List[Tuple[float, float]]:
    return DifferentialExpressionAnalyzer().analyze(group_a, group_b)
