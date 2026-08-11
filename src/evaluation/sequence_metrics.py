"""Sequence evaluation scaffolding."""

from __future__ import annotations

from typing import Iterable, List


class SequenceMetricEvaluator:
    """Simple placeholder for sequence-based evaluation metrics."""

    def evaluate(self, values: Iterable[float]) -> List[float]:
        return [float(value) for value in values]


def evaluate_sequence_metrics(values: Iterable[float]) -> List[float]:
    return SequenceMetricEvaluator().evaluate(values)
