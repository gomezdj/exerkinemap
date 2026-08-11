"""Spatial evaluation scaffolding."""

from __future__ import annotations

from typing import Iterable, List


class SpatialMetricEvaluator:
    """Simple placeholder for spatial evaluation metrics."""

    def evaluate(self, values: Iterable[float]) -> List[float]:
        return [float(value) for value in values]


def evaluate_spatial_metrics(values: Iterable[float]) -> List[float]:
    return SpatialMetricEvaluator().evaluate(values)
