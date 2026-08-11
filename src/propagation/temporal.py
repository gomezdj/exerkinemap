"""Temporal propagation scaffolding."""

from __future__ import annotations

from typing import Iterable, List


class TemporalPropagationModel:
    """Simple placeholder for temporal signal propagation."""

    def propagate(self, values: Iterable[float]) -> List[float]:
        return [float(value) for value in values]


def propagate_temporally(values: Iterable[float]) -> List[float]:
    return TemporalPropagationModel().propagate(values)
