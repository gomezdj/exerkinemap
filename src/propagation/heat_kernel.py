"""Heat-kernel propagation scaffolding."""

from __future__ import annotations

from typing import Iterable, List


class HeatKernelModel:
    """Simple placeholder for heat-kernel propagation."""

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def apply(self, values: Iterable[float]) -> List[float]:
        return [float(value) * self.scale for value in values]


def apply_heat_kernel(values: Iterable[float], scale: float = 1.0) -> List[float]:
    return HeatKernelModel(scale=scale).apply(values)
