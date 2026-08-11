"""Diffusion model scaffolding."""

from __future__ import annotations

from typing import Iterable, List


class DiffusionModel:
    """Simple placeholder for diffusion-based propagation."""

    def __init__(self, steps: int = 1):
        self.steps = steps

    def diffuse(self, values: Iterable[float]) -> List[float]:
        return [float(value) for value in values]


def diffuse_signal(values: Iterable[float], steps: int = 1) -> List[float]:
    return DiffusionModel(steps=steps).diffuse(values)
