"""Forward model scaffolding."""

from __future__ import annotations

from typing import Iterable, List


class ForwardModel:
    """Simple placeholder for a forward prediction model."""

    def __init__(self, name: str = "forward"):
        self.name = name

    def predict(self, values: Iterable[float]) -> List[float]:
        return [float(value) for value in values]


def build_forward_model(name: str = "forward") -> ForwardModel:
    return ForwardModel(name=name)
