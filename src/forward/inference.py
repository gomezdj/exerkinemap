"""Forward inference scaffolding."""

from __future__ import annotations

from typing import Iterable, List


class ForwardInference:
    """Simple placeholder for running forward inference."""

    def run(self, values: Iterable[float]) -> List[float]:
        return [float(value) for value in values]


def run_inference(values: Iterable[float]) -> List[float]:
    return ForwardInference().run(values)
