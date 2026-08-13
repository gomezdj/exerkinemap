"""Prior construction scaffolding for exerkine analysis."""

from __future__ import annotations

from typing import Iterable, List


class PriorBuilder:
    """Simple placeholder for prior construction."""

    def build(self, values: Iterable[float]) -> List[float]:
        return [float(value) for value in values]


def build_priors(values: Iterable[float]) -> List[float]:
    return PriorBuilder().build(values)
