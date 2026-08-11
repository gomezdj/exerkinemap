"""Pathway-to-receptor mapping scaffolding."""

from __future__ import annotations

from typing import Iterable, List


class PathwayToReceptorMapper:
    """Simple placeholder for pathway-to-receptor mapping."""

    def map(self, values: Iterable[float]) -> List[float]:
        return [float(value) for value in values]


def map_pathway_to_receptor(values: Iterable[float]) -> List[float]:
    return PathwayToReceptorMapper().map(values)
