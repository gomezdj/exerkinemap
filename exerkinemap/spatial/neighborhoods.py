"""Neighborhood construction helpers."""

from __future__ import annotations

from typing import List, Sequence, Tuple


class NeighborhoodBuilder:
    """Simple placeholder for building spatial neighborhoods."""

    def __init__(self, radius: float = 1.0):
        self.radius = radius

    def build(self, points: Sequence[Tuple[float, float]]) -> List[List[int]]:
        return [[idx for idx, _ in enumerate(points) if idx != i] for i in range(len(points))]


def build_neighborhoods(points: Sequence[Tuple[float, float]], radius: float = 1.0) -> List[List[int]]:
    return NeighborhoodBuilder(radius=radius).build(points)
