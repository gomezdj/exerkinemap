"""Spatial coordinate helpers."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


class SpatialCoordinates:
    """Simple container for spatial coordinates."""

    def __init__(self, coordinates: Sequence[Tuple[float, float]] | None = None):
        self.coordinates = list(coordinates or [])

    def append(self, point: Tuple[float, float]) -> None:
        self.coordinates.append(tuple(point))

    def __len__(self) -> int:
        return len(self.coordinates)

    def __iter__(self):
        return iter(self.coordinates)


def normalize_coordinates(coordinates: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    return [tuple(point) for point in coordinates]
