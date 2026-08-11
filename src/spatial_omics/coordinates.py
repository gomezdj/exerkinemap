"""Spatial coordinate scaffolding for omics workflows."""

from __future__ import annotations

from typing import Iterable, List, Tuple


class SpatialCoordinates:
    """Simple container for spatial coordinates."""

    def __init__(self, coordinates: Iterable[Tuple[float, float]] | None = None):
        self.coordinates = [tuple(point) for point in (coordinates or [])]

    def add(self, point: Tuple[float, float]) -> None:
        self.coordinates.append(tuple(point))


def normalize_coordinates(coordinates: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    return [tuple(point) for point in coordinates]
