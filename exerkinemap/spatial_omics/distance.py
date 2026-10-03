"""Distance helpers for spatial data."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import math


def pairwise_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def compute_distance_matrix(points: Sequence[Tuple[float, float]]) -> List[List[float]]:
    return [
        [pairwise_distance(point_a, point_b) for point_b in points]
        for point_a in points
    ]
