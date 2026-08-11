"""Kernel functions for spatial analysis."""

from __future__ import annotations

from typing import Sequence, Tuple

import math


class GaussianKernel:
    """Simple Gaussian kernel for spatial weighting."""

    def __init__(self, bandwidth: float = 1.0):
        self.bandwidth = bandwidth

    def __call__(self, distance: float) -> float:
        if self.bandwidth <= 0:
            return 1.0 if distance == 0 else 0.0
        return math.exp(-(distance**2) / (2 * self.bandwidth**2))


def compute_kernel_matrix(points: Sequence[Tuple[float, float]], bandwidth: float = 1.0) -> list[list[float]]:
    kernel = GaussianKernel(bandwidth=bandwidth)
    return [
        [kernel(pairwise_distance(point_a, point_b)) for point_b in points]
        for point_a in points
    ]


def pairwise_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])
