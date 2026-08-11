"""Laplacian matrix scaffolding."""

from __future__ import annotations

from typing import List


class LaplacianMatrix:
    """Simple Laplacian matrix wrapper."""

    def __init__(self, size: int = 0):
        self.size = size
        self.matrix: List[List[float]] = [[0.0 for _ in range(size)] for _ in range(size)]


def build_laplacian_matrix(size: int = 0) -> LaplacianMatrix:
    return LaplacianMatrix(size=size)
