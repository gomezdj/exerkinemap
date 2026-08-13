"""Adjacency matrix scaffolding."""

from __future__ import annotations

from typing import Dict, List


class AdjacencyMatrix:
    """Simple adjacency matrix wrapper."""

    def __init__(self, rows: int = 0, cols: int = 0):
        self.rows = rows
        self.cols = cols
        self.matrix: List[List[float]] = [[0.0 for _ in range(cols)] for _ in range(rows)]


def build_adjacency_matrix(rows: int = 0, cols: int = 0) -> AdjacencyMatrix:
    return AdjacencyMatrix(rows=rows, cols=cols)
