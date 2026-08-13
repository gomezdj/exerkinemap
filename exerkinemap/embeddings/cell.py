"""Cell embedding scaffolding."""

from __future__ import annotations

from typing import Iterable

import numpy as np


class CellEmbeddingModel:
    """Simple embedding model for single-cell or cell-style features."""

    def __init__(self, embedding_dim: int = 32):
        self.embedding_dim = embedding_dim

    def encode(self, cells: Iterable[str]) -> np.ndarray:
        cell_list = list(cells)
        if not cell_list:
            return np.zeros((0, self.embedding_dim), dtype=float)
        return np.random.randn(len(cell_list), self.embedding_dim).astype(float)


def embed_cells(cells: Iterable[str], embedding_dim: int = 32) -> np.ndarray:
    return CellEmbeddingModel(embedding_dim=embedding_dim).encode(cells)
