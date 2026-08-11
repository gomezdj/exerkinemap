"""Protein embedding scaffolding."""

from __future__ import annotations

from typing import Iterable

import numpy as np


class ProteinEmbeddingModel:
    """Simple embedding model for protein sequences."""

    def __init__(self, embedding_dim: int = 32):
        self.embedding_dim = embedding_dim

    def encode(self, sequences: Iterable[str]) -> np.ndarray:
        seq_list = list(sequences)
        if not seq_list:
            return np.zeros((0, self.embedding_dim), dtype=float)
        return np.random.randn(len(seq_list), self.embedding_dim).astype(float)


def embed_proteins(sequences: Iterable[str], embedding_dim: int = 32) -> np.ndarray:
    return ProteinEmbeddingModel(embedding_dim=embedding_dim).encode(sequences)
