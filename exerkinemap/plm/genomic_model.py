"""Protein genomic-context wrapper."""

from __future__ import annotations

from typing import Iterable

import numpy as np


class ProteinGenomicModel:
    """Minimal wrapper for combining protein and genomic context representations."""

    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim

    def encode(self, sequences: Iterable[str]) -> np.ndarray:
        seq_list = list(sequences)
        if not seq_list:
            return np.zeros((0, self.embedding_dim), dtype=float)
        return np.random.randn(len(seq_list), self.embedding_dim).astype(float)


def encode_genomic_context(sequences: Iterable[str], embedding_dim: int = 64) -> np.ndarray:
    return ProteinGenomicModel(embedding_dim=embedding_dim).encode(sequences)
