"""A simple amino-acid BERT scaffold."""

from __future__ import annotations

import numpy as np


class AminoBertModel:
    """Minimal amino-BERT-like model wrapper."""

    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.embeddings = np.random.randn(256, embedding_dim).astype(float)

    def encode(self, tokens: list[int] | np.ndarray) -> np.ndarray:
        arr = np.asarray(tokens, dtype=int)
        if arr.size == 0:
            return np.zeros(self.embedding_dim, dtype=float)
        return np.mean(self.embeddings[arr], axis=0)


def train_amino_bert_model(embedding_dim: int = 64) -> AminoBertModel:
    return AminoBertModel(embedding_dim=embedding_dim)
