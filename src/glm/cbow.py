"""Simple CBOW-style embedding scaffolding for the GLM."""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np


class CBOWModel:
    """A minimal CBOW-style model that stores token embeddings."""

    def __init__(self, vocab_size: int, embedding_dim: int = 64):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embeddings = np.random.randn(vocab_size, embedding_dim).astype(float)

    def fit(self, tokenized_sequences: Iterable[Iterable[int]], epochs: int = 1) -> "CBOWModel":
        """Placeholder fit method for future integration with gensim/torch."""
        return self

    def encode(self, tokens: Iterable[int]) -> np.ndarray:
        token_ids = list(tokens)
        if not token_ids:
            return np.zeros(self.embedding_dim, dtype=float)
        return np.mean(self.embeddings[token_ids], axis=0)


def train_cbow_model(vocab_size: int, embedding_dim: int = 64) -> CBOWModel:
    """Convenience wrapper for constructing a CBOW model."""
    return CBOWModel(vocab_size=vocab_size, embedding_dim=embedding_dim)
