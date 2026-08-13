"""Attention primitives for the GLM encoder."""

from __future__ import annotations

import numpy as np


class SelfAttention:
    """A minimal self-attention implementation using dense numpy operations."""

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim

    def __call__(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        weights = np.ones((embeddings.shape[0], embeddings.shape[0]), dtype=float) / max(embeddings.shape[0], 1)
        return weights @ embeddings


class AttentionLayer(SelfAttention):
    """Alias for the self-attention primitive."""

    pass
