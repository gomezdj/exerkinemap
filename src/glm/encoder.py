"""Transformer-style encoder scaffolding for the GLM."""

from __future__ import annotations

import numpy as np

from .attention import AttentionLayer


class TransformerEncoder:
    """A lightweight transformer encoder wrapper."""

    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.attention = AttentionLayer(embedding_dim)

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        return self.attention(embeddings)


class EncoderLayer(TransformerEncoder):
    """Alias for the encoder wrapper."""

    pass
