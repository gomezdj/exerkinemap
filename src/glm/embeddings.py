"""Embedding layers used by the GLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TokenEmbedding:
    """Simple token embedding layer wrapper."""

    vocab_size: int
    embedding_dim: int = 64
    weights: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.weights is None:
            self.weights = np.random.randn(self.vocab_size, self.embedding_dim).astype(float)


@dataclass
class PositionalEmbedding:
    """Simple positional embedding layer wrapper."""

    max_length: int
    embedding_dim: int = 64
    weights: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.weights is None:
            self.weights = np.random.randn(self.max_length, self.embedding_dim).astype(float)


def build_embedding_layers(vocab_size: int, embedding_dim: int = 64, max_length: int = 512) -> tuple[TokenEmbedding, PositionalEmbedding]:
    """Create token and positional embedding layers."""
    return TokenEmbedding(vocab_size=vocab_size, embedding_dim=embedding_dim), PositionalEmbedding(max_length=max_length, embedding_dim=embedding_dim)
