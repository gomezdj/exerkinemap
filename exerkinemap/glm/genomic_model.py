"""High-level genomic model wrapper for the GLM."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .cbow import CBOWModel
from .embeddings import build_embedding_layers
from .encoder import TransformerEncoder


class GenomicLanguageModel:
    """A lightweight wrapper combining CBOW embeddings and an encoder."""

    def __init__(self, vocab_size: int, embedding_dim: int = 64, max_length: int = 512):
        self.cbow = CBOWModel(vocab_size=vocab_size, embedding_dim=embedding_dim)
        self.token_embedding, self.positional_embedding = build_embedding_layers(vocab_size, embedding_dim=embedding_dim, max_length=max_length)
        self.encoder = TransformerEncoder(embedding_dim=embedding_dim)

    def encode(self, token_ids: Iterable[int]) -> np.ndarray:
        token_ids = list(token_ids)
        embeddings = self.cbow.embeddings[token_ids]
        return self.encoder.encode(embeddings)


def train_genomic_model(vocab_size: int, embedding_dim: int = 64) -> GenomicLanguageModel:
    """Convenience function to build a genomic language model."""
    return GenomicLanguageModel(vocab_size=vocab_size, embedding_dim=embedding_dim)
