"""Molecular embedding scaffolding."""

from __future__ import annotations

from typing import Iterable

import numpy as np


class MolecularEmbeddingModel:
    """Simple embedding model for molecular descriptors or smiles."""

    def __init__(self, embedding_dim: int = 32):
        self.embedding_dim = embedding_dim

    def encode(self, molecules: Iterable[str]) -> np.ndarray:
        mol_list = list(molecules)
        if not mol_list:
            return np.zeros((0, self.embedding_dim), dtype=float)
        return np.random.randn(len(mol_list), self.embedding_dim).astype(float)


def embed_molecules(molecules: Iterable[str], embedding_dim: int = 32) -> np.ndarray:
    return MolecularEmbeddingModel(embedding_dim=embedding_dim).encode(molecules)
