"""Protein sequence tokenization helpers."""

from __future__ import annotations

from typing import List


class ProteinTokenizer:
    """Tokenize protein sequences into amino-acid characters."""

    def __init__(self, alphabet: List[str] | None = None):
        self.alphabet = tuple(alphabet) if alphabet is not None else None

    def tokenize(self, sequence: str) -> List[str]:
        if sequence is None:
            return []
        chars = list(str(sequence).strip())
        if self.alphabet is not None:
            return [ch for ch in chars if ch in self.alphabet]
        return chars


def tokenize_protein(sequence: str, alphabet: List[str] | None = None) -> List[str]:
    """Convenience function for protein tokenization."""
    return ProteinTokenizer(alphabet=alphabet).tokenize(sequence)
