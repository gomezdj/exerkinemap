"""k-mer tokenization helpers."""

from __future__ import annotations

from typing import List


class KMerTokenizer:
    """Split a sequence into overlapping k-mers."""

    def __init__(self, k: int = 3):
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k

    def tokenize(self, sequence: str) -> List[str]:
        if sequence is None:
            return []
        seq = str(sequence).strip()
        if len(seq) < self.k:
            return []
        return [seq[i : i + self.k] for i in range(len(seq) - self.k + 1)]


def tokenize_kmers(sequence: str, k: int = 3) -> List[str]:
    """Convenience function for k-mer tokenization."""
    return KMerTokenizer(k=k).tokenize(sequence)
