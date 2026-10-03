"""k-mer tokenization helpers."""

from __future__ import annotations

from typing import List


class KMerTokenizer:
    """Split a sequence into k-mers with a configurable stride."""

    def __init__(self, k: int = 3, stride: int = 1):
        if k <= 0:
            raise ValueError("k must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")
        self.k = k
        self.stride = stride

    def tokenize(self, sequence: str) -> List[str]:
        if sequence is None:
            return []
        seq = str(sequence).strip()
        if len(seq) < self.k:
            return []
        return [
            seq[i : i + self.k]
            for i in range(0, len(seq) - self.k + 1, self.stride)
        ]


def tokenize_kmers(sequence: str, k: int = 3, stride: int = 1) -> List[str]:
    """Convenience function for k-mer tokenization."""
    return KMerTokenizer(k=k, stride=stride).tokenize(sequence)
