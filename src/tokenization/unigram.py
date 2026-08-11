"""A lightweight unigram tokenizer scaffold."""

from __future__ import annotations

from typing import List


class UnigramTokenizer:
    """A minimal unigram tokenizer that splits on whitespace."""

    def __init__(self, vocab: List[str] | None = None):
        self.vocab = list(vocab or [])

    def tokenize(self, sequence: str) -> List[str]:
        if sequence is None:
            return []
        seq = str(sequence).strip()
        if not seq:
            return []
        return [token for token in seq.split() if token]


def tokenize_unigram(sequence: str, vocab: List[str] | None = None) -> List[str]:
    """Convenience function for unigram-style tokenization."""
    return UnigramTokenizer(vocab=vocab).tokenize(sequence)
