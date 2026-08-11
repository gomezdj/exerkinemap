"""A lightweight BPE-style tokenizer scaffold."""

from __future__ import annotations

from typing import List


class BpeTokenizer:
    """A minimal BPE-like tokenizer that splits on whitespace and characters."""

    def __init__(self, vocab: List[str] | None = None):
        self.vocab = list(vocab or [])

    def tokenize(self, sequence: str) -> List[str]:
        if sequence is None:
            return []
        seq = str(sequence).strip()
        if not seq:
            return []
        return [token for token in seq.split() if token]


def tokenize_bpe(sequence: str, vocab: List[str] | None = None) -> List[str]:
    """Convenience function for BPE-style tokenization."""
    return BpeTokenizer(vocab=vocab).tokenize(sequence)
