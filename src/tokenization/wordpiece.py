"""A lightweight WordPiece-style tokenizer scaffold."""

from __future__ import annotations

from typing import List


class WordPieceTokenizer:
    """A minimal WordPiece-style tokenizer that splits on whitespace."""

    def __init__(self, vocab: List[str] | None = None):
        self.vocab = list(vocab or [])

    def tokenize(self, sequence: str) -> List[str]:
        if sequence is None:
            return []
        seq = str(sequence).strip()
        if not seq:
            return []
        return [token for token in seq.split() if token]


def tokenize_wordpiece(sequence: str, vocab: List[str] | None = None) -> List[str]:
    """Convenience function for WordPiece-style tokenization."""
    return WordPieceTokenizer(vocab=vocab).tokenize(sequence)
