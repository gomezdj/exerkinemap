"""Character-based tokenization for genomic and protein sequences."""

from __future__ import annotations

from typing import Iterable, List


class CharacterTokenizer:
    """Simple character tokenizer that splits sequences into single characters."""

    def __init__(self, alphabet: Iterable[str] | None = None):
        self.alphabet = tuple(alphabet) if alphabet is not None else None

    def tokenize(self, sequence: str) -> List[str]:
        if sequence is None:
            return []
        chars = list(str(sequence))
        if self.alphabet is not None:
            return [ch for ch in chars if ch in self.alphabet]
        return chars


def tokenize_characters(sequence: str, alphabet: Iterable[str] | None = None) -> List[str]:
    """Convenience function for character tokenization."""
    return CharacterTokenizer(alphabet=alphabet).tokenize(sequence)
