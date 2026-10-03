"""A deterministic WordPiece-style tokenizer for biological sequences."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, List

from ._training import bounded_training_sequences


class WordPieceTokenizer:
    """Learn a substring vocabulary and apply longest-match tokenization."""

    def __init__(
        self,
        vocab: List[str] | None = None,
        max_training_characters: int | None = None,
    ):
        self.vocab = list(vocab or [])
        self._vocab_set = set(self.vocab)
        self.max_training_characters = max_training_characters
        self.max_token_length = max((len(token) for token in self.vocab), default=1)

    def fit(
        self,
        sequences: Iterable[str],
        vocab_size: int = 256,
        max_token_length: int = 12,
    ) -> "WordPieceTokenizer":
        """Learn frequent sequence substrings from training sequences."""
        if vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if max_token_length < 1:
            raise ValueError("max_token_length must be positive")

        sequences = bounded_training_sequences(
            sequences, self.max_training_characters
        )
        alphabet = {character for sequence in sequences for character in sequence}
        counts: Counter[str] = Counter()
        for sequence in sequences:
            for start in range(len(sequence)):
                for end in range(start + 2, min(len(sequence), start + max_token_length) + 1):
                    counts[sequence[start:end]] += 1

        candidates = sorted(counts, key=lambda token: (-counts[token], -len(token), token))
        self.vocab = sorted(alphabet) + candidates[: max(0, vocab_size - len(alphabet))]
        self._vocab_set = set(self.vocab)
        self.max_token_length = max(map(len, self.vocab), default=1)
        return self

    def tokenize(self, sequence: str) -> List[str]:
        if sequence is None:
            return []
        seq = str(sequence).strip()
        if not seq:
            return []
        if not self._vocab_set:
            return list(seq)
        tokens: List[str] = []
        index = 0
        while index < len(seq):
            for end in range(
                min(len(seq), index + self.max_token_length), index, -1
            ):
                candidate = seq[index:end]
                if candidate in self._vocab_set:
                    tokens.append(candidate)
                    index = end
                    break
            else:
                tokens.append(seq[index])
                index += 1
        return tokens


def tokenize_wordpiece(sequence: str, vocab: List[str] | None = None) -> List[str]:
    """Convenience function for WordPiece-style tokenization."""
    return WordPieceTokenizer(vocab=vocab).tokenize(sequence)
