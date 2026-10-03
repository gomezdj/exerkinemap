"""A deterministic unigram tokenizer for biological sequences."""

from __future__ import annotations

from collections import Counter
from math import log
from typing import Iterable, List

from ._training import bounded_training_sequences


class UnigramTokenizer:
    """Learn substring scores and select the highest-scoring segmentation."""

    def __init__(
        self,
        vocab: List[str] | None = None,
        max_training_characters: int | None = None,
    ):
        self.vocab = list(vocab or [])
        self.scores: dict[str, float] = {token: 0.0 for token in self.vocab}
        self.max_training_characters = max_training_characters
        self.max_token_length = max((len(token) for token in self.vocab), default=1)

    def fit(
        self,
        sequences: Iterable[str],
        vocab_size: int = 256,
        max_token_length: int = 12,
    ) -> "UnigramTokenizer":
        """Estimate a unigram vocabulary and log-frequency token scores."""
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
                for end in range(start + 1, min(len(sequence), start + max_token_length) + 1):
                    counts[sequence[start:end]] += 1

        candidates = sorted(counts, key=lambda token: (-counts[token], -len(token), token))
        self.vocab = sorted(alphabet) + [
            token for token in candidates if token not in alphabet
        ][: max(0, vocab_size - len(alphabet))]
        total = sum(counts[token] for token in self.vocab)
        self.scores = {
            token: log((counts[token] + 1) / (total + len(self.vocab)))
            for token in self.vocab
        }
        self.max_token_length = max(map(len, self.vocab), default=1)
        return self

    def tokenize(self, sequence: str) -> List[str]:
        if sequence is None:
            return []
        seq = str(sequence).strip()
        if not seq:
            return []
        if not self.scores:
            return list(seq)

        best_scores = [float("-inf")] * (len(seq) + 1)
        segments: List[List[str] | None] = [None] * (len(seq) + 1)
        best_scores[0] = 0.0
        for end in range(1, len(seq) + 1):
            for start in range(max(0, end - self.max_token_length), end):
                token = seq[start:end]
                if token not in self.scores or best_scores[start] == float("-inf"):
                    continue
                score = best_scores[start] + self.scores[token]
                if score > best_scores[end]:
                    best_scores[end] = score
                    segments[end] = (segments[start] or []) + [token]
        return segments[-1] or list(seq)


def tokenize_unigram(sequence: str, vocab: List[str] | None = None) -> List[str]:
    """Convenience function for unigram-style tokenization."""
    return UnigramTokenizer(vocab=vocab).tokenize(sequence)
