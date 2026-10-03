"""A deterministic byte-pair tokenizer for biological sequences."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, List

from ._training import bounded_training_sequences


class BpeTokenizer:
    """Learn and apply character-level byte-pair merges."""

    def __init__(
        self,
        vocab: List[str] | None = None,
        merges: List[tuple[str, str]] | None = None,
        max_training_characters: int | None = None,
    ):
        self.vocab = list(vocab or [])
        self.merges = list(merges or [])
        self.max_training_characters = max_training_characters

    def fit(self, sequences: Iterable[str], vocab_size: int = 256) -> "BpeTokenizer":
        """Learn merges from training sequences only."""
        if vocab_size < 1:
            raise ValueError("vocab_size must be positive")

        words = [
            list(sequence)
            for sequence in bounded_training_sequences(
                sequences, self.max_training_characters
            )
        ]
        alphabet = sorted({token for word in words for token in word})
        self.vocab = list(alphabet)
        self.merges = []

        while len(self.vocab) < vocab_size:
            pair_counts: Counter[tuple[str, str]] = Counter(
                pair
                for word in words
                for pair in zip(word, word[1:])
            )
            if not pair_counts:
                break
            pair, count = min(pair_counts.items(), key=lambda item: (-item[1], item[0]))
            if count < 2:
                break
            merged = "".join(pair)
            self.merges.append(pair)
            self.vocab.append(merged)
            words = [
                self._merge_pair(word, pair, merged)
                for word in words
            ]
        return self

    def tokenize(self, sequence: str) -> List[str]:
        if sequence is None:
            return []
        tokens = list(str(sequence).strip())
        for pair in self.merges:
            tokens = self._merge_pair(tokens, pair, "".join(pair))
        return tokens

    @staticmethod
    def _merge_pair(tokens: List[str], pair: tuple[str, str], merged: str) -> List[str]:
        output: List[str] = []
        index = 0
        while index < len(tokens):
            if index + 1 < len(tokens) and (tokens[index], tokens[index + 1]) == pair:
                output.append(merged)
                index += 2
            else:
                output.append(tokens[index])
                index += 1
        return output


def tokenize_bpe(
    sequence: str,
    vocab: List[str] | None = None,
    merges: List[tuple[str, str]] | None = None,
) -> List[str]:
    """Convenience function for BPE-style tokenization."""
    return BpeTokenizer(vocab=vocab, merges=merges).tokenize(sequence)
