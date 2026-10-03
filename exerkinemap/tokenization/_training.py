"""Shared deterministic corpus limits for learned sequence tokenizers."""

from __future__ import annotations

from typing import Iterable


def bounded_training_sequences(
    sequences: Iterable[str], max_training_characters: int | None
) -> list[str]:
    """Bound tokenizer-training text while preserving every input sequence."""
    normalized = [str(sequence).strip() for sequence in sequences if str(sequence).strip()]
    if max_training_characters is None:
        return normalized
    if max_training_characters < len(normalized):
        raise ValueError(
            "max_training_characters must allow at least one character per sequence"
        )
    if sum(map(len, normalized)) <= max_training_characters:
        return normalized

    lengths = [min(len(sequence), max_training_characters // len(normalized)) for sequence in normalized]
    remaining = max_training_characters - sum(lengths)
    while remaining:
        progressed = False
        for index, sequence in enumerate(normalized):
            if lengths[index] < len(sequence):
                lengths[index] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            break
    return [sequence[:length] for sequence, length in zip(normalized, lengths)]
