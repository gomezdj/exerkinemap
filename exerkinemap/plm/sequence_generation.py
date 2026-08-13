"""Protein sequence generation scaffolding."""

from __future__ import annotations

from typing import List


class ProteinSequenceGenerator:
    """Generate simple protein-like token sequences from a prompt."""

    def __init__(self, token_vocab: List[str] | None = None):
        self.token_vocab = list(token_vocab or [])

    def generate(self, prompt: str, max_length: int = 16) -> List[str]:
        if not prompt:
            return []
        tokens = str(prompt).split()
        if len(tokens) >= max_length:
            return tokens[:max_length]
        return tokens + ["<PAD>"] * max(0, max_length - len(tokens))


def generate_protein_sequences(prompt: str, max_length: int = 16, token_vocab: List[str] | None = None) -> List[str]:
    return ProteinSequenceGenerator(token_vocab=token_vocab).generate(prompt, max_length=max_length)
