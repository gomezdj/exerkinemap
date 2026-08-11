"""RNA sequence generation scaffolding."""

from __future__ import annotations


class RNAGenerator:
    """Simple placeholder for RNA sequence generation."""

    def generate(self, seed: str = "", length: int = 16) -> str:
        return seed or "A" * length


def generate_rna_sequence(seed: str = "", length: int = 16) -> str:
    return RNAGenerator().generate(seed=seed, length=length)
