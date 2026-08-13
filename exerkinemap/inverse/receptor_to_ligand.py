"""Receptor-to-ligand mapping scaffolding."""

from __future__ import annotations

from typing import Iterable, List


class ReceptorToLigandMapper:
    """Simple placeholder for receptor-to-ligand mapping."""

    def map(self, values: Iterable[float]) -> List[float]:
        return [float(value) for value in values]


def map_receptor_to_ligand(values: Iterable[float]) -> List[float]:
    return ReceptorToLigandMapper().map(values)
