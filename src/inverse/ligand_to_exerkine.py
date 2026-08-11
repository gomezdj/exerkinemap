"""Ligand-to-exerkine mapping scaffolding."""

from __future__ import annotations

from typing import Iterable, List


class LigandToExerkineMapper:
    """Simple placeholder for ligand-to-exerkine mapping."""

    def map(self, values: Iterable[float]) -> List[float]:
        return [float(value) for value in values]


def map_ligand_to_exerkine(values: Iterable[float]) -> List[float]:
    return LigandToExerkineMapper().map(values)
