"""Cell-type annotation scaffolding."""

from __future__ import annotations

from typing import Iterable, List


class CellTypeAnnotator:
    """Simple placeholder for assigning cell-type labels."""

    def annotate(self, labels: Iterable[str]) -> List[str]:
        return [str(label) for label in labels]


def annotate_cell_types(labels: Iterable[str]) -> List[str]:
    return CellTypeAnnotator().annotate(labels)
