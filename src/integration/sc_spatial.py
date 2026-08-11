"""Single-cell to spatial integration scaffolding."""

from __future__ import annotations

from typing import Iterable, List


class SingleCellSpatialIntegrator:
    """Simple placeholder for integrating single-cell and spatial data."""

    def integrate(self, values: Iterable[float]) -> List[float]:
        return [float(value) for value in values]


def integrate_sc_spatial(values: Iterable[float]) -> List[float]:
    return SingleCellSpatialIntegrator().integrate(values)
