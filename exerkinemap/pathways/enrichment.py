"""Pathway enrichment scaffolding."""

from __future__ import annotations

from typing import Iterable, List


class PathwayEnrichmentAnalyzer:
    """Simple placeholder for pathway enrichment analysis."""

    def analyze(self, values: Iterable[float]) -> List[float]:
        return [float(value) for value in values]


def analyze_enrichment(values: Iterable[float]) -> List[float]:
    return PathwayEnrichmentAnalyzer().analyze(values)
