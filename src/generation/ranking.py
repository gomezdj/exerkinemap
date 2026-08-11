"""Candidate ranking scaffolding."""

from __future__ import annotations

from typing import Iterable, List, Tuple


class CandidateRanker:
    """Simple placeholder for ranking generated candidates."""

    def rank(self, candidates: Iterable[str]) -> List[Tuple[str, float]]:
        return [(candidate, float(index)) for index, candidate in enumerate(candidates)]


def rank_candidates(candidates: Iterable[str]) -> List[Tuple[str, float]]:
    return CandidateRanker().rank(candidates)
