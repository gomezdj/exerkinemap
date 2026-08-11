"""Compatibility checks for exerkine candidates."""

from __future__ import annotations

from typing import Iterable, List


class CompatibilityChecker:
    """Simple placeholder for compatibility checks."""

    def check(self, values: Iterable[float], threshold: float = 0.0) -> List[bool]:
        return [value >= threshold for value in values]


def check_compatibility(values: Iterable[float], threshold: float = 0.0) -> List[bool]:
    return CompatibilityChecker().check(values, threshold=threshold)
