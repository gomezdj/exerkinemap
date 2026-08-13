"""Validation scaffolding for inverse modeling."""

from __future__ import annotations

from typing import Iterable, List


class ValidationRoutine:
    """Simple placeholder for validation routines."""

    def validate(self, values: Iterable[float]) -> List[bool]:
        return [value is not None for value in values]


def validate_inverse_model(values: Iterable[float]) -> List[bool]:
    return ValidationRoutine().validate(values)
