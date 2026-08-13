"""Forward validation scaffolding."""

from __future__ import annotations

from typing import Iterable, List


class ForwardValidator:
    """Simple placeholder for validating forward results."""

    def validate(self, values: Iterable[float]) -> List[bool]:
        return [value is not None for value in values]


def validate_forward_results(values: Iterable[float]) -> List[bool]:
    return ForwardValidator().validate(values)
