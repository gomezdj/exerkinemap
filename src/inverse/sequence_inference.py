"""Sequence inference scaffolding."""

from __future__ import annotations

from typing import Iterable, List


class SequenceInferenceModel:
    """Simple placeholder for sequence inference."""

    def infer(self, values: Iterable[str]) -> List[str]:
        return [str(value) for value in values]


def infer_sequences(values: Iterable[str]) -> List[str]:
    return SequenceInferenceModel().infer(values)
