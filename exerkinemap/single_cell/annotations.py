"""Annotation management scaffolding."""

from __future__ import annotations

from typing import Dict, Iterable, List


class AnnotationManager:
    """Simple placeholder for storing annotations."""

    def __init__(self):
        self.annotations: Dict[str, List[str]] = {}

    def add(self, key: str, values: Iterable[str]) -> None:
        self.annotations[key] = [str(value) for value in values]

    def get(self, key: str) -> List[str]:
        return list(self.annotations.get(key, []))


def manage_annotations() -> AnnotationManager:
    return AnnotationManager()
