"""Xenium spatial omics scaffolding."""

from __future__ import annotations

from typing import Iterable, List


class XeniumProcessor:
    """Simple placeholder for Xenium data processing."""

    def process(self, values: Iterable[str]) -> List[str]:
        return [str(value) for value in values]


def process_xenium_data(values: Iterable[str]) -> List[str]:
    return XeniumProcessor().process(values)
