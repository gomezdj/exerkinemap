"""Graph scaffolding for network analysis."""

from __future__ import annotations

from typing import Dict, List, Tuple


class NetworkGraph:
    """A lightweight graph container."""

    def __init__(self):
        self.adjacency: Dict[str, List[str]] = {}

    def add_edge(self, source: str, target: str) -> None:
        self.adjacency.setdefault(source, []).append(target)

    def nodes(self) -> List[str]:
        return sorted(set(self.adjacency))


def build_graph() -> NetworkGraph:
    return NetworkGraph()
