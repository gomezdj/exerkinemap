"""Signal state scaffolding for propagation workflows."""

from __future__ import annotations

from typing import Dict, List


class SignalState:
    """Simple container for signal values across nodes or states."""

    def __init__(self):
        self.values: Dict[str, float] = {}

    def add(self, name: str, value: float) -> None:
        self.values[name] = float(value)

    def get(self, name: str) -> float:
        return self.values.get(name, 0.0)


def propagate_signal_state(state: SignalState) -> Dict[str, float]:
    return dict(state.values)
