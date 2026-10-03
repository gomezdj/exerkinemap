"""Backward-compatible imports for moved wearable modules."""

from .dhdr_tools import load_dhdr_cgm_csv
from .flirt_features import causal_window_features
from .gate import ExtrinsicGate

__all__ = ["ExtrinsicGate", "causal_window_features", "load_dhdr_cgm_csv"]
