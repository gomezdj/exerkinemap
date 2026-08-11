"""Propagation and signal-flow primitives for EXERKINEMAP."""

from .diffusion import DiffusionModel, diffuse_signal
from .heat_kernel import HeatKernelModel, apply_heat_kernel
from .signal_state import SignalState, propagate_signal_state
from .temporal import TemporalPropagationModel, propagate_temporally

__all__ = [
    "DiffusionModel",
    "diffuse_signal",
    "HeatKernelModel",
    "apply_heat_kernel",
    "SignalState",
    "propagate_signal_state",
    "TemporalPropagationModel",
    "propagate_temporally",
]
