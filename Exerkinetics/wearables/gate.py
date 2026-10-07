"""Bounded exogenous-modulation gate for wearable-derived context."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExtrinsicGate:
    """Map wearable features to a bounded modulation weight."""

    weights: np.ndarray | None = None
    bias: float = 0.0
    minimum_weight: float = 0.5
    maximum_weight: float = 1.5

    def __post_init__(self) -> None:
        if self.minimum_weight >= self.maximum_weight:
            raise ValueError("minimum_weight must be lower than maximum_weight.")
        if self.weights is not None and self.weights.ndim != 1:
            raise ValueError("weights must be a one-dimensional array.")

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Return bounded weights; zero logits yield the neutral weight of one."""
        values = np.asarray(features, dtype=float)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2:
            raise ValueError("features must be a one- or two-dimensional array.")
        if self.weights is None:
            logits = np.full(values.shape[0], self.bias, dtype=float)
        else:
            if values.shape[1] != len(self.weights):
                raise ValueError(
                    f"features have {values.shape[1]} columns but gate expects "
                    f"{len(self.weights)}."
                )
            logits = values @ self.weights + self.bias
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        return self.minimum_weight + (
            self.maximum_weight - self.minimum_weight
        ) * probabilities


def apply_illness_mask(
    weights: np.ndarray, acute_illness: np.ndarray | list[bool]
) -> np.ndarray:
    """Set modulation to neutral for windows flagged as acute illness."""
    values = np.asarray(weights, dtype=float).copy()
    mask = np.asarray(acute_illness, dtype=bool)
    if values.shape != mask.shape:
        raise ValueError("weights and acute_illness must have matching shapes.")
    values[mask] = 1.0
    return values


def modulate_scores(scores: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Apply wearable-derived modulation to static communication scores."""
    return np.asarray(scores, dtype=float) * np.asarray(weights, dtype=float)
