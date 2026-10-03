"""Adapters for externally trained DeepPostures and HAR classifiers."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd


def attach_predicted_labels(
    features: pd.DataFrame,
    predictor: Callable[[pd.DataFrame], Sequence[str]],
    *,
    label_column: str,
) -> pd.DataFrame:
    """Attach labels from a caller-supplied posture or activity classifier."""
    labels = list(predictor(features))
    if len(labels) != len(features):
        raise ValueError(
            f"Predictor returned {len(labels)} labels for {len(features)} feature rows."
        )
    if any(not str(label).strip() for label in labels):
        raise ValueError("Predictor returned an empty posture or activity label.")
    labeled = features.copy()
    labeled[label_column] = [str(label).strip() for label in labels]
    return labeled


def attach_deep_postures(
    features: pd.DataFrame, predictor: Callable[[pd.DataFrame], Sequence[str]]
) -> pd.DataFrame:
    """Attach posture labels from an externally configured DeepPostures model."""
    return attach_predicted_labels(features, predictor, label_column="posture")


def attach_har_labels(
    features: pd.DataFrame, predictor: Callable[[pd.DataFrame], Sequence[str]]
) -> pd.DataFrame:
    """Attach activity labels from an externally configured HAR model."""
    return attach_predicted_labels(features, predictor, label_column="activity")
