"""Causal wearable-window features compatible with FLIRT-style telemetry inputs."""

from __future__ import annotations

import pandas as pd

from .schema import validate_wearable_frame


def causal_window_features(
    frame: pd.DataFrame,
    *,
    window: str | pd.Timedelta = "30min",
    min_periods: int = 1,
) -> pd.DataFrame:
    """Compute windows of observations in ``(end - window, end]`` only."""
    if min_periods < 1:
        raise ValueError("min_periods must be positive.")
    window_delta = pd.to_timedelta(window)
    if window_delta <= pd.Timedelta(0):
        raise ValueError("window must be positive.")

    normalized = validate_wearable_frame(frame)
    rows: list[dict[str, object]] = []
    for (subject_id, signal), group in normalized.groupby(
        ["subject_id", "signal"], sort=False
    ):
        group = group.sort_values("timestamp", kind="stable")
        for window_end in group["timestamp"].drop_duplicates():
            window_start = window_end - window_delta
            observations = group.loc[
                (group["timestamp"] > window_start)
                & (group["timestamp"] <= window_end),
                "value",
            ]
            if len(observations) < min_periods:
                continue
            rows.append(
                {
                    "subject_id": subject_id,
                    "signal": signal,
                    "source": group["source"].iloc[0],
                    "unit": group["unit"].iloc[0],
                    "window_start": window_start,
                    "window_end": window_end,
                    "count": len(observations),
                    "mean": observations.mean(),
                    "std": observations.std(ddof=0),
                    "min": observations.min(),
                    "max": observations.max(),
                    "last": observations.iloc[-1],
                    "delta": observations.iloc[-1] - observations.iloc[0],
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "subject_id",
            "signal",
            "source",
            "unit",
            "window_start",
            "window_end",
            "count",
            "mean",
            "std",
            "min",
            "max",
            "last",
            "delta",
        ],
    )
