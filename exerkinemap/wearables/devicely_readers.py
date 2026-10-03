"""Readers for devicely-compatible device exports without a vendor dependency."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema import normalize_wearable_frame


def read_devicely_csv(
    path: Path,
    *,
    subject_column: str = "subject_id",
    timestamp_column: str = "timestamp",
    signal_column: str = "signal",
    value_column: str = "value",
    unit_column: str = "unit",
) -> pd.DataFrame:
    """Read a normalized devicely CSV export into the shared wearable schema."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"devicely input CSV was not found: {path}")
    frame = pd.read_csv(path)
    return normalize_wearable_frame(
        frame,
        column_map={
            "subject_id": subject_column,
            "timestamp": timestamp_column,
            "signal": signal_column,
            "value": value_column,
            "unit": unit_column,
        },
        source="devicely",
    )
