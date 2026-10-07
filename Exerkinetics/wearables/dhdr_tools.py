"""DHDR-style wearable loaders that emit the common EXERKINEMAP frame."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema import normalize_wearable_frame


def load_dhdr_signal_csv(
    path: Path,
    *,
    source: str,
    subject_column: str,
    timestamp_column: str,
    value_column: str,
    signal: str,
    unit: str | None = None,
) -> pd.DataFrame:
    """Load a single-signal DHDR-style CSV into the canonical long-form frame."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"DHDR input CSV was not found: {path}")
    frame = pd.read_csv(path)
    return normalize_wearable_frame(
        frame,
        column_map={
            "subject_id": subject_column,
            "timestamp": timestamp_column,
            "value": value_column,
        },
        source=source,
        default_signal=signal,
        default_unit=unit,
    )


def load_dhdr_cgm_csv(
    path: Path,
    *,
    subject_column: str = "subject_id",
    timestamp_column: str = "timestamp",
    glucose_column: str = "glucose_mg_dl",
) -> pd.DataFrame:
    """Load continuous glucose-monitoring data for T2D, obesity, and GDM maps."""
    return load_dhdr_signal_csv(
        path,
        source="DHDR_CGM",
        subject_column=subject_column,
        timestamp_column=timestamp_column,
        value_column=glucose_column,
        signal="glucose_mg_dl",
        unit="mg/dL",
    )


def load_dhdr_spo2_csv(
    path: Path,
    *,
    subject_column: str = "subject_id",
    timestamp_column: str = "timestamp",
    spo2_column: str = "spo2_percent",
) -> pd.DataFrame:
    """Load blood-oxygen-saturation observations."""
    return load_dhdr_signal_csv(
        path,
        source="DHDR_SPO2",
        subject_column=subject_column,
        timestamp_column=timestamp_column,
        value_column=spo2_column,
        signal="spo2_percent",
        unit="percent",
    )


def load_dhdr_cardiovascular_csv(
    path: Path,
    *,
    subject_column: str = "subject_id",
    timestamp_column: str = "timestamp",
    heart_rate_column: str = "heart_rate_bpm",
) -> pd.DataFrame:
    """Load cardiovascular wearable observations as heart-rate signals."""
    return load_dhdr_signal_csv(
        path,
        source="DHDR_CARDIOVASCULAR",
        subject_column=subject_column,
        timestamp_column=timestamp_column,
        value_column=heart_rate_column,
        signal="heart_rate_bpm",
        unit="bpm",
    )


def load_dhdr_covid_csv(
    path: Path,
    *,
    subject_column: str = "subject_id",
    timestamp_column: str = "timestamp",
    illness_score_column: str = "illness_score",
) -> pd.DataFrame:
    """Load an illness-risk signal used to gate acute-illness windows."""
    return load_dhdr_signal_csv(
        path,
        source="DHDR_COVID",
        subject_column=subject_column,
        timestamp_column=timestamp_column,
        value_column=illness_score_column,
        signal="illness_score",
        unit="score",
    )


def load_dhdr_wearables_csv(
    path: Path,
    *,
    subject_column: str = "subject_id",
    timestamp_column: str = "timestamp",
    signal_column: str = "signal",
    value_column: str = "value",
    unit_column: str = "unit",
) -> pd.DataFrame:
    """Load a multi-signal generic DHDR wearable export."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"DHDR input CSV was not found: {path}")
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
        source="DHDR_WEARABLES",
    )
