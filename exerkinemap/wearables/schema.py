"""Canonical long-form schema for wearable and digital-biomarker observations."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


CANONICAL_WEARABLE_COLUMNS = (
    "subject_id",
    "timestamp",
    "signal",
    "value",
    "source",
    "unit",
)


class WearableSchemaError(ValueError):
    """Raised when a wearable frame cannot be represented in the common schema."""


def normalize_wearable_frame(
    frame: pd.DataFrame,
    *,
    column_map: Mapping[str, str],
    source: str,
    default_signal: str | None = None,
    default_unit: str | None = None,
) -> pd.DataFrame:
    """Map a source frame to the canonical subject-time-signal observation schema."""
    required = {"subject_id", "timestamp", "value"}
    missing_mapping = required - set(column_map)
    if missing_mapping:
        raise WearableSchemaError(
            f"column_map is missing required canonical fields: {sorted(missing_mapping)}"
        )
    unknown_source_columns = {
        source_column
        for source_column in column_map.values()
        if source_column not in frame.columns
    }
    if unknown_source_columns:
        raise WearableSchemaError(
            f"Input frame is missing mapped columns: {sorted(unknown_source_columns)}"
        )
    if "signal" not in column_map and not default_signal:
        raise WearableSchemaError(
            "Provide column_map['signal'] or a non-empty default_signal."
        )
    if not source.strip():
        raise WearableSchemaError("source must be non-empty.")

    normalized = pd.DataFrame(
        {
            "subject_id": frame[column_map["subject_id"]].astype(str).str.strip(),
            "timestamp": pd.to_datetime(
                frame[column_map["timestamp"]], utc=True, errors="coerce"
            ),
            "value": pd.to_numeric(frame[column_map["value"]], errors="coerce"),
        }
    )
    if "signal" in column_map:
        normalized["signal"] = frame[column_map["signal"]].astype(str).str.strip()
    else:
        normalized["signal"] = default_signal
    normalized["source"] = source
    if "unit" in column_map:
        normalized["unit"] = frame[column_map["unit"]].astype(str).str.strip()
    else:
        normalized["unit"] = default_unit or ""

    invalid_subjects = normalized["subject_id"].eq("").sum()
    invalid_signals = normalized["signal"].eq("").sum()
    invalid_timestamps = normalized["timestamp"].isna().sum()
    invalid_values = normalized["value"].isna().sum()
    if invalid_subjects or invalid_signals or invalid_timestamps or invalid_values:
        raise WearableSchemaError(
            "Invalid wearable observations: "
            f"{invalid_subjects} empty subject IDs, {invalid_signals} empty signals, "
            f"{invalid_timestamps} invalid timestamps, and {invalid_values} invalid values."
        )
    return validate_wearable_frame(normalized)


def validate_wearable_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and stably sort a canonical wearable observation frame."""
    missing_columns = set(CANONICAL_WEARABLE_COLUMNS) - set(frame.columns)
    if missing_columns:
        raise WearableSchemaError(
            f"Wearable frame is missing columns: {sorted(missing_columns)}"
        )
    if frame.empty:
        raise WearableSchemaError("Wearable frame contains no observations.")

    normalized = frame.loc[:, list(CANONICAL_WEARABLE_COLUMNS)].copy()
    normalized["timestamp"] = pd.to_datetime(
        normalized["timestamp"], utc=True, errors="coerce"
    )
    normalized["value"] = pd.to_numeric(normalized["value"], errors="coerce")
    for column in ("subject_id", "signal", "source", "unit"):
        normalized[column] = normalized[column].fillna("").astype(str).str.strip()
    if (
        normalized["subject_id"].eq("").any()
        or normalized["signal"].eq("").any()
        or normalized["source"].eq("").any()
        or normalized["timestamp"].isna().any()
        or normalized["value"].isna().any()
    ):
        raise WearableSchemaError(
            "Canonical wearable frame contains missing identifiers, timestamps, or values."
        )
    return normalized.sort_values(
        ["subject_id", "signal", "timestamp"], kind="stable"
    ).reset_index(drop=True)
