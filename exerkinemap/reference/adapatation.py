"""Helpers for adapting tabular reference data into EXERKINEMAP-friendly structures."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd


def prepare_reference_frame(data: Any, *, columns: Optional[list[str]] = None) -> pd.DataFrame:
    """Convert a mapping or list of mappings into a pandas DataFrame."""
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    elif isinstance(data, (list, tuple)):
        frame = pd.DataFrame(list(data))
    elif isinstance(data, dict):
        frame = pd.DataFrame([data])
    else:
        raise TypeError("Unsupported reference data type")

    if columns is not None:
        for column in columns:
            if column not in frame.columns:
                frame[column] = None
        frame = frame.loc[:, columns]
    return frame


def adapt_reference_table(data: Any, *, columns: Optional[list[str]] = None) -> pd.DataFrame:
    """Alias for prepare_reference_frame for compatibility with workflow naming."""
    return prepare_reference_frame(data, columns=columns)
