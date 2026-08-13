"""Protein reference utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .adapatation import prepare_reference_frame


def build_protein_reference(data: Any, *, output_path: Any | None = None) -> pd.DataFrame:
    """Create a standardized protein reference table."""
    frame = prepare_reference_frame(data, columns=["protein_id", "gene_symbol", "sequence"])
    if output_path is not None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out_path, index=False)
    return frame


def load_protein_reference(path: Any) -> pd.DataFrame:
    """Load a protein reference table from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)
