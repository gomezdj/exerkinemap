"""Ligand-receptor database helpers for EXERKINEMAP."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


class FANTOM5LigandReceptorDatabase:
    """Minimal loader for FANTOM5-style ligand-receptor data."""

    def __init__(self, data_path: Optional[Path | str] = None):
        self.data_path = Path(data_path) if data_path is not None else None

    def load(self) -> pd.DataFrame:
        if self.data_path is not None and self.data_path.exists():
            return pd.read_csv(self.data_path)

        # Fallback to an in-memory placeholder table that mirrors the expected columns.
        return pd.DataFrame(
            columns=["source_genesymbol", "target_genesymbol", "source", "target", "confidence"]
        )

    def filter_exerkines(self, exerkine_genes: list[str]) -> pd.DataFrame:
        df = self.load()
        if df.empty:
            return df
        return df[df["source_genesymbol"].isin(exerkine_genes)].copy()


def load_fantom5_lri(data_path: Optional[Path | str] = None) -> pd.DataFrame:
    return FANTOM5LigandReceptorDatabase(data_path=data_path).load()
