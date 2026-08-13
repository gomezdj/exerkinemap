"""Preprocessing helpers tailored for single-cell data."""

from __future__ import annotations

from typing import Any, Optional

from .normalization import normalize_anndata
from .quality_control import filter_low_quality_cells


def normalize_single_cell(adata: Any, layer: Optional[str] = None, method: str = "log1p", copy: bool = True) -> Any:
    """Normalize a single-cell AnnData object."""
    return normalize_anndata(adata, layer=layer, method=method, copy=copy)


def basic_single_cell_preprocessing(
    adata: Any,
    min_genes: int = 200,
    min_counts: int = 1000,
    layer: Optional[str] = None,
    method: str = "log1p",
    copy: bool = True,
) -> Any:
    """Apply a minimal quality-control and normalization pipeline to single-cell data."""
    adata_out = filter_low_quality_cells(adata, min_genes=min_genes, min_counts=min_counts, copy=copy)
    return normalize_single_cell(adata_out, layer=layer, method=method, copy=False)
