"""Basic quality-control helpers for single-cell and spatial data."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def summarize_qc(adata: Any) -> Dict[str, float]:
    """Return simple QC summary metrics from an AnnData-like object."""
    if hasattr(adata, "n_obs") and hasattr(adata, "n_vars"):
        return {
            "n_obs": float(adata.n_obs),
            "n_vars": float(adata.n_vars),
        }
    return {"n_obs": 0.0, "n_vars": 0.0}


def filter_low_quality_cells(
    adata: Any,
    min_genes: int = 200,
    min_counts: int = 1000,
    inplace: bool = False,
    copy: bool = False,
) -> Any:
    """Filter low-quality cells by minimum gene count and total count."""
    if copy:
        adata_out = adata.copy()
    else:
        adata_out = adata if inplace else adata.copy()

    try:
        import scanpy as sc
    except ImportError:  # pragma: no cover - optional dependency guard
        if not hasattr(adata_out, "obs"):
            raise ImportError("scanpy is required for QC filtering")
        counts = np.asarray(adata_out.X)
        gene_count = np.count_nonzero(counts > 0, axis=1)
        total_count = counts.sum(axis=1)
        mask = (gene_count >= min_genes) & (total_count >= min_counts)
        adata_out = adata_out[mask]
        return adata_out

    sc.pp.filter_cells(adata_out, min_genes=min_genes)
    sc.pp.filter_cells(adata_out, min_counts=min_counts)
    return adata_out
