"""Normalization helpers for count matrices and AnnData objects."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def normalize_counts(counts: Any, method: str = "log1p") -> np.ndarray:
    """Normalize a numeric array to a dense float array."""
    arr = np.asarray(counts, dtype=float)
    if method == "log1p":
        return np.log1p(arr)
    if method == "zscore":
        mean = arr.mean(axis=0, keepdims=True)
        std = arr.std(axis=0, keepdims=True)
        std = np.where(std == 0, 1.0, std)
        return (arr - mean) / std
    if method == "sqrt":
        return np.sqrt(arr)
    raise ValueError(f"Unsupported normalization method: {method}")


def normalize_matrix(matrix: Any, method: str = "log1p") -> np.ndarray:
    """Alias for normalize_counts for matrix-like inputs."""
    return normalize_counts(matrix, method=method)


def normalize_anndata(adata: Any, layer: Optional[str] = None, method: str = "log1p", copy: bool = False) -> Any:
    """Normalize the matrix in an AnnData object if scanpy is available."""
    try:
        import scanpy as sc
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError("scanpy is required for AnnData normalization") from exc

    adata_out = adata.copy() if copy else adata
    if layer is not None:
        X = adata_out.layers[layer]
    else:
        X = adata_out.X

    normalized = normalize_counts(X, method=method)
    if layer is not None:
        adata_out.layers[layer] = normalized
    else:
        adata_out.X = normalized
    if copy:
        return adata_out
    return adata_out
