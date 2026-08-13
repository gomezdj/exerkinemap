"""Preprocessing helpers tailored for spatial data."""

from __future__ import annotations

from typing import Any, Optional

from .normalization import normalize_anndata


def normalize_spatial(adata: Any, layer: Optional[str] = None, method: str = "log1p", copy: bool = True) -> Any:
    """Normalize a spatial AnnData object."""
    return normalize_anndata(adata, layer=layer, method=method, copy=copy)


def basic_spatial_preprocessing(
    adata: Any,
    layer: Optional[str] = None,
    method: str = "log1p",
    copy: bool = True,
) -> Any:
    """Apply a minimal normalization pipeline to spatial data."""
    return normalize_spatial(adata, layer=layer, method=method, copy=copy)
