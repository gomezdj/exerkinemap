"""
anndata_io.py
Read and write utilities for Scanpy AnnData (.h5ad) single-cell and spatial objects.
"""
import logging
import scanpy as sc
from pathlib import Path

logger = logging.getLogger(__name__)

def load_anndata(filepath: Path) -> sc.AnnData:
    """Loads an AnnData object from an .h5ad file."""
    filepath = Path(filepath)
    if not filepath.exists():
        logger.error(f"AnnData file not found at {filepath}")
        raise FileNotFoundError(f"AnnData file not found at {filepath}")
    logger.info(f"Loading AnnData object from {filepath}")
    adata = sc.read_h5ad(filepath)
    logger.info(f"Loaded AnnData with dimensions: {adata.shape}")
    return adata

def save_anndata(adata: sc.AnnData, filepath: Path) -> None:
    """Saves an AnnData object to an .h5ad file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving AnnData object to {filepath}")
    adata.write_h5ad(filepath)
    logger.info("AnnData successfully saved.")
