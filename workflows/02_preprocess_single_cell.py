"""
02_preprocess_single_cell.py

This script processes the raw single-cell transcriptomics data (HuBMAP, Human Cell Atlas).
It handles both compiled .h5ad objects and raw count matrices, merges metadata,
performs quality control (QC), normalizes expression, and extracts highly 
variable genes for downstream integration.
"""
import sys
import logging
import scanpy as sc
import pandas as pd
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Resolve repository root and set data paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_SC_DIR = PROJECT_ROOT / "data" / "raw" / "single_cell"
METADATA_DIR = PROJECT_ROOT / "data" / "raw" / "metadata"
PROCESSED_SC_DIR = PROJECT_ROOT / "data" / "processed" / "anndata"

def create_directories():
    """Ensure output directories exist."""
    PROCESSED_SC_DIR.mkdir(parents=True, exist_ok=True)

def load_and_merge_data(h5ad_name="sc_transcriptomics.h5ad", meta_name="sc_metadata.csv"):
    """Load the .h5ad file and merge external metadata if available."""
    h5ad_path = RAW_SC_DIR / h5ad_name
    meta_path = METADATA_DIR / meta_name

    if not h5ad_path.exists():
        logger.error(f"Single-cell dataset not found: {h5ad_path}")
        sys.exit(1)

    logger.info(f"Loading single-cell data from {h5ad_path}...")
    adata = sc.read_h5ad(h5ad_path)

    # Merge CSV metadata if it exists
    if meta_path.exists():
        logger.info(f"Merging metadata from {meta_path}...")
        meta_df = pd.read_csv(meta_path, index_col=0)
        # Ensure indices align before joining
        adata.obs = adata.obs.join(meta_df, how='left', rsuffix='_csv')
    else:
        logger.warning(f"Metadata file not found at {meta_path}. Proceeding with internal metadata only.")
        
    return adata

def run_quality_control(adata):
    """Filter out low-quality cells and genes."""
    logger.info("Starting Quality Control...")
    
    # Identify mitochondrial genes to calculate QC metrics
    adata.var['mt'] = adata.var_names.str.startswith('MT-') | adata.var_names.str.startswith('mt-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

    initial_cells, initial_genes = adata.shape
    
    # Standard filtering thresholds
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    
    # Filter cells with too high mitochondrial content (e.g., > 15%) or abnormally high counts
    adata = adata[adata.obs.pct_counts_mt < 15, :]
    
    final_cells, final_genes = adata.shape
    logger.info(f"QC Complete. Filtered out {initial_cells - final_cells} cells and {initial_genes - final_genes} genes.")
    
    return adata

def normalize_and_scale(adata):
    """Normalize total counts per cell, log-transform, and identify highly variable genes."""
    logger.info("Normalizing and log-transforming data...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    logger.info("Extracting highly variable genes...")
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    
    # Save the raw state before scaling
    adata.raw = adata
    
    return adata

def main():
    logger.info("Initializing 02_preprocess_single_cell workflow...")
    create_directories()

    # Load data
    adata = load_and_merge_data()

    # QC and Normalization
    adata = run_quality_control(adata)
    adata = normalize_and_scale(adata)

    # Save processed AnnData
    output_path = PROCESSED_SC_DIR / "motrpac_sc_processed.h5ad"
    logger.info(f"Saving processed dataset to {output_path}...")
    adata.write(output_path)
    
    logger.info("Workflow 02_preprocess_single_cell complete.")

if __name__ == "__main__":
    main()
