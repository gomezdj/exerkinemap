"""
03_preprocess_spatial.py

This script processes the raw spatial omics data.
It handles spatial transcriptomics or proteomics data, performs QC,
normalizes the expression matrices, and computes spatial neighborhoods 
and connectivity graphs using Squidpy.
"""
import sys
import logging
import scanpy as sc
import squidpy as sq
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Resolve repository root and set data paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_SPATIAL_DIR = PROJECT_ROOT / "data" / "raw" / "spatial"
PROCESSED_SPATIAL_DIR = PROJECT_ROOT / "data" / "processed" / "spatial"

def create_directories():
    """Ensure output directories exist."""
    PROCESSED_SPATIAL_DIR.mkdir(parents=True, exist_ok=True)

def load_spatial_data(filename="spatial_omics.h5ad"):
    """Load the spatial .h5ad file."""
    file_path = RAW_SPATIAL_DIR / filename
    
    if not file_path.exists():
        logger.error(f"Spatial dataset not found: {file_path}")
        sys.exit(1)
        
    logger.info(f"Loading spatial data from {file_path}...")
    adata = sc.read_h5ad(file_path)
    return adata

def run_spatial_qc(adata):
    """Filter out low-quality spots/cells and genes."""
    logger.info("Starting Spatial Quality Control...")
    
    # Calculate QC metrics based on mitochondrial genes if present
    adata.var['mt'] = adata.var_names.str.startswith('MT-') | adata.var_names.str.startswith('mt-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    
    initial_spots, initial_genes = adata.shape
    
    # Basic filtering (adjust min_counts based on Visium vs Xenium/CODEX depth)
    sc.pp.filter_cells(adata, min_counts=500)
    sc.pp.filter_genes(adata, min_cells=3)
    
    final_spots, final_genes = adata.shape
    logger.info(f"QC Complete. Filtered out {initial_spots - final_spots} spots and {initial_genes - final_genes} genes.")
    
    return adata

def normalize_and_scale(adata):
    """Normalize total counts per spot and log-transform."""
    logger.info("Normalizing and log-transforming spatial data...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    logger.info("Extracting highly variable genes...")
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    
    # Save raw state
    adata.raw = adata
    return adata

def compute_spatial_graphs(adata):
    """Compute spatial neighborhoods and graphs using Squidpy."""
    logger.info("Computing spatial neighborhoods...")
    
    # Check for coordinate keys, usually 'spatial' in adata.obsm
    if 'spatial' not in adata.obsm.keys():
        logger.warning("No 'spatial' key found in adata.obsm. Skipping spatial graph construction.")
        return adata
        
    # Construct spatial neighbors graph. 
    # Defaults to a generic coordinate-based KNN approach suitable for single-cell spatial data.
    # For grid-based Visium data, coord_type="grid" with n_rings=1 can be used instead.
    sq.gr.spatial_neighbors(adata, n_neighs=6, coord_type="generic")
    
    logger.info("Spatial neighborhood graph computed and stored in adata.obsp.")
    return adata

def main():
    logger.info("Initializing 03_preprocess_spatial workflow...")
    create_directories()

    # Load data
    adata = load_spatial_data()

    # QC and Normalization
    adata = run_spatial_qc(adata)
    adata = normalize_and_scale(adata)

    # Spatial Graphs
    adata = compute_spatial_graphs(adata)

    # Save processed AnnData
    output_path = PROCESSED_SPATIAL_DIR / "spatial_omics_processed.h5ad"
    logger.info(f"Saving processed spatial dataset to {output_path}...")
    adata.write(output_path)
    
    logger.info("Workflow 03_preprocess_spatial complete.")

if __name__ == "__main__":
    main()
