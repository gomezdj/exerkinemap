"""
09_spatial_communication.py

This script constructs the spatially resolved exerkine communication network.
Following Section 10 of the Mathematical Model, it computes:
1. The Spatial Kernel (K^S_ij) based on physical tissue coordinates (Section 10.3).
2. The Spatially Informed Interaction Score (S_tilde_ij) for exerkine signaling (Section 10.4).
"""
import sys
import logging
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial.distance import cdist
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_SPATIAL_DIR = PROJECT_ROOT / "data" / "processed" / "spatial"
LR_DIR = PROJECT_ROOT / "data" / "processed" / "ligand_receptor"
NETWORK_DIR = PROJECT_ROOT / "data" / "processed" / "networks"

def create_directories():
    """Ensure output directories exist."""
    NETWORK_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load processed spatial omics data and the LRI prior network."""
    spatial_path = PROCESSED_SPATIAL_DIR / "motrpac_spatial_processed.h5ad"
    lr_path = LR_DIR / "exerkine_lr_network.csv"
    
    if not spatial_path.exists() or not lr_path.exists():
        logger.error("Missing required inputs. Ensure scripts 03 and 08 were run.")
        sys.exit(1)
        
    logger.info("Loading spatial dataset and base exerkine LR network...")
    adata = sc.read_h5ad(spatial_path)
    lr_network = pd.read_csv(lr_path)
    
    # We only need the unique biological interactions (Ligand, Receptor, alpha, Gamma)
    # Deduplicate the cluster-based interactions from step 08 to get the pure molecular pairs
    unique_lr_pairs = lr_network[['ligand_exerkine', 'receptor', 'alpha_km', 'Gamma_km']].drop_duplicates()
    
    return adata, unique_lr_pairs

def compute_spatial_kernel(adata, sigma_S=150.0):
    """
    Section 10.3: Spatial Kernel K_{ij}^S
    Computes Gaussian distance decay between all spatial spots i and j.
    Formula: K_{ij}^S = exp(-|s_i - s_j|^2 / (2 * sigma_S^2))
    """
    logger.info(f"Computing Spatial Kernel K^S with sigma_S={sigma_S}...")
    
    if 'spatial' not in adata.obsm:
        logger.error("Spatial coordinates 'spatial' not found in adata.obsm.")
        sys.exit(1)
        
    coords = adata.obsm['spatial']
    
    # |s_i - s_j|^2 (Squared Euclidean distance)
    dist_sq = cdist(coords, coords, metric='sqeuclidean')
    
    # Apply Gaussian kernel
    K_S = np.exp(-dist_sq / (2 * (sigma_S ** 2)))
    
    # Sparsify the kernel to remove negligible long-distance interactions (e.g., < 0.01)
    # This prevents the network from becoming a massive dense graph
    K_S[K_S < 0.01] = 0.0 
    
    logger.info(f"Spatial Kernel computed. Non-zero spatial edges: {np.count_nonzero(K_S)}")
    return K_S

def compute_spatially_informed_interactions(adata, lr_pairs, K_S):
    """
    Section 10.4: Spatially Informed Interaction Score
    S_tilde_{ij} = x_i(l_k) * x_j(r_m) * alpha_km * Gamma_km * K_{ij}^S
    """
    logger.info("Computing Spatially Informed Interactions (S_tilde_ij)...")
    
    # Extract dense expression matrix for faster computation
    # Use .raw if it exists to access normalized counts before scaling/subsetting
    if adata.raw is not None:
        expr_matrix = adata.raw.to_adata().to_df()
    else:
        expr_matrix = adata.to_df()
        
    spatial_edges = []
    spot_names = adata.obs_names
    
    # Get the indices of non-zero spatial connections to optimize the loops
    nonzero_i, nonzero_j = np.nonzero(K_S)
    
    for _, row in lr_pairs.iterrows():
        l_k = row['ligand_exerkine']
        r_m = row['receptor']
        alpha = row['alpha_km']
        gamma = row['Gamma_km']
        
        # Check if the genes are captured in the spatial dataset
        if l_k not in expr_matrix.columns or r_m not in expr_matrix.columns:
            continue
            
        # x_i(l_k): Ligand expression in sender spots
        x_i = expr_matrix[l_k].values
        # x_j(r_m): Receptor expression in receiver spots
        x_j = expr_matrix[r_m].values
        
        # Vectorized computation of S_tilde for valid spatial neighbors
        # S_tilde_{ij} = x_i * x_j * K^S_{ij} * alpha * gamma
        for idx in range(len(nonzero_i)):
            i = nonzero_i[idx]
            j = nonzero_j[idx]
            
            sender_expr = x_i[i]
            receiver_expr = x_j[j]
            
            if sender_expr > 0 and receiver_expr > 0:
                k_s_val = K_S[i, j]
                
                s_tilde = sender_expr * receiver_expr * alpha * gamma * k_s_val
                
                if s_tilde > 0:
                    spatial_edges.append({
                        "sender_spot": spot_names[i],
                        "receiver_spot": spot_names[j],
                        "ligand_exerkine": l_k,
                        "receptor": r_m,
                        "K_S_ij": k_s_val,
                        "S_tilde_score": s_tilde
                    })
                    
    df_edges = pd.DataFrame(spatial_edges)
    logger.info(f"Identified {len(df_edges)} spatially informed signaling edges.")
    return df_edges

def main():
    logger.info("Initializing 09_spatial_communication workflow...")
    create_directories()

    # 1. Load spatial data and unique LR pairs
    adata, lr_pairs = load_data()
    
    # 2. Compute the Spatial Kernel (K_S)
    # sigma_S dictates the diffusion/communication radius. Adjust based on Visium vs Xenium scale.
    K_S = compute_spatial_kernel(adata, sigma_S=100.0)
    
    # 3. Compute final spatially informed edges
    spatial_network = compute_spatially_informed_interactions(adata, lr_pairs, K_S)
    
    # 4. Save Network
    output_path = NETWORK_DIR / "spatial_communication_network.csv"
    spatial_network.to_csv(output_path, index=False)
    
    logger.info(f"Successfully saved spatial communication network to {output_path}")
    logger.info("Workflow 09_spatial_communication complete.")

if __name__ == "__main__":
    main()
