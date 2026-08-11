"""
10_signal_propagation.py

This script models the spatial diffusion of exerkine signals across the tissue.
Following Sections 11, 12, and 13 of the Mathematical Model, it computes:
1. The Exerkine Signaling Network Laplacian (L_E = D - W_E).
2. The Initial Exerkine Secretion State (f_0).
3. The Propagated Signal Matrix F(t) = exp(-t * L_E) * f_0.
"""
import sys
import logging
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import expm_multiply
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_SPATIAL_DIR = PROJECT_ROOT / "data" / "processed" / "spatial"
EXERKINES_DIR = PROJECT_ROOT / "data" / "processed" / "exerkines"
NETWORK_DIR = PROJECT_ROOT / "data" / "processed" / "networks"
PROPAGATION_DIR = PROJECT_ROOT / "data" / "processed" / "pathways"  # Storing downstream signals here

def create_directories():
    """Ensure output directories exist."""
    PROPAGATION_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load spatial omics data, identified exerkines, and the spatial communication network."""
    spatial_path = PROCESSED_SPATIAL_DIR / "motrpac_spatial_processed.h5ad"
    exerkines_path = EXERKINES_DIR / "identified_exerkines.csv"
    network_path = NETWORK_DIR / "spatial_communication_network.csv"
    
    if not all(p.exists() for p in [spatial_path, exerkines_path, network_path]):
        logger.error("Missing required inputs. Ensure scripts 03, 07, and 09 were run.")
        sys.exit(1)
        
    logger.info("Loading spatial dataset, exerkine scores, and spatial network...")
    adata = sc.read_h5ad(spatial_path)
    exerkines_df = pd.read_csv(exerkines_path)
    spatial_network = pd.read_csv(network_path)
    
    return adata, exerkines_df, spatial_network

def compute_graph_laplacian(adata, spatial_network):
    """
    Section 11: Graph Laplacian L_E = D - W_E
    Constructs the sparse adjacency matrix (W_E) by summing S_tilde across all 
    ligand-receptor pairs for each cell-cell edge, then computes the Laplacian.
    """
    logger.info("Constructing Graph Laplacian (L_E)...")
    
    # Aggregate S_tilde_score for each sender-receiver pair (i, j)
    # w^E_ij = sum(S_tilde_ij^(lk, rm))
    edge_weights = spatial_network.groupby(['sender_spot', 'receiver_spot'])['S_tilde_score'].sum().reset_index()
    
    # Map spot names to numerical indices for sparse matrix construction
    spot_to_idx = {name: idx for idx, name in enumerate(adata.obs_names)}
    
    row_idx = edge_weights['sender_spot'].map(spot_to_idx).values
    col_idx = edge_weights['receiver_spot'].map(spot_to_idx).values
    weights = edge_weights['S_tilde_score'].values
    
    num_spots = adata.n_obs
    
    # Construct Sparse Adjacency Matrix (W_E)
    W_E = coo_matrix((weights, (row_idx, col_idx)), shape=(num_spots, num_spots)).tocsr()
    
    # Compute Degree Matrix (D)
    # D_ii = sum_j(w_ij)
    out_degrees = np.array(W_E.sum(axis=1)).flatten()
    D = diags(out_degrees, format='csr')
    
    # Compute Graph Laplacian (L_E)
    L_E = D - W_E
    
    logger.info(f"Laplacian L_E constructed. Shape: {L_E.shape}, Non-zero elements: {L_E.nnz}")
    return L_E

def compute_initial_state(adata, exerkines_df):
    """
    Section 12: Initial Exerkine Secretion State f_0
    f_0(i) = sum_{l_k in E} rho_{l_k} * x_i(l_k)
    """
    logger.info("Computing Initial Secretion State (f_0)...")
    
    # Extract dense expression matrix
    expr_matrix = adata.raw.to_adata().to_df() if adata.raw is not None else adata.to_df()
    
    num_spots = adata.n_obs
    f_0 = np.zeros(num_spots)
    
    # Map rho scores to genes
    rho_scores = dict(zip(exerkines_df['gene'], exerkines_df['rho_q']))
    
    for l_k, rho in rho_scores.items():
        if l_k in expr_matrix.columns:
            # rho_{l_k} * x_i(l_k)
            f_0 += rho * expr_matrix[l_k].values
            
    logger.info("Initial state f_0 formulated.")
    return f_0

def propagate_signal(L_E, f_0, t=1.0):
    """
    Section 13: Exerkine Signal Propagation F(t)
    F(t) = exp(-t * L_E) * f_0
    Uses scipy's expm_multiply for highly efficient sparse matrix exponential action.
    """
    logger.info(f"Propagating signal over communication graph (t={t})...")
    
    # We want to compute exp(-t * L_E) * f_0
    # scipy's expm_multiply computes exp(A) * v. So we set A = -t * L_E
    A = -t * L_E
    
    # Execute the heat diffusion
    F_t = expm_multiply(A, f_0)
    
    logger.info("Signal propagation complete.")
    return F_t

def main():
    logger.info("Initializing 10_signal_propagation workflow...")
    create_directories()

    # 1. Load Data
    adata, exerkines_df, spatial_network = load_data()
    
    # 2. Compute Graph Laplacian (L_E)
    L_E = compute_graph_laplacian(adata, spatial_network)
    
    # 3. Compute Initial Secretion State (f_0)
    f_0 = compute_initial_state(adata, exerkines_df)
    
    # 4. Propagate Signal F(t)
    # t represents diffusion time. Can be swept across multiple values for temporal modeling.
    t_val = 2.0 
    F_t = propagate_signal(L_E, f_0, t=t_val)
    
    # 5. Save Outputs
    # Store the propagated signal back into the spatial AnnData object for easy visualization
    adata.obs['initial_exerkine_state'] = f_0
    adata.obs[f'propagated_exerkine_signal_t{t_val}'] = F_t
    
    # Save updated AnnData
    spatial_out_path = PROCESSED_SPATIAL_DIR / "exerkinemap_spatial_propagated.h5ad"
    adata.write(spatial_out_path)
    logger.info(f"Updated spatial dataset saved to {spatial_out_path}")
    
    # Save the raw propagated vector
    np.save(PROPAGATION_DIR / f"F_t_signal_t{t_val}.npy", F_t)
    
    logger.info("Workflow 10_signal_propagation complete.")

if __name__ == "__main__":
    main()
