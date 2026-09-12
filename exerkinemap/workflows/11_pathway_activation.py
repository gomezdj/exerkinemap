"""
11_pathway_activation.py

This script calculates the downstream biological pathway activation induced by 
exerkine signaling across the spatial tissue.
Following Section 14 of the Mathematical Model, it computes:
1. The Receptor-Pathway contribution matrix (beta_mp).
2. The Pathway Activation Score A_j(p) = sum_i sum_{l_k, r_m} (S_tilde_ij * beta_mp).
"""
import sys
import logging
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_SPATIAL_DIR = PROJECT_ROOT / "data" / "processed" / "spatial"
NETWORK_DIR = PROJECT_ROOT / "data" / "processed" / "networks"
REF_DIR = PROJECT_ROOT / "data" / "references" / "pathways"
PATHWAY_DIR = PROJECT_ROOT / "data" / "processed" / "pathways"

def create_directories():
    """Ensure output directories exist."""
    PATHWAY_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load the spatial network and the propagated spatial dataset."""
    network_path = NETWORK_DIR / "spatial_communication_network.csv"
    spatial_path = PROCESSED_SPATIAL_DIR / "exerkinemap_spatial_propagated.h5ad"
    
    if not network_path.exists():
        logger.error(f"Spatial network not found at {network_path}. Run script 09 first.")
        sys.exit(1)
        
    logger.info("Loading spatial communication network (S_tilde_ij)...")
    spatial_network = pd.read_csv(network_path)
    
    logger.info("Loading propagated spatial AnnData...")
    if spatial_path.exists():
        adata = sc.read_h5ad(spatial_path)
    else:
        logger.warning(f"Propagated data not found at {spatial_path}. Will not update AnnData.")
        adata = None
        
    return spatial_network, adata

def build_receptor_pathway_matrix(receptors):
    """
    Section 14: Receptor-Pathway Contribution (beta_mp).
    Maps receptors to biological pathways (e.g., using Reactome or KEGG).
    For this implementation, we parse a simplified pathway mapping.
    """
    logger.info("Building receptor-to-pathway contribution matrix (beta_mp)...")
    
    # In a production run, you would parse the Reactome/Ensembl mappings downloaded in script refs.
    # We will simulate the beta_mp weights for the receptors present in our network.
    # beta_mp represents the contribution of receptor r_m to pathway p.
    
    # Mocking pathway databases for demonstration (e.g., PI3K-Akt, MAPK, JAK-STAT)
    pathways = ["PI3K-Akt Signaling", "MAPK Signaling", "JAK-STAT Signaling", "mTOR Signaling", "AMPK Signaling"]
    
    mapping_records = []
    for r_m in receptors:
        # Assign random pathways to receptors to simulate beta_mp > 0
        assigned_pathways = np.random.choice(pathways, size=np.random.randint(1, 4), replace=False)
        for p in assigned_pathways:
            mapping_records.append({
                "receptor": r_m,
                "pathway": p,
                "beta_mp": np.random.uniform(0.5, 1.0) # Contribution weight
            })
            
    beta_df = pd.DataFrame(mapping_records)
    logger.info(f"Mapped {len(receptors)} unique receptors to {len(pathways)} pathways.")
    return beta_df

def compute_pathway_activation(spatial_network, beta_df):
    """
    Section 14: Pathway Activation Score A_j(p)
    A_j(p) = sum_i sum_{l_k, r_m} [ S_tilde_ij^(lk, rm) * beta_mp ]
    """
    logger.info("Computing Pathway Activation Matrix A_P...")
    
    # Merge the spatial interactions with the receptor-pathway contributions
    # This brings beta_mp into the same dataframe as S_tilde_score
    merged_df = pd.merge(spatial_network, beta_df, on='receptor', how='inner')
    
    # Calculate the raw contribution before summation: S_tilde * beta_mp
    merged_df['pathway_signal'] = merged_df['S_tilde_score'] * merged_df['beta_mp']
    
    # Aggregate (sum) over all senders (i) and ligand-receptor pairs (lk, rm)
    # Grouping by receiver_spot (j) and pathway (p)
    activation_scores = merged_df.groupby(['receiver_spot', 'pathway'])['pathway_signal'].sum().reset_index()
    
    # Pivot to create the final N x Q matrix (Spots x Pathways) -> A_P
    A_P = activation_scores.pivot(index='receiver_spot', columns='pathway', values='pathway_signal').fillna(0)
    
    logger.info(f"Pathway Activation Matrix A_P constructed. Shape: {A_P.shape}")
    return A_P

def main():
    logger.info("Initializing 11_pathway_activation workflow...")
    create_directories()

    # 1. Load Data
    spatial_network, adata = load_data()
    unique_receptors = spatial_network['receptor'].unique()
    
    # 2. Build beta_mp matrix
    beta_df = build_receptor_pathway_matrix(unique_receptors)
    
    # 3. Compute A_j(p)
    A_P = compute_pathway_activation(spatial_network, beta_df)
    
    # 4. Save Outputs
    output_csv = PATHWAY_DIR / "pathway_activation_matrix.csv"
    A_P.to_csv(output_csv)
    logger.info(f"Saved Pathway Activation Matrix to {output_csv}")
    
    # Optional: Integrate directly into spatial AnnData object for visualization
    if adata is not None:
        logger.info("Updating spatial AnnData object with pathway activations...")
        # Ensure indices align
        common_spots = A_P.index.intersection(adata.obs_names)
        
        # Store A_P in obsm for downstream UMAP integration (Section 16)
        aligned_A_P = A_P.reindex(adata.obs_names).fillna(0)
        adata.obsm['pathway_activation'] = aligned_A_P.values
        
        # Save updated h5ad
        adata.write(PROCESSED_SPATIAL_DIR / "exerkinemap_spatial_propagated.h5ad")
        logger.info("Updated spatial dataset saved.")
        
    logger.info("Workflow 11_pathway_activation complete.")

if __name__ == "__main__":
    main()
