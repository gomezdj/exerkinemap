"""
12_build_crossorgan_network.py

This script constructs the systemic crossorgan communication graph.
Following Section 15 of the Mathematical Model, it computes:
1. The cellular-to-organ mapping function (omega: C -> O).
2. The organ-level edge weights (W^O_ab) by aggregating the spatial/cellular 
   exerkine edge weights (w^E_ij) across distinct tissues.
"""
import sys
import logging
import pandas as pd
import scanpy as sc
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_SPATIAL_DIR = PROJECT_ROOT / "data" / "processed" / "spatial"
NETWORK_DIR = PROJECT_ROOT / "data" / "processed" / "networks"

def create_directories():
    """Ensure output directories exist."""
    NETWORK_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load the cellular/spatial communication network and metadata."""
    network_path = NETWORK_DIR / "spatial_communication_network.csv"
    spatial_path = PROCESSED_SPATIAL_DIR / "exerkinemap_spatial_propagated.h5ad"
    
    if not network_path.exists():
        logger.error(f"Spatial network not found at {network_path}. Run script 09 first.")
        sys.exit(1)
        
    logger.info("Loading spatial communication network (w^E_ij components)...")
    spatial_network = pd.read_csv(network_path)
    
    logger.info("Loading spatial AnnData to extract organ mappings (omega: C -> O)...")
    if spatial_path.exists():
        adata = sc.read_h5ad(spatial_path)
        metadata = adata.obs
    else:
        logger.warning(f"Spatial dataset not found. Will simulate organ mapping for demonstration.")
        metadata = pd.DataFrame(index=pd.concat([spatial_network['sender_spot'], spatial_network['receiver_spot']]).unique())
        
    return spatial_network, metadata

def map_cells_to_organs(metadata, organ_key="tissue"):
    """
    Section 15: omega: C -> O
    Maps each cellular spot (c_i) to its respective organ (o_a).
    """
    logger.info(f"Mapping cells to organs using metadata key '{organ_key}'...")
    
    # If the dataset represents a single tissue but we want to simulate crossorgan (or if the key is missing)
    if organ_key not in metadata.columns:
        logger.warning(f"Organ key '{organ_key}' not found in metadata. Generating mock multi-organ assignments.")
        organs = ["Skeletal Muscle", "Liver", "Adipose Tissue", "Heart", "Brain"]
        # In a real MoTrPAC systemic integration, spots/cells would inherently have this metadata 
        # from their source sample or through plasma exerkine mapping.
        import numpy as np
        metadata[organ_key] = np.random.choice(organs, size=len(metadata))
        
    # Return a dictionary mapping spot_id -> organ
    return metadata[organ_key].to_dict()

def compute_crossorgan_network(spatial_network, cell_to_organ_map):
    """
    Section 15: Organ-level edge weight W^O_ab
    W^O_ab = sum_{i in o_a, j in o_b} w^E_ij
    Aggregates the S_tilde_score for all crossactions between organ a and organ b.
    """
    logger.info("Projecting cellular network (G_E) into organ-level graph (G_O)...")
    
    # Apply omega: C -> O mapping to the network edges
    spatial_network['sender_organ'] = spatial_network['sender_spot'].map(cell_to_organ_map)
    spatial_network['receiver_organ'] = spatial_network['receiver_spot'].map(cell_to_organ_map)
    
    # Drop unmapped edges just in case
    spatial_network = spatial_network.dropna(subset=['sender_organ', 'receiver_organ'])
    
    # Aggregate weights W^O_ab = sum(S_tilde_score) grouped by Sender Organ and Receiver Organ
    # We also keep the specific ligand-receptor pairs to see WHICH exerkines mediate the crossorgan crosstalk
    organ_network = spatial_network.groupby(
        ['sender_organ', 'receiver_organ', 'ligand_exerkine', 'receptor']
    )['S_tilde_score'].sum().reset_index()
    
    organ_network = organ_network.rename(columns={'S_tilde_score': 'W_O_ab_score'})
    
    # Sort by strongest crossorgan signaling pathways
    organ_network = organ_network.sort_values(by='W_O_ab_score', ascending=False).reset_index(drop=True)
    
    logger.info(f"Crossorgan network constructed with {len(organ_network)} specific molecular edges.")
    return organ_network

def main():
    logger.info("Initializing 12_build_crossorgan_network workflow...")
    create_directories()

    # 1. Load data
    spatial_network, metadata = load_data()
    
    # 2. Extract omega: C -> O
    cell_to_organ_map = map_cells_to_organs(metadata, organ_key="tissue")
    
    # 3. Compute W^O_ab (Crossorgan Graph G_O)
    organ_network = compute_crossorgan_network(spatial_network, cell_to_organ_map)
    
    # 4. Save Outputs
    output_csv = NETWORK_DIR / "crossorgan_communication_network.csv"
    organ_network.to_csv(output_csv, index=False)
    
    # Also save a highly aggregated matrix just showing total organ-to-organ signal flow
    flow_matrix = organ_network.groupby(['sender_organ', 'receiver_organ'])['W_O_ab_score'].sum().unstack(fill_value=0)
    flow_matrix_csv = NETWORK_DIR / "crossorgan_flow_matrix.csv"
    flow_matrix.to_csv(flow_matrix_csv)
    
    logger.info(f"Saved specific crossorgan edges to {output_csv}")
    logger.info(f"Saved total crossorgan flow matrix to {flow_matrix_csv}")
    logger.info("Workflow 12_build_crossorgan_network complete.")

if __name__ == "__main__":
    main()
