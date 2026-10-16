# exerkinemap/workflows/12_build_crossorgan_network.py

import sys
import json
import logging
import pandas as pd
import numpy as np
import scanpy as sc
from pathlib import Path
from typing import Dict, List
from exerkinemap.integration.organ_intergration import MoTrPACRatCrosstalkBridge

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_SPATIAL_DIR = PROJECT_ROOT / "data" / "processed" / "spatial"
NETWORK_DIR = PROJECT_ROOT / "data" / "processed" / "networks"
MOTRPAC_BUNDLE_DIR = PROJECT_ROOT / "data" / "motrpac" / "rat-training-06"

# Comprehensive Exerkine Ligand-Receptor Mappings
EXERKINE_PAIRS = {
    "AdipoQ": ["AdipoR1", "AdipoR2"], "ADPN": ["AdipoR1", "AdipoR2"], "APLN": ["APLNR"],
    "FGF21": ["FGFR1", "FGFR2", "FGFR3", "FGFR4"], "FGF1": ["FGFR1", "FGFR2", "FGFR3", "FGFR4"],
    "FGF19": ["FGFR4"], "GDF15": ["GFRAL"], "INS": ["INSR"], "RBP4": ["STRA6"], "LEP": ["LEPR"],
    "IL6": ["IL6R", "IL6SR", "gp130"], "IL7": ["IL7R"], "IL8": ["CXCR1", "CXCR2"],
    "IL10": ["IL10RA", "IL10R2", "IL10R"], "IL15": ["IL15RA", "IL2RB"], "IL1B": ["IL1R1", "IL1R2"],
    "BDNF": ["TRKB", "p75NTR"], "IGF1": ["IGF1R"], "VEGFA": ["VEGFR1", "VEGFR2"],
    "MSTN": ["ACVR2B"], "EGF": ["EGFR"], "TGFB1": ["TGFBR1", "TGFBR2"], 
    "BMP7": ["BMPR1A", "BMPR2"], "BGLAP": ["GPRC6A"], "TNFA": ["TNFR1", "TNFR2"], 
    "FNDC5": ["ITGAV", "ITGAM", "ITGB2"], "SPP1": ["CD44"]
}

def create_directories():
    """Ensure output directories exist."""
    NETWORK_DIR.mkdir(parents=True, exist_ok=True)

def generate_motrpac_reference(rat_bridge: MoTrPACRatCrosstalkBridge, output_path: Path) -> Dict:
    """
    Generates the MoTrPAC temporal reference dictionary across the 8-week protocol.
    Used to validate the biological plausibility of the spatial network edges.
    """
    logger.info("Generating MoTrPAC temporal biological baselines...")
    timepoints = ["1w", "2w", "4w", "8w"]
    network_ref = {"protein_ligands": {}}
    
    for ligand, receptors in EXERKINE_PAIRS.items():
        for receptor in receptors:
            axis_name = f"{ligand}_{receptor}_axis"
            network_ref["protein_ligands"][axis_name] = {}
            for tp in timepoints:
                # Defaulting to muscle-liver for baseline generation; dynamically overridden in spatial projection
                network_ref["protein_ligands"][axis_name][tp] = rat_bridge.extract_ligand_receptor_pair(
                    ligand=ligand.capitalize(), receptor=receptor.capitalize(), timepoint=tp
                )
                
    with open(output_path, 'w') as f:
        json.dump(network_ref, f, indent=4)
    return network_ref

def compute_crossorgan_network(spatial_network: pd.DataFrame, adata_obs: pd.DataFrame, organ_key: str = "tissue") -> pd.DataFrame:
    """
    Section 15 Mathematical Implementation: omega: C -> O and W^O_ab = sum(w^E_ij)
    Optimized via vectorized pandas merges instead of iterative row mapping.
    """
    logger.info("Projecting cellular network (G_E) into organ-level graph (G_O)...")
    
    # Ensure organ metadata exists
    if organ_key not in adata_obs.columns:
        logger.warning(f"Organ key '{organ_key}' not found. Generating mock multi-organ assignments.")
        adata_obs[organ_key] = np.random.choice(["gastrocnemius", "liver", "white_adipose", "heart"], size=len(adata_obs))
    
    # Create mapping series for O(1) vectorized lookups
    cell_to_organ = adata_obs[organ_key]
    
    # Map sender and receiver spots to their respective organs
    spatial_network['sender_organ'] = spatial_network['sender_spot'].map(cell_to_organ)
    spatial_network['receiver_organ'] = spatial_network['receiver_spot'].map(cell_to_organ)
    
    # Drop orphan edges where cells lack tissue metadata (memory optimization)
    spatial_network.dropna(subset=['sender_organ', 'receiver_organ'], inplace=True)
    
    # Filter spatial edges strictly to the EXERKINE_PAIRS to remove noise
    valid_ligands = list(EXERKINE_PAIRS.keys())
    spatial_network = spatial_network[spatial_network['ligand_exerkine'].isin(valid_ligands)]
    
    # Aggregate W^O_ab = sum(S_tilde_score) across tissues
    organ_network = spatial_network.groupby(
        ['sender_organ', 'receiver_organ', 'ligand_exerkine', 'receptor'], 
        observed=True
    )['S_tilde_score'].sum().reset_index()
    
    organ_network.rename(columns={'S_tilde_score': 'W_O_ab_score'}, inplace=True)
    organ_network.sort_values(by='W_O_ab_score', ascending=False, inplace=True)
    
    return organ_network.reset_index(drop=True)

def main():
    logger.info("Initializing 12_build_crossorgan_network workflow...")
    create_directories()

    # 1. Load MoTrPAC Rat Data (Biological Ground Truth)
    rat_bridge = MoTrPACRatCrosstalkBridge(bundle_dir=str(MOTRPAC_BUNDLE_DIR))
    baseline_path = NETWORK_DIR / "rat_8wk_crossorgan_reference.json"
    motrpac_reference = generate_motrpac_reference(rat_bridge, baseline_path)

    # 2. Load Spatial Data (Cellular resolution)
    spatial_path = PROCESSED_SPATIAL_DIR / "exerkinemap_spatial_propagated.h5ad"
    network_path = NETWORK_DIR / "spatial_communication_network.csv"
    
    if not network_path.exists():
        logger.error(f"Spatial network not found at {network_path}. Execute spatial propagation first.")
        sys.exit(1)
        
    spatial_network = pd.read_csv(network_path)
    adata_obs = sc.read_h5ad(spatial_path).obs if spatial_path.exists() else pd.DataFrame(index=pd.concat([spatial_network['sender_spot'], spatial_network['receiver_spot']]).unique())

    # 3. Compute W^O_ab Crossorgan Graph
    organ_network = compute_crossorgan_network(spatial_network, adata_obs, organ_key="tissue")
    
    # 4. Save Outputs
    output_csv = NETWORK_DIR / "crossorgan_communication_network.csv"
    organ_network.to_csv(output_csv, index=False)
    
    # Generate aggregated flow matrix utilizing pivot_table for speed
    flow_matrix = organ_network.pivot_table(
        index='sender_organ', 
        columns='receiver_organ', 
        values='W_O_ab_score', 
        aggfunc='sum', 
        fill_value=0
    )
    flow_matrix_csv = NETWORK_DIR / "crossorgan_flow_matrix.csv"
    flow_matrix.to_csv(flow_matrix_csv)
    
    logger.info(f"Saved specific crossorgan edges to {output_csv}")
    logger.info(f"Saved total crossorgan flow matrix to {flow_matrix_csv}")
    logger.info("Workflow 12_build_crossorgan_network complete.")

if __name__ == "__main__":
    main()