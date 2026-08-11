"""
08_build_ligand_receptor_network.py

This script constructs the base Exerkine-Receptor interaction network.
Following Section 10 of the Mathematical Model, it computes:
1. The Biological Prior (alpha_km) using established LR databases via LIANA.
2. Molecular Compatibility (Gamma_km) between ligands and receptors using PLM embeddings.
3. The base expression potential (S_ij) across identified cell types.
"""
import sys
import logging
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

from src.ligand_receptor.database import load_fantom5_lri

# Configure logging
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_SC_DIR = PROJECT_ROOT / "data" / "processed" / "anndata"
EXERKINES_DIR = PROJECT_ROOT / "data" / "processed" / "exerkines"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
LR_DIR = PROJECT_ROOT / "data" / "processed" / "ligand_receptor"

def create_directories():
    """Ensure output directories exist."""
    LR_DIR.mkdir(parents=True, exist_ok=True)

def load_inputs():
    """Load the single-cell data, identified exerkines, and PLM embeddings."""
    sc_path = PROCESSED_SC_DIR / "motrpac_sc_processed.h5ad"
    exerkines_path = EXERKINES_DIR / "identified_exerkines.csv"
    
    if not sc_path.exists() or not exerkines_path.exists():
        logger.error("Missing required input files. Ensure scripts 02 and 07 were run.")
        sys.exit(1)
        
    logger.info("Loading single-cell dataset and identified exerkine set (E)...")
    adata = sc.read_h5ad(sc_path)
    exerkines_df = pd.read_csv(exerkines_path)
    exerkine_genes = exerkines_df['gene'].tolist()
    
    # Load PLM embeddings for Gamma_km (Mocked as random for demonstration if file missing)
    emb_path = EMBEDDINGS_DIR / "plm_embeddings.npy"
    if emb_path.exists():
        plm_embeddings = np.load(emb_path)
        # Assume we have a mapping of gene -> index. For now, creating a mock dictionary.
        # In production, load the reference Parquet file to map gene names to embedding indices.
        gene_embeddings = {gene: plm_embeddings[i % len(plm_embeddings)] for i, gene in enumerate(adata.var_names)}
    else:
        logger.warning(f"PLM embeddings not found at {emb_path}. Using synthetic embeddings for Gamma_km.")
        gene_embeddings = {gene: np.random.rand(256) for gene in adata.var_names}

    return adata, exerkine_genes, gene_embeddings

def build_prior_network(exerkine_genes):
    """
    Section 10.2: Biological Prior (alpha_km).
    Uses a FANTOM5-based ligand-receptor table as the default prior source.
    If the table is absent locally, the workflow falls back to an empty placeholder.
    """
    logger.info("Loading FANTOM5-based ligand-receptor database...")

    fantom5_path = PROJECT_ROOT / "data" / "processed" / "ligand_receptor" / "fantom5_lri.csv"
    lr_db = load_fantom5_lri(fantom5_path)

    if lr_db.empty:
        logger.warning(
            "No FANTOM5 LRI data found at %s. Using an empty prior network.",
            fantom5_path,
        )
        return pd.DataFrame(columns=["source_genesymbol", "target_genesymbol", "source", "target", "confidence", "alpha_km"])

    exerkine_network = lr_db[lr_db['source_genesymbol'].isin(exerkine_genes)].copy()
    exerkine_network['alpha_km'] = 1.0

    logger.info(f"Filtered to {len(exerkine_network)} potential exerkine-receptor interactions.")
    return exerkine_network

def compute_molecular_compatibility(lr_network, gene_embeddings):
    """
    Section 10.1: Molecular Compatibility (Gamma_km) = g(z_lk, z_rm).
    Computes the cosine similarity between the PLM/GLM embeddings of the ligand and receptor.
    """
    logger.info("Computing Molecular Compatibility (Gamma_km) from sequence embeddings...")
    
    gamma_scores = []
    for _, row in lr_network.iterrows():
        ligand = row['source_genesymbol']
        receptor = row['target_genesymbol']
        
        if ligand in gene_embeddings and receptor in gene_embeddings:
            z_l = gene_embeddings[ligand].reshape(1, -1)
            z_r = gene_embeddings[receptor].reshape(1, -1)
            # Gamma_km based on embedding alignment
            gamma = cosine_similarity(z_l, z_r)[0][0] 
            # Normalize to [0, 1]
            gamma = (gamma + 1) / 2 
        else:
            gamma = 0.0
            
        gamma_scores.append(gamma)
        
    lr_network['Gamma_km'] = gamma_scores
    return lr_network

def compute_base_expression_potential(adata, lr_network, cluster_key="cell_type"):
    """
    Section 10 (Base): S_ij = x_i(l_k) * x_j(r_m).
    Calculates the mean expression product between sender and receiver cell types.
    """
    logger.info(f"Computing base expression potential across '{cluster_key}' clusters...")
    
    if cluster_key not in adata.obs.columns:
        logger.warning(f"Cluster key '{cluster_key}' not found. Creating generic cell types.")
        adata.obs[cluster_key] = np.random.choice(["Myocyte", "Fibroblast", "Macrophage", "Endothelial"], size=adata.n_obs)

    # Calculate mean expression per cluster
    mean_expr = pd.DataFrame(
        np.zeros((len(adata.obs[cluster_key].unique()), adata.n_vars)),
        index=adata.obs[cluster_key].unique(),
        columns=adata.var_names
    )
    
    for cluster in mean_expr.index:
        mean_expr.loc[cluster] = np.ravel(adata[adata.obs[cluster_key] == cluster].X.mean(axis=0))

    # Calculate communication potential
    interaction_records = []
    for _, row in lr_network.iterrows():
        l_k = row['source_genesymbol']
        r_m = row['target_genesymbol']
        alpha = row['alpha_km']
        gamma = row['Gamma_km']
        
        if l_k not in mean_expr.columns or r_m not in mean_expr.columns:
            continue
            
        for sender in mean_expr.index:
            for receiver in mean_expr.index:
                x_i_lk = mean_expr.loc[sender, l_k]
                x_j_rm = mean_expr.loc[receiver, r_m]
                
                # Base potential: S_ij * alpha * Gamma
                base_score = x_i_lk * x_j_rm * alpha * gamma
                
                if base_score > 0:
                    interaction_records.append({
                        "sender_cell": sender,
                        "receiver_cell": receiver,
                        "ligand_exerkine": l_k,
                        "receptor": r_m,
                        "x_i_lk": x_i_lk,
                        "x_j_rm": x_j_rm,
                        "alpha_km": alpha,
                        "Gamma_km": gamma,
                        "base_interaction_score": base_score
                    })
                    
    return pd.DataFrame(interaction_records)

def main():
    logger.info("Initializing 08_build_ligand_receptor_network workflow...")
    create_directories()

    # 1. Load Data
    adata, exerkine_genes, gene_embeddings = load_inputs()
    
    # 2. Build biological prior network (alpha_km)
    lr_network = build_prior_network(exerkine_genes)
    
    # 3. Compute molecular compatibility (Gamma_km)
    lr_network = compute_molecular_compatibility(lr_network, gene_embeddings)
    
    # 4. Compute base expression potential across cell types
    interaction_df = compute_base_expression_potential(adata, lr_network)
    
    # 5. Save outputs
    network_path = LR_DIR / "exerkine_lr_network.csv"
    interaction_df.to_csv(network_path, index=False)
    
    logger.info(f"Successfully saved LRI network with {len(interaction_df)} edges to {network_path}")
    logger.info("Workflow 08_build_ligand_receptor_network complete.")

if __name__ == "__main__":
    main()
