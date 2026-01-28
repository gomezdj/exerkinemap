"""
Cell-level Ligand-Receptor Interaction (LRI) Score Computation
Implements the mathematical framework from EXERKINEMAP
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.sparse import csr_matrix
import warnings


def compute_initial_exerkine_profile(
    G_expr: np.ndarray,
    adjacency: np.ndarray,
    exerkines: List[str],
    gene_names: List[str],
    edge_weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute initial exerkine secretion profile f_0(i)
    
    Formula:
    f_0(i) = (x_i^exerkine + Σ_{j∈N_i} w_ij * x_j^exerkine) / (1 + Σ_{j∈N_i} w_ij)
    
    where x_i^exerkine = Σ_{l_k∈E} x_i(l_k)
    
    Parameters:
    -----------
    G_expr : np.ndarray, shape (N, G)
        Gene expression matrix (cells × genes)
    adjacency : np.ndarray, shape (N, N)
        Binary adjacency matrix A_ij from k-NN or radius graph
    exerkines : list of str
        Exerkine ligand gene names (E ⊂ L)
    gene_names : list of str
        All gene names corresponding to G_expr columns
    edge_weights : np.ndarray, shape (N, N), optional
        Edge weights w_ij (e.g., 1/(1 + d_ij))
        If None, uses binary adjacency
    
    Returns:
    --------
    f_0 : np.ndarray, shape (N,)
        Initial exerkine secretion profile for each cell
    """
    N = G_expr.shape[0]
    
    # Get exerkine gene indices
    gene_name_to_idx = {name: idx for idx, name in enumerate(gene_names)}
    exerkine_indices = [gene_name_to_idx[lig] for lig in exerkines if lig in gene_name_to_idx]
    
    if len(exerkine_indices) == 0:
        warnings.warn("No exerkine genes found in expression matrix")
        return np.zeros(N)
    
    # Compute x_i^exerkine = sum of exerkine expression
    x_exerkine = G_expr[:, exerkine_indices].sum(axis=1)  # Shape: (N,)
    
    # Use edge weights if provided, otherwise binary adjacency
    if edge_weights is None:
        edge_weights = adjacency.copy()
    
    # Compute neighbor contribution: Σ_{j∈N_i} w_ij * x_j^exerkine
    neighbor_contrib = edge_weights @ x_exerkine  # Matrix-vector product
    
    # Compute weight normalization: 1 + Σ_{j∈N_i} w_ij
    weight_sum = 1.0 + edge_weights.sum(axis=1)  # Shape: (N,)
    
    # Compute f_0(i)
    f_0 = (x_exerkine + neighbor_contrib) / weight_sum
    
    return f_0


def compute_lri_interaction_score(
    G_expr: np.ndarray,
    ligand_idx: int,
    receptor_idx: int,
    sender_cell_idx: int,
    receiver_cell_idx: int
) -> float:
    """
    Compute base LRI interaction score
    
    Formula:
    S_ij^(l_k, r_m) = x_i(l_k) · x_j(r_m)
    
    Parameters:
    -----------
    G_expr : np.ndarray
        Gene expression matrix
    ligand_idx : int
        Index of ligand gene in G_expr
    receptor_idx : int
        Index of receptor gene in G_expr
    sender_cell_idx : int
        Index of sender cell i
    receiver_cell_idx : int
        Index of receiver cell j
    
    Returns:
    --------
    score : float
        Interaction score
    """
    x_i_lk = G_expr[sender_cell_idx, ligand_idx]
    x_j_rm = G_expr[receiver_cell_idx, receptor_idx]
    
    return x_i_lk * x_j_rm


def compute_cell_lri_scores(
    G_expr: np.ndarray,
    ligands: List[str],
    receptors: List[str],
    gene_names: List[str],
    adjacency: np.ndarray,
    exerkines: Optional[List[str]] = None,
    beta_pathway: Optional[Dict[str, float]] = None,
    expression_threshold: float = 0.0
) -> np.ndarray:
    """
    Compute comprehensive cell-level LRI scores
    
    Implements:
    P_activation(c_i | c_j) = Σ_{(l_k, r_m)} δ_{l_k∈E} · S_ij^(l_k, r_m) · β_{r_m}
    
    Parameters:
    -----------
    G_expr : np.ndarray, shape (N, G)
        Gene expression matrix
    ligands : list of str
        Ligand gene names
    receptors : list of str
        Receptor gene names
    gene_names : list of str
        All gene names in G_expr
    adjacency : np.ndarray, shape (N, N)
        Cell-cell adjacency matrix
    exerkines : list of str, optional
        Exerkine subset (for δ indicator)
        If None, all ligands are considered exerkines
    beta_pathway : dict, optional
        Receptor -> pathway impact score (β_{r_m})
        If None, uses uniform weight of 1.0
    expression_threshold : float
        Minimum expression level to consider (filters noise)
    
    Returns:
    --------
    cell_scores : np.ndarray, shape (N,)
        Per-cell composite LRI activation score
    """
    N = G_expr.shape[0]
    cell_scores = np.zeros(N)
    
    # Create gene name to index mapping
    gene_name_to_idx = {name: idx for idx, name in enumerate(gene_names)}
    
    # Get ligand and receptor indices
    ligand_indices = {lig: gene_name_to_idx[lig] for lig in ligands if lig in gene_name_to_idx}
    receptor_indices = {rec: gene_name_to_idx[rec] for rec in receptors if rec in gene_name_to_idx}
    
    if len(ligand_indices) == 0 or len(receptor_indices) == 0:
        warnings.warn("No valid ligand-receptor pairs found")
        return cell_scores
    
    # Set exerkines (δ indicator)
    if exerkines is None:
        exerkines = set(ligands)
    else:
        exerkines = set(exerkines)
    
    # Set pathway impact scores (β)
    if beta_pathway is None:
        beta_pathway = {rec: 1.0 for rec in receptors}
    
    # Compute scores for each cell
    for i in range(N):
        for j in range(N):
            # Only consider connected cells
            if adjacency[i, j] == 0:
                continue
            
            for lig_name, lig_idx in ligand_indices.items():
                # Check expression threshold
                if G_expr[i, lig_idx] < expression_threshold:
                    continue
                
                # δ indicator: is this ligand an exerkine?
                delta = 1.0 if lig_name in exerkines else 0.0
                
                for rec_name, rec_idx in receptor_indices.items():
                    # Check expression threshold
                    if G_expr[j, rec_idx] < expression_threshold:
                        continue
                    
                    # Compute S_ij^(l_k, r_m)
                    S_ij = G_expr[i, lig_idx] * G_expr[j, rec_idx]
                    
                    # Get pathway impact
                    beta = beta_pathway.get(rec_name, 1.0)
                    
                    # Accumulate score
                    cell_scores[i] += delta * S_ij * beta * adjacency[i, j]
    
    return cell_scores


def compute_pairwise_lri_matrix(
    G_expr: np.ndarray,
    ligands: List[str],
    receptors: List[str],
    gene_names: List[str],
    lr_pairs: Optional[List[Tuple[str, str]]] = None,
    expression_threshold: float = 0.0
) -> np.ndarray:
    """
    Compute full pairwise LRI score matrix
    
    Parameters:
    -----------
    G_expr : np.ndarray, shape (N, G)
    ligands : list of str
    receptors : list of str
    gene_names : list of str
    lr_pairs : list of (str, str), optional
        Specific ligand-receptor pairs to consider
        If None, considers all ligand × receptor combinations
    expression_threshold : float
    
    Returns:
    --------
    lri_matrix : np.ndarray, shape (N, N)
        LRI scores between all cell pairs (summed across all L-R pairs)
    """
    N = G_expr.shape[0]
    lri_matrix = np.zeros((N, N))
    
    gene_name_to_idx = {name: idx for idx, name in enumerate(gene_names)}
    
    # Define L-R pairs
    if lr_pairs is None:
        ligand_indices = {lig: gene_name_to_idx[lig] for lig in ligands if lig in gene_name_to_idx}
        receptor_indices = {rec: gene_name_to_idx[rec] for rec in receptors if rec in gene_name_to_idx}
        lr_pairs = [(lig, rec) for lig in ligand_indices.keys() for rec in receptor_indices.keys()]
    
    # Compute scores
    for lig_name, rec_name in lr_pairs:
        if lig_name not in gene_name_to_idx or rec_name not in gene_name_to_idx:
            continue
        
        lig_idx = gene_name_to_idx[lig_name]
        rec_idx = gene_name_to_idx[rec_name]
        
        # Get expression vectors
        lig_expr = G_expr[:, lig_idx]  # Shape: (N,)
        rec_expr = G_expr[:, rec_idx]  # Shape: (N,)
        
        # Filter by threshold
        lig_expr = np.where(lig_expr >= expression_threshold, lig_expr, 0)
        rec_expr = np.where(rec_expr >= expression_threshold, rec_expr, 0)
        
        # Outer product: S_ij = x_i(l_k) * x_j(r_m)
        lri_matrix += np.outer(lig_expr, rec_expr)
    
    return lri_matrix


def compute_receptor_activation_scores(
    G_expr: np.ndarray,
    receptors: List[str],
    gene_names: List[str],
    lri_matrix: np.ndarray,
    beta_pathway: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """
    Compute per-cell receptor activation scores
    
    Parameters:
    -----------
    G_expr : np.ndarray, shape (N, G)
    receptors : list of str
    gene_names : list of str
    lri_matrix : np.ndarray, shape (N, N)
        LRI interaction matrix
    beta_pathway : dict, optional
        Receptor -> pathway impact
    
    Returns:
    --------
    activation_scores : np.ndarray, shape (N,)
        Per-cell receptor activation (incoming signals)
    """
    N = G_expr.shape[0]
    activation_scores = np.zeros(N)
    
    gene_name_to_idx = {name: idx for idx, name in enumerate(gene_names)}
    
    if beta_pathway is None:
        beta_pathway = {rec: 1.0 for rec in receptors}
    
    for rec_name in receptors:
        if rec_name not in gene_name_to_idx:
            continue
        
        rec_idx = gene_name_to_idx[rec_name]
        rec_expr = G_expr[:, rec_idx]  # Shape: (N,)
        
        # Incoming LRI signals (column sum)
        incoming_signals = lri_matrix.sum(axis=0)  # Shape: (N,)
        
        # Weight by receptor expression and pathway impact
        beta = beta_pathway.get(rec_name, 1.0)
        activation_scores += rec_expr * incoming_signals * beta
    
    return activation_scores


def compute_spatial_distance_kernel(
    spatial_coords: np.ndarray,
    sigma: float = 10.0
) -> np.ndarray:
    """
    Compute spatial distance kernel D_ij
    
    Formula:
    D_ij = exp(-||s_i - s_j||^2 / 2σ^2)
    
    Parameters:
    -----------
    spatial_coords : np.ndarray, shape (N, 2)
        Spatial coordinates s_i = (x_i, y_i)
    sigma : float
        Bandwidth parameter
    
    Returns:
    --------
    D : np.ndarray, shape (N, N)
        Distance kernel matrix
    """
    from scipy.spatial.distance import cdist
    
    # Compute pairwise distances
    distances = cdist(spatial_coords, spatial_coords, metric='euclidean')
    
    # Apply Gaussian kernel
    D = np.exp(-distances**2 / (2 * sigma**2))
    
    return D


def apply_spatial_weighting(
    lri_matrix: np.ndarray,
    spatial_coords: np.ndarray,
    sigma: float = 10.0
) -> np.ndarray:
    """
    Apply spatial weighting to LRI matrix
    
    Formula:
    ω_ij^spatial = D_ij · ω_ij
    
    Parameters:
    -----------
    lri_matrix : np.ndarray, shape (N, N)
    spatial_coords : np.ndarray, shape (N, 2)
    sigma : float
    
    Returns:
    --------
    weighted_lri : np.ndarray, shape (N, N)
    """
    D = compute_spatial_distance_kernel(spatial_coords, sigma)
    return lri_matrix * D
