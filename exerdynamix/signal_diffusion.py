"""
Exerkine Signal Diffusion via Graph Laplacian
Implements: F(t) = e^(-tL) · f_0
"""

import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
from typing import Optional, List, Tuple
import warnings


def compute_graph_laplacian(
    adjacency: np.ndarray,
    normalized: bool = True
) -> np.ndarray:
    """
    Compute graph Laplacian L from adjacency matrix
    
    For normalized Laplacian:
    L = I - D^(-1/2) A D^(-1/2)
    
    For unnormalized Laplacian:
    L = D - A
    
    Parameters:
    -----------
    adjacency : np.ndarray, shape (N, N)
        Adjacency matrix with edge weights
    normalized : bool
        Whether to compute normalized Laplacian
    
    Returns:
    --------
    L : np.ndarray, shape (N, N)
        Graph Laplacian
    """
    N = adjacency.shape[0]
    
    # Compute degree matrix
    degrees = adjacency.sum(axis=1)
    
    # Handle isolated nodes (degree = 0)
    degrees = np.where(degrees == 0, 1, degrees)
    
    if normalized:
        # Normalized Laplacian
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
        L = np.eye(N) - D_inv_sqrt @ adjacency @ D_inv_sqrt
    else:
        # Unnormalized Laplacian
        D = np.diag(degrees)
        L = D - adjacency
    
    return L


def compute_diffused_signal(
    adjacency: np.ndarray,
    f_0: np.ndarray,
    t: float = 1.0,
    normalized_laplacian: bool = True,
    sparse: bool = False
) -> np.ndarray:
    """
    Compute signal diffusion F(t) = e^(-tL) · f_0
    
    Models how exerkines propagate through the cell-cell network:
    - Autocrine: self-signaling (diagonal of adjacency)
    - Paracrine: local neighbor signaling
    - Endocrine: long-range via graph connectivity
    
    Parameters:
    -----------
    adjacency : np.ndarray, shape (N, N)
        Cell-cell interaction graph
    f_0 : np.ndarray, shape (N,)
        Initial exerkine secretion profile
    t : float
        Diffusion time parameter
        - Small t (~0.1-1): local diffusion
        - Large t (~5-10): global diffusion
    normalized_laplacian : bool
        Use normalized Laplacian (recommended for heterogeneous networks)
    sparse : bool
        Use sparse matrix operations (for large N)
    
    Returns:
    --------
    F_t : np.ndarray, shape (N,)
        Signal distribution after diffusion
    """
    if f_0.shape[0] != adjacency.shape[0]:
        raise ValueError("f_0 must have same length as adjacency dimension")
    
    # Compute Laplacian
    L = compute_graph_laplacian(adjacency, normalized=normalized_laplacian)
    
    # Matrix exponential
    if sparse:
        L_sparse = csr_matrix(L)
        F_t = expm_multiply(-t * L_sparse, f_0)
    else:
        exp_L = expm(-t * L)
        F_t = exp_L @ f_0
    
    return F_t


def compute_diffusion_time_series(
    adjacency: np.ndarray,
    f_0: np.ndarray,
    time_points: List[float]
) -> np.ndarray:
    """
    Compute signal diffusion across multiple timepoints
    
    Parameters:
    -----------
    adjacency : np.ndarray, shape (N, N)
    f_0 : np.ndarray, shape (N,)
    time_points : list of float
        Diffusion times to evaluate
    
    Returns:
    --------
    F_time_series : np.ndarray, shape (N, T)
        Signal distribution at each timepoint
    """
    N = f_0.shape[0]
    T = len(time_points)
    F_time_series = np.zeros((N, T))
    
    for idx, t in enumerate(time_points):
        F_time_series[:, idx] = compute_diffused_signal(adjacency, f_0, t=t)
    
    return F_time_series


def compute_steady_state_signal(
    adjacency: np.ndarray,
    f_0: np.ndarray,
    tolerance: float = 1e-6,
    max_time: float = 100.0
) -> Tuple[np.ndarray, float]:
    """
    Compute steady-state signal distribution
    
    Parameters:
    -----------
    adjacency : np.ndarray, shape (N, N)
    f_0 : np.ndarray, shape (N,)
    tolerance : float
        Convergence threshold
    max_time : float
        Maximum diffusion time to check
    
    Returns:
    --------
    F_steady : np.ndarray, shape (N,)
        Steady-state distribution
    converge_time : float
        Time at which steady state was reached
    """
    L = compute_graph_laplacian(adjacency, normalized=True)
    
    # Check for convergence at increasing time points
    time_points = np.logspace(-1, np.log10(max_time), 50)
    
    F_prev = f_0.copy()
    for t in time_points:
        F_t = compute_diffused_signal(adjacency, f_0, t=t)
        
        # Check convergence
        if np.linalg.norm(F_t - F_prev) < tolerance:
            return F_t, t
        
        F_prev = F_t
    
    warnings.warn(f"Did not converge to steady state within t={max_time}")
    return F_prev, max_time


def compute_signal_propagation_distance(
    adjacency: np.ndarray,
    f_0: np.ndarray,
    source_cells: List[int],
    t: float = 1.0
) -> np.ndarray:
    """
    Measure how far signals propagate from source cells
    
    Parameters:
    -----------
    adjacency : np.ndarray, shape (N, N)
    f_0 : np.ndarray, shape (N,)
    source_cells : list of int
        Indices of source cells
    t : float
        Diffusion time
    
    Returns:
    --------
    propagation : np.ndarray, shape (N,)
        Signal strength at each cell originating from sources
    """
    # Initialize signal only at source cells
    f_source = np.zeros_like(f_0)
    f_source[source_cells] = f_0[source_cells]
    
    # Diffuse
    F_t = compute_diffused_signal(adjacency, f_source, t=t)
    
    return F_t


def compute_effective_diffusion_coefficient(
    adjacency: np.ndarray,
    spatial_coords: np.ndarray
) -> float:
    """
    Estimate effective diffusion coefficient from graph structure
    
    Parameters:
    -----------
    adjacency : np.ndarray, shape (N, N)
    spatial_coords : np.ndarray, shape (N, 2)
    
    Returns:
    --------
    D_eff : float
        Effective diffusion coefficient
    """
    from scipy.spatial.distance import cdist
    
    # Compute average edge length
    edge_lengths = []
    for i in range(adjacency.shape[0]):
        neighbors = np.where(adjacency[i, :] > 0)[0]
        if len(neighbors) > 0:
            dists = cdist([spatial_coords[i]], spatial_coords[neighbors])[0]
            edge_lengths.extend(dists)
    
    if len(edge_lengths) == 0:
        return 0.0
    
    avg_edge_length = np.mean(edge_lengths)
    
    # D_eff ≈ (average edge length)^2 / (2 * average degree)
    avg_degree = adjacency.sum(axis=1).mean()
    D_eff = avg_edge_length**2 / (2 * avg_degree) if avg_degree > 0 else 0.0
    
    return D_eff


def analyze_signal_propagation_modes(
    adjacency: np.ndarray,
    f_0: np.ndarray,
    n_modes: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analyze signal propagation via Laplacian eigenmodes
    
    Parameters:
    -----------
    adjacency : np.ndarray, shape (N, N)
    f_0 : np.ndarray, shape (N,)
    n_modes : int
        Number of eigenmodes to compute
    
    Returns:
    --------
    eigenvalues : np.ndarray, shape (n_modes,)
        Laplacian eigenvalues (decay rates)
    eigenvectors : np.ndarray, shape (N, n_modes)
        Spatial modes of diffusion
    """
    L = compute_graph_laplacian(adjacency, normalized=True)
    
    # Compute eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    # Sort by eigenvalue (smallest first)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx][:n_modes]
    eigenvectors = eigenvectors[:, idx][:, :n_modes]
    
    return eigenvalues, eigenvectors
