from scipy.linalg import expm

def compute_diffused_signal(adjacency, f_0, t=1.0):
    """
    Compute F(t) = e^(-tL) · f_0
    
    Parameters:
    -----------
    adjacency : np.array, shape (N, N)
    f_0 : np.array, shape (N,)
        Initial exerkine secretion profile
    t : float
        Diffusion time parameter
    
    Returns:
    --------
    F_t : np.array, shape (N,)
        Signal distribution after diffusion
    """
    # Compute graph Laplacian
    D = np.diag(adjacency.sum(axis=1))
    L = D - adjacency
    
    # Matrix exponential
    F_t = expm(-t * L) @ f_0
    
    return F_t
