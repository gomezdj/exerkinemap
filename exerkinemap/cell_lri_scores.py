def compute_cell_lri_scores(G_expr, ligands, receptors, adjacency, beta_pathway):
    """
    Compute per-cell LRI scores from your framework
    
    Parameters:
    -----------
    G_expr : np.array, shape (N, G)
        Gene expression matrix
    ligands : list of str
        Ligand gene names (subset of E ⊂ L)
    receptors : list of str
        Receptor gene names
    adjacency : np.array, shape (N, N)
        A_ij from k-NN or radius graph
    beta_pathway : dict
        Receptor -> pathway impact score
    
    Returns:
    --------
    cell_scores : np.array, shape (N,)
        Per-cell composite LRI score
    """
    N = G_expr.shape[0]
    cell_scores = np.zeros(N)
    
    for i in range(N):
        for j in range(N):
            if adjacency[i, j] == 0:
                continue
            
            for l_k in ligands:
                for r_m in receptors:
                    # S_ij^(l_k, r_m) = x_i(l_k) * x_j(r_m)
                    S_ij = G_expr[i, l_k] * G_expr[j, r_m]
                    
                    # P_activation
                    delta = 1 if l_k in exerkines else 0  # indicator
                    beta = beta_pathway.get(r_m, 1.0)
                    
                    cell_scores[i] += delta * S_ij * beta * adjacency[i, j]
    
    return cell_scores
