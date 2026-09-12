import pytest
import numpy as np
import scipy.sparse as sp
from workflows.propagation import SignalPropagator

@pytest.fixture
def mock_adjacency_matrix():
    # Create a simple 4x4 undirected graph
    A = np.array([
        [0, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 0],
        [0, 1, 0, 0]
    ])
    return sp.csr_matrix(A)

def test_transition_matrix_normalization(mock_adjacency_matrix):
    propagator = SignalPropagator(alpha=0.7)
    T = propagator._normalize_adjacency(mock_adjacency_matrix)
    
    # Columns of transition matrix should sum to 1
    col_sums = np.array(T.sum(axis=0)).flatten()
    np.testing.assert_allclose(col_sums, np.ones(4), rtol=1e-5)

def test_random_walk_convergence(mock_adjacency_matrix):
    propagator = SignalPropagator(alpha=0.7)
    seeds = np.array([1.0, 0.0, 0.0, 0.0]) # Signal starts entirely at node 0
    
    steady_state = propagator.run_rwr(adjacency=mock_adjacency_matrix, seeds=seeds, max_iter=100, tol=1e-6)
    
    assert steady_state.shape == (4,)
    # Total signal should be conserved
    np.testing.assert_allclose(np.sum(steady_state), 1.0, rtol=1e-5)
    # Node 0 should retain highest signal due to restart probability
    assert np.argmax(steady_state) == 0