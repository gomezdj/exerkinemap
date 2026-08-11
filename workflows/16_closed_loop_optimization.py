"""
16_closed_loop_optimization.py

This script implements Phase 17 of the EXERKINEMAP framework.
It executes closed-loop optimization:
Design -> Predict -> Compare -> Optimize

Iteratively updates candidate sequences (A_{n+1} = Optimize(A_n, L_INV)) 
until the predicted pathway activation matches the target state (L_INV < epsilon).
"""
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "optimization"

def create_directories():
    """Ensure output directories exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def forward_model_simulation(candidate_seq):
    """
    Simulates the forward model pipeline for a given candidate sequence:
    A_hat_0 -> G_E_hat -> F_hat(t) -> A_P_hat
    Returns a mock pathway activation vector A_P_hat.
    """
    # In production, this runs scripts 08 through 11 programmatically on the mutated sequence.
    # We simulate vector responses based on sequence length and amino acid properties.
    np.random.seed(len(candidate_seq))
    simulated_activation = np.random.uniform(0.1, 1.0, size=5) # 5 core pathways (e.g., PI3K, MAPK, etc.)
    return simulated_activation

def compute_inverse_loss(a_p_hat, a_p_target):
    """
    Computes L_INV = d(A_P_hat, A_P^*) using Mean Squared Error (MSE).
    """
    return float(np.mean((a_p_hat - a_p_target) ** 2))

def optimize_sequence_loop(initial_candidate, target_a_p, max_iter=10, epsilon=0.05):
    """
    Phase 17: Closed-Loop Optimization Loop
    Repeatedly optimizes candidate until L_INV < epsilon.
    """
    logger.info(f"Starting closed-loop optimization for candidate: {initial_candidate['candidate_id']}...")
    
    current_seq = initial_candidate['sequence']
    best_loss = float('inf')
    optimization_history = []
    
    for iteration in range(1, max_iter + 1):
        # 1. Predict (Run Forward Model)
        a_p_hat = forward_model_simulation(current_seq)
        
        # 2. Compare (Compute L_INV loss against target state A_P^*)
        loss = compute_inverse_loss(a_p_hat, target_a_p)
        
        logger.info(f"Iteration {iteration}/{max_iter} | Loss L_INV: {loss:.4f} (Target epsilon: {epsilon})")
        
        optimization_history.append({
            "iteration": iteration,
            "sequence": current_seq,
            "loss": loss
        })
        
        if loss < best_loss:
            best_loss = loss
            
        # 3. Check convergence condition
        if loss < epsilon:
            logger.info(f"Convergence reached! L_INV ({loss:.4f}) < epsilon ({epsilon}).")
            break
            
        # 4. Optimize (Simulate directed mutation / gradient-based token adjustment in sequence space)
        # In a full deployment, this mutates weak residues to minimize L_INV gradients.
        mutation_index = np.random.randint(5, len(current_seq))
        amino_acids = "VLIFAWMCQEDRKHSTY"
        mutated_aa = amino_acids[np.random.randint(0, len(amino_acids))]
        current_seq = current_seq[:mutation_index] + mutated_aa + current_seq[mutation_index+1:]
        
    return {
        "candidate_id": initial_candidate['candidate_id'],
        "final_sequence": current_seq,
        "final_loss": best_loss,
        "converged": best_loss < epsilon,
        "history": optimization_history
    }

def main():
    logger.info("Initializing 16_closed_loop_optimization workflow...")
    create_directories()

    # 1. Define desired target pathway activation state (A_P^*) e.g., high exercise-induced signaling
    target_a_p = np.array([0.95, 0.90, 0.85, 0.92, 0.88])
    
    # 2. Load generated candidate from Phase 16
    candidates_path = PROJECT_ROOT / "results" / "sequences" / "generated_and_filtered_candidates.csv"
    if candidates_path.exists():
        candidates_df = pd.read_csv(candidates_path)
    else:
        logger.warning("Filtered candidates file not found. Creating a synthetic baseline candidate.")
        candidates_df = pd.DataFrame([{
            "candidate_id": "CAND_001",
            "sequence": "MKWVTFISLLFLFSSAYSRVGLPRVLLAALLGAAALAPG"
        }])
        
    # 3. Execute Closed-Loop Optimization on top candidates
    optimization_results = []
    for _, candidate in candidates_df.head(3).iterrows():
        result = optimize_sequence_loop(candidate, target_a_p, max_iter=8, epsilon=0.08)
        optimization_results.append(result)
        
    # 4. Save Optimization Summary
    summary_df = pd.DataFrame([{
        "candidate_id": res["candidate_id"],
        "final_sequence": res["final_sequence"],
        "final_loss": res["final_loss"],
        "converged": res["converged"]
    } for res in optimization_results])
    
    output_path = RESULTS_DIR / "optimized_candidates_summary.csv"
    summary_df.to_csv(output_path, index=False)
    
    logger.info(f"Closed-Loop Optimization complete. Summary saved to {output_path}")
    logger.info(f"\n{summary_df.to_string(index=False)}")

if __name__ == "__main__":
    main()
