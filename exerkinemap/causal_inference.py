"""
Layer 4: Causal Inference — do-Calculus & Structural Causal Models
====================================================================
C_ij = E[Y_i | do(F_j)] - E[Y_i]

Pearl do-calculus framework for inferring causal influence of exerkine
signals across cells and tissues. Supports counterfactual simulation of
exercise interventions via Bayesian networks and structural equations.

Reference (JSX):  2.2.6 Causal Inference  |  Tag: SCM · do-Calculus · Counterfactual
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Structural Causal Model
# ---------------------------------------------------------------------------

class StructuralCausalModel:
    """
    Structural Causal Model (SCM) for exerkine signaling networks.

    Implements Pearl's do-calculus to estimate causal effects::

        C_ij = E[Y_i | do(F_j)] - E[Y_i]

    where ``F_j`` is an exerkine signal (do-intervened on cell *j*),
    ``Y_i`` is the downstream response in cell *i*.

    Parameters
    ----------
    n_cells : int
        Number of cells (nodes) in the signaling graph.
    n_exerkines : int
        Number of exerkine features per cell.
    """

    def __init__(self, n_cells: int, n_exerkines: int) -> None:
        self.n_cells = n_cells
        self.n_exerkines = n_exerkines
        self.adjacency: Optional[np.ndarray] = None
        self.F: Optional[np.ndarray] = None
        self.Y: Optional[np.ndarray] = None
        self.structural_weights: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def fit(
        self,
        F: np.ndarray,
        Y: np.ndarray,
        adjacency: np.ndarray,
    ) -> "StructuralCausalModel":
        """
        Fit structural equations from observational data via OLS.

        Parameters
        ----------
        F : np.ndarray, shape (N, E)
            Exerkine flux matrix (cells × exerkines).
        Y : np.ndarray, shape (N, K)
            Response matrix (cells × outcomes).
        adjacency : np.ndarray, shape (N, N)
            Weighted cell–cell adjacency (parent → child).

        Returns
        -------
        self
        """
        N, E = F.shape
        if Y.ndim == 1:
            Y = Y[:, np.newaxis]
        K = Y.shape[1]

        self.adjacency = adjacency
        self.F = F
        self.Y = Y
        self.structural_weights = np.zeros((N, E, K))

        for i in range(N):
            parents = np.where(adjacency[:, i] > 0)[0]
            if len(parents) == 0:
                continue
            X = F[parents]  # (n_parents, E)
            try:
                # Fit Y_i ~ linear(F_parents)
                X_flat = X.reshape(len(parents), E)
                coef, *_ = np.linalg.lstsq(X_flat, np.tile(Y[i], (len(parents), 1)), rcond=None)
                self.structural_weights[i, : min(E, len(coef)), :] = coef[: min(E, len(coef))]
            except np.linalg.LinAlgError:
                warnings.warn(f"LSQ failed for cell {i}; skipping.", RuntimeWarning)

        return self

    # ------------------------------------------------------------------
    def do_calculus(
        self,
        F: np.ndarray,
        intervention_cell: int,
        intervention_value: float,
        outcome_cells: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Apply the do-operator and return per-cell causal effects.

        Implements::

            C_ij = E[Y_i | do(F_j = v)] - E[Y_i]

        The graph is *mutilated* at node *j* (all incoming edges removed),
        then the intervention propagates downstream via the adjacency.

        Parameters
        ----------
        F : np.ndarray, shape (N, E)
            Current exerkine flux.
        intervention_cell : int
            Cell *j* where ``do(F_j)`` is applied.
        intervention_value : float
            Scalar value to set for the intervened cell's mean flux.
        outcome_cells : list of int, optional
            Cells *i* to evaluate (default: all cells).

        Returns
        -------
        C : np.ndarray, shape (len(outcome_cells),)
            Causal effect per outcome cell.
        """
        if outcome_cells is None:
            outcome_cells = list(range(self.n_cells))

        # Mutilate: sever all incoming edges to j, set F_j = intervention
        F_do = F.copy()
        F_do[intervention_cell, :] = intervention_value

        # Propagate 1-hop downstream
        if self.adjacency is not None:
            children = np.where(self.adjacency[intervention_cell, :] > 0)[0]
            delta = intervention_value - np.mean(F[intervention_cell, :])
            for child in children:
                w = self.adjacency[intervention_cell, child]
                F_do[child, :] += w * delta

        # Causal effect: interventional mean − observational mean per cell
        E_Y_do = np.mean(F_do[outcome_cells, :], axis=1)
        E_Y_obs = np.mean(F[outcome_cells, :], axis=1)
        return E_Y_do - E_Y_obs

    # ------------------------------------------------------------------
    def counterfactual(
        self,
        F_factual: np.ndarray,
        exercise_protocol: Dict[int, float],
    ) -> np.ndarray:
        """
        Counterfactual query: "What if the exercise protocol had been different?"

        Abduction → Action → Prediction (Pearl's three-step algorithm).

        Parameters
        ----------
        F_factual : np.ndarray, shape (N, E)
            Observed exerkine flux under the factual exercise condition.
        exercise_protocol : dict {exerkine_index: value}
            Hypothetical exerkine concentrations to set globally.

        Returns
        -------
        F_cf : np.ndarray, shape (N, E)
            Counterfactual flux under the hypothetical protocol.
        """
        F_cf = F_factual.copy()

        # Action: fix exerkine columns
        for exerkine_idx, value in exercise_protocol.items():
            F_cf[:, int(exerkine_idx)] = value

        # Prediction: propagate through structural equations (1-step)
        if self.adjacency is not None:
            for i in range(self.n_cells):
                parents = np.where(self.adjacency[:, i] > 0)[0]
                if len(parents) == 0:
                    continue
                parent_delta = F_cf[parents] - F_factual[parents]
                propagated = np.sum(
                    self.adjacency[parents, i:i+1] * parent_delta, axis=0
                )
                F_cf[i] += 0.5 * propagated  # partial propagation coefficient

        return F_cf

    # ------------------------------------------------------------------
    def compute_causal_matrix(
        self,
        F: np.ndarray,
        intervention_values: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute the full N×N causal influence matrix.

        ``C[i, j]`` = causal effect on cell *i* from intervening on cell *j*.

        Parameters
        ----------
        F : np.ndarray, shape (N, E)
        intervention_values : np.ndarray, shape (N,), optional
            Per-cell intervention level (default: L2 norm of each row).

        Returns
        -------
        C_matrix : np.ndarray, shape (N, N)
        """
        N = F.shape[0]
        if intervention_values is None:
            intervention_values = np.linalg.norm(F, axis=1)

        C_matrix = np.zeros((N, N))
        for j in range(N):
            C_matrix[:, j] = self.do_calculus(
                F,
                intervention_cell=j,
                intervention_value=float(intervention_values[j]),
            )
        return C_matrix


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------

def compute_causal_scores(
    F: np.ndarray,
    adjacency: np.ndarray,
    Y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit an SCM and return the causal influence matrix + top upstream cells.

    Parameters
    ----------
    F : np.ndarray, shape (N, E)   — exerkine flux
    adjacency : np.ndarray, (N, N) — cell graph
    Y : np.ndarray, (N, K)         — outcome responses

    Returns
    -------
    C_matrix : np.ndarray, shape (N, N)
    top_interventions : np.ndarray, shape (N,)
        For each cell *i*, the index of its most causally influential upstream cell.
    """
    N, E = F.shape
    scm = StructuralCausalModel(n_cells=N, n_exerkines=E)
    scm.fit(F, Y, adjacency)
    C_matrix = scm.compute_causal_matrix(F)
    top_interventions = np.argmax(np.abs(C_matrix), axis=1)
    return C_matrix, top_interventions


def exercise_intervention_effect(
    F: np.ndarray,
    adjacency: np.ndarray,
    Y: np.ndarray,
    exercise_protocols: List[Dict[int, float]],
) -> List[np.ndarray]:
    """
    Simulate counterfactual effects of multiple exercise protocols.

    Parameters
    ----------
    F : np.ndarray, shape (N, E)
    adjacency : np.ndarray, shape (N, N)
    Y : np.ndarray, shape (N, K)
    exercise_protocols : list of dicts {exerkine_idx: value}

    Returns
    -------
    counterfactuals : list of np.ndarray, each shape (N, E)
    """
    N, E = F.shape
    scm = StructuralCausalModel(n_cells=N, n_exerkines=E)
    scm.fit(F, Y, adjacency)
    return [scm.counterfactual(F, protocol) for protocol in exercise_protocols]
