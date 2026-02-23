"""
Layer 7: Generative Feedback — Biomolecule Design Loop
=======================================================
Objective::

    θ* = argmax_θ  U( G_ExerkineMap(θ) )

Workflow:
    1. **Gap detection** — identify exerkines/receptors absent or under-expressed
       in the current signaling network.
    2. **Candidate proposal** — sample candidate biomolecule parameters θ
       (sequence motifs, binding affinities, secretion rates).
    3. **Simulation** — run the ExerkineMap forward model G(θ) to predict
       the resulting signaling network.
    4. **Optimisation** — maximise therapeutic utility U via gradient ascent
       or black-box evolution (CMA-ES / Bayesian optimisation).

This module provides a numpy-/scipy-based implementation that can be wrapped
around a pre-trained GNN simulator or an analytical signal diffusion model.

Reference (JSX):  2.2.7 Generative Feedback  |  Tag: Diffusion · Transformer · ProteinDesign
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize, differential_evolution


# ---------------------------------------------------------------------------
# Gap Detection
# ---------------------------------------------------------------------------

def detect_signaling_gaps(
    F: np.ndarray,
    lri_scores: np.ndarray,
    threshold_flux: float = 0.1,
    threshold_lri: float = 0.05,
) -> Dict[str, np.ndarray]:
    """
    Identify cells and exerkines with insufficient signaling activity.

    Parameters
    ----------
    F : np.ndarray, shape (N, E)
        Exerkine flux matrix.
    lri_scores : np.ndarray, shape (N, N)
        Pairwise ligand–receptor interaction scores.
    threshold_flux : float
        Minimum acceptable mean flux per exerkine.
    threshold_lri : float
        Minimum acceptable mean LRI score per cell pair.

    Returns
    -------
    gaps : dict with keys "silent_exerkines", "weak_pairs", "gap_cells"
    """
    mean_flux = np.mean(F, axis=0)  # (E,)
    silent_exerkines = np.where(mean_flux < threshold_flux)[0]

    mean_lri = np.mean(lri_scores, axis=1)  # (N,) sender score
    gap_cells = np.where(mean_lri < threshold_lri)[0]

    weak_pairs = np.argwhere(lri_scores < threshold_lri)

    return {
        "silent_exerkines": silent_exerkines,
        "weak_pairs": weak_pairs,
        "gap_cells": gap_cells,
        "mean_flux": mean_flux,
        "mean_lri": mean_lri,
    }


# ---------------------------------------------------------------------------
# Biomolecule Parameter Space
# ---------------------------------------------------------------------------

class BiomoleculeDesignSpace:
    """
    Defines the parameter space θ for a candidate exerkine/biomolecule.

    Parameters θ include:
        - ``secretion_rate`` : rate of secretion per cell (per minute)
        - ``binding_affinity`` : log-affinity to cognate receptor (log Kd)
        - ``stability``        : half-life in circulation (minutes)
        - ``target_exerkines`` : which flux channels the molecule modulates

    Parameters
    ----------
    n_exerkines : int
        Number of exerkine channels in the model.
    n_params : int
        Total dimensionality of θ (default: 3 + n_exerkines).
    bounds : list of (lo, hi), optional
        Parameter bounds for optimisation.
    """

    def __init__(
        self,
        n_exerkines: int,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        self.n_exerkines = n_exerkines
        # θ = [secretion_rate, binding_affinity, stability, *exerkine_weights]
        self.n_params = 3 + n_exerkines
        if bounds is None:
            self.bounds = (
                [(0.0, 10.0)]   # secretion_rate
                + [(-10.0, 0.0)]  # log binding affinity (higher = tighter)
                + [(1.0, 120.0)]  # stability (minutes)
                + [(-1.0, 1.0)] * n_exerkines  # exerkine modulation weights
            )
        else:
            self.bounds = bounds

    def sample(self, n: int = 1, seed: Optional[int] = None) -> np.ndarray:
        """
        Sample *n* random biomolecule parameter vectors from the design space.

        Returns
        -------
        theta : np.ndarray, shape (n, n_params)
        """
        rng = np.random.default_rng(seed)
        lo = np.array([b[0] for b in self.bounds])
        hi = np.array([b[1] for b in self.bounds])
        return rng.uniform(lo, hi, size=(n, self.n_params))

    def decode(self, theta: np.ndarray) -> Dict[str, float | np.ndarray]:
        """Decode a flat θ vector into named biomolecule properties."""
        return {
            "secretion_rate": float(theta[0]),
            "binding_affinity_log_kd": float(theta[1]),
            "stability_min": float(theta[2]),
            "exerkine_modulation": theta[3:],
        }


# ---------------------------------------------------------------------------
# Forward Simulator G(θ)
# ---------------------------------------------------------------------------

def simulate_signaling(
    F_baseline: np.ndarray,
    theta: np.ndarray,
    adjacency: np.ndarray,
    n_steps: int = 10,
    dt: float = 1.0,
) -> np.ndarray:
    """
    Forward-simulate the exerkine signaling network after introducing a
    candidate biomolecule with parameters θ.

    G(θ) is approximated as a simple linear diffusion update seeded by θ.

    Parameters
    ----------
    F_baseline : np.ndarray, shape (N, E)
        Baseline exerkine flux without the candidate molecule.
    theta : np.ndarray, shape (n_params,)
        Biomolecule parameters.
    adjacency : np.ndarray, shape (N, N)
        Cell–cell adjacency.
    n_steps : int
        Number of forward simulation steps.
    dt : float
        Time step.

    Returns
    -------
    F_final : np.ndarray, shape (N, E)
        Predicted exerkine flux after the molecule acts.
    """
    secretion = theta[0]
    stability = max(theta[2], 1e-3)
    decay = np.log(2) / stability  # first-order decay

    exerkine_weights = theta[3:]
    n_exerkines = F_baseline.shape[1]
    # Clip weights to match available channels
    exerkine_weights = exerkine_weights[:n_exerkines]

    F = F_baseline.copy()
    # Row-normalised adjacency for diffusion
    W = adjacency / (adjacency.sum(axis=1, keepdims=True) + 1e-8)

    for _ in range(n_steps):
        # Secretion: all cells secrete the biomolecule proportionally
        secretion_input = secretion * exerkine_weights[np.newaxis, :]  # (1, E)
        # Diffusion step
        F = F + dt * (W @ F - decay * F + secretion_input)

    return F


# ---------------------------------------------------------------------------
# Utility Function U
# ---------------------------------------------------------------------------

def therapeutic_utility(
    F_simulated: np.ndarray,
    F_target: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.1,
    lri_scores: Optional[np.ndarray] = None,
) -> float:
    """
    Compute therapeutic utility U(G(θ)).

    U = -α ‖F_sim − F_target‖² + β · mean(LRI_scores)

    Higher is better (maximisation problem).

    Parameters
    ----------
    F_simulated : np.ndarray, shape (N, E)
    F_target : np.ndarray, shape (N, E)
    alpha : float   — state matching weight
    beta : float    — signaling richness bonus
    lri_scores : np.ndarray, shape (N, N), optional

    Returns
    -------
    U : float
    """
    state_cost = -alpha * np.mean((F_simulated - F_target) ** 2)
    lri_bonus = beta * np.mean(lri_scores) if lri_scores is not None else 0.0
    return float(state_cost + lri_bonus)


# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------

class GenerativeBiomoleculeOptimiser:
    """
    Optimise biomolecule parameters θ to maximise therapeutic utility U.

    Wraps scipy global/local optimisers around the forward simulator.

    Parameters
    ----------
    design_space : BiomoleculeDesignSpace
    F_baseline : np.ndarray, shape (N, E)
    F_target : np.ndarray, shape (N, E)
    adjacency : np.ndarray, shape (N, N)
    n_sim_steps : int
    """

    def __init__(
        self,
        design_space: BiomoleculeDesignSpace,
        F_baseline: np.ndarray,
        F_target: np.ndarray,
        adjacency: np.ndarray,
        n_sim_steps: int = 10,
        lri_scores: Optional[np.ndarray] = None,
    ) -> None:
        self.ds = design_space
        self.F_baseline = F_baseline
        self.F_target = F_target
        self.adjacency = adjacency
        self.n_sim_steps = n_sim_steps
        self.lri_scores = lri_scores

        self.best_theta_: Optional[np.ndarray] = None
        self.best_utility_: float = -np.inf
        self.history_: List[float] = []

    def _objective(self, theta: np.ndarray) -> float:
        """Negative utility for minimisation."""
        F_sim = simulate_signaling(
            self.F_baseline, theta, self.adjacency, self.n_sim_steps
        )
        return -therapeutic_utility(F_sim, self.F_target, lri_scores=self.lri_scores)

    def optimise(
        self,
        method: str = "differential_evolution",
        maxiter: int = 200,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Run global optimisation to find θ* = argmax_θ U(G(θ)).

        Parameters
        ----------
        method : str
            ``"differential_evolution"`` (recommended for non-convex landscape)
            or ``"L-BFGS-B"`` for fast local refinement.
        maxiter : int
        seed : int

        Returns
        -------
        theta_star : np.ndarray, shape (n_params,)
        """
        bounds = self.ds.bounds

        if method == "differential_evolution":
            result = differential_evolution(
                self._objective,
                bounds=bounds,
                maxiter=maxiter,
                seed=seed,
                tol=1e-6,
                polish=True,
            )
        else:
            theta0 = self.ds.sample(1, seed=seed)[0]
            result = minimize(
                self._objective,
                theta0,
                method=method,
                bounds=bounds,
                options={"maxiter": maxiter},
            )

        self.best_theta_ = result.x
        self.best_utility_ = -result.fun
        return self.best_theta_

    def decode_best(self) -> Dict[str, float | np.ndarray]:
        """Return human-readable properties of the optimal θ*."""
        if self.best_theta_ is None:
            raise RuntimeError("Call optimise() first.")
        return self.ds.decode(self.best_theta_)


# ---------------------------------------------------------------------------
# End-to-end pipeline convenience function
# ---------------------------------------------------------------------------

def generative_exerkine_design(
    F_baseline: np.ndarray,
    F_target: np.ndarray,
    adjacency: np.ndarray,
    lri_scores: Optional[np.ndarray] = None,
    n_sim_steps: int = 10,
    maxiter: int = 100,
    seed: int = 0,
) -> Tuple[np.ndarray, Dict, float]:
    """
    Full generative feedback loop:  detect gaps → propose → simulate → optimise.

    Returns
    -------
    theta_star : np.ndarray  — optimal biomolecule parameters
    properties : dict        — decoded human-readable properties
    utility    : float       — achieved therapeutic utility U(G(θ*))
    """
    n_exerkines = F_baseline.shape[1]
    ds = BiomoleculeDesignSpace(n_exerkines=n_exerkines)
    opt = GenerativeBiomoleculeOptimiser(
        ds, F_baseline, F_target, adjacency,
        n_sim_steps=n_sim_steps,
        lri_scores=lri_scores,
    )
    theta_star = opt.optimise(maxiter=maxiter, seed=seed)
    return theta_star, opt.decode_best(), opt.best_utility_
