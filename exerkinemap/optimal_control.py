"""
Layer 6: Optimal Control — Linear Quadratic Regulator (LQR)
=============================================================
Objective::

    min_u  ∫₀ᵀ [ F(t)ᵀ Q F(t) + u(t)ᵀ R u(t) ] dt

Subject to::

    dF/dt = A F(t) + B u(t)

where
    - ``F(t)`` : exerkine flux state vector (N×E)
    - ``u(t)``  : exercise/therapy intervention control vector
    - ``A``     : signaling dynamics matrix (learned from ODE layer)
    - ``B``     : input/actuation matrix (which exerkines can be targeted)
    - ``Q``     : state cost (penalise pathological flux deviations)
    - ``R``     : control cost (penalise high-intensity interventions)

Solving the continuous-time algebraic Riccati equation (CARE) yields the
optimal feedback gain **K** and the optimal prescription **u*(t) = −K F(t)**.

Reference (JSX):  2.2.8 Optimal Control  |  Tag: LQR · Control Theory · Optimal Rx
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import solve_continuous_are, solve_discrete_are


# ---------------------------------------------------------------------------
# Continuous-time LQR
# ---------------------------------------------------------------------------

def solve_lqr(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the infinite-horizon continuous-time LQR problem.

    Solves the Continuous Algebraic Riccati Equation (CARE)::

        Aᵀ P + P A − P B R⁻¹ Bᵀ P + Q = 0

    Parameters
    ----------
    A : np.ndarray, shape (n, n)
        Signaling dynamics matrix.
    B : np.ndarray, shape (n, m)
        Control input matrix.
    Q : np.ndarray, shape (n, n)
        State cost matrix (positive semi-definite).
    R : np.ndarray, shape (m, m)
        Control cost matrix (positive definite).

    Returns
    -------
    K : np.ndarray, shape (m, n)
        Optimal feedback gain such that u*(t) = −K F(t).
    P : np.ndarray, shape (n, n)
        Solution to the CARE (value function matrix).
    eigvals : np.ndarray
        Closed-loop eigenvalues of (A − B K).
    """
    P = solve_continuous_are(A, B, Q, R)
    R_inv = np.linalg.inv(R)
    K = R_inv @ B.T @ P
    closed_loop = A - B @ K
    eigvals = np.linalg.eigvals(closed_loop)
    return K, P, eigvals


def solve_lqr_discrete(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the infinite-horizon discrete-time LQR problem.

    Solves the Discrete Algebraic Riccati Equation (DARE).

    Returns
    -------
    K : np.ndarray, shape (m, n)
    P : np.ndarray, shape (n, n)
    """
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    return K, P


# ---------------------------------------------------------------------------
# Exerkine LQR Controller
# ---------------------------------------------------------------------------

class ExerkineLQRController:
    """
    Optimal exercise prescription via Linear Quadratic Regulator.

    Learns the linearised signaling dynamics around a target physiological
    state and computes the minimum-cost intervention trajectory u*(t).

    Parameters
    ----------
    n_states : int
        Dimensionality of the exerkine state vector (N cells × E exerkines,
        flattened or summarised).
    n_controls : int
        Number of controllable intervention channels (exercise modalities,
        targeted therapies, etc.).
    dt : float
        Simulation time step (minutes).
    """

    def __init__(
        self,
        n_states: int,
        n_controls: int,
        dt: float = 1.0,
    ) -> None:
        self.n_states = n_states
        self.n_controls = n_controls
        self.dt = dt

        # Dynamics + actuation (set by fit or provided directly)
        self.A: Optional[np.ndarray] = None
        self.B: Optional[np.ndarray] = None

        # LQR solution
        self.K: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def set_dynamics(
        self,
        A: np.ndarray,
        B: np.ndarray,
    ) -> "ExerkineLQRController":
        """
        Provide the linearised signaling dynamics matrices directly.

        Parameters
        ----------
        A : np.ndarray, shape (n_states, n_states)
            State transition / signaling matrix.  Typically estimated from
            the ODE layer (signal_diffusion) around the equilibrium point.
        B : np.ndarray, shape (n_states, n_controls)
            Actuation matrix mapping exercise inputs to flux changes.
        """
        assert A.shape == (self.n_states, self.n_states), "A shape mismatch"
        assert B.shape == (self.n_states, self.n_controls), "B shape mismatch"
        self.A = A
        self.B = B
        return self

    def estimate_dynamics_from_flux(
        self,
        F_trajectory: np.ndarray,
        U_trajectory: np.ndarray,
    ) -> "ExerkineLQRController":
        """
        Estimate A and B from an observed flux trajectory via least squares.

        Parameters
        ----------
        F_trajectory : np.ndarray, shape (T, n_states)
            State trajectory (T time steps).
        U_trajectory : np.ndarray, shape (T-1, n_controls)
            Control inputs applied at each step.

        Returns
        -------
        self
        """
        T = F_trajectory.shape[0]
        dF = np.diff(F_trajectory, axis=0) / self.dt  # (T-1, n_states)

        # Linear regression: dF = F[:-1] @ Aᵀ + U @ Bᵀ
        Z = np.hstack([F_trajectory[:-1], U_trajectory])  # (T-1, n_states+n_controls)
        params, *_ = np.linalg.lstsq(Z, dF, rcond=None)
        self.A = params[: self.n_states].T
        self.B = params[self.n_states :].T
        return self

    # ------------------------------------------------------------------
    def solve(
        self,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
    ) -> "ExerkineLQRController":
        """
        Solve the LQR problem and store the optimal feedback gain K.

        Parameters
        ----------
        Q : np.ndarray, shape (n_states, n_states), optional
            State cost.  Default: identity (equal cost per state dimension).
        R : np.ndarray, shape (n_controls, n_controls), optional
            Control cost.  Default: 0.1 × identity (favour low-intensity Rx).

        Returns
        -------
        self
        """
        if self.A is None or self.B is None:
            raise RuntimeError("Call set_dynamics() or estimate_dynamics_from_flux() first.")

        Q = Q if Q is not None else np.eye(self.n_states)
        R = R if R is not None else 0.1 * np.eye(self.n_controls)

        self.K, self.P, self.eigvals_ = solve_lqr(self.A, self.B, Q, R)
        return self

    # ------------------------------------------------------------------
    def optimal_control(self, F_current: np.ndarray) -> np.ndarray:
        """
        Compute the optimal intervention u*(t) = −K F(t).

        Parameters
        ----------
        F_current : np.ndarray, shape (n_states,)
            Current exerkine flux state.

        Returns
        -------
        u_star : np.ndarray, shape (n_controls,)
            Optimal exercise prescription (control input).
        """
        if self.K is None:
            raise RuntimeError("Call solve() before computing control.")
        return -self.K @ F_current

    # ------------------------------------------------------------------
    def simulate_trajectory(
        self,
        F0: np.ndarray,
        T_steps: int,
        F_target: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate closed-loop state trajectory under optimal control.

        Parameters
        ----------
        F0 : np.ndarray, shape (n_states,)
            Initial exerkine flux.
        T_steps : int
            Number of simulation steps.
        F_target : np.ndarray, shape (n_states,), optional
            Target physiological state (tracking LQR).  If provided, the error
            ``e(t) = F(t) − F_target`` is regulated to zero.

        Returns
        -------
        F_traj : np.ndarray, shape (T_steps+1, n_states)
        U_traj : np.ndarray, shape (T_steps, n_controls)
        """
        if self.K is None:
            raise RuntimeError("Call solve() first.")

        F_traj = np.zeros((T_steps + 1, self.n_states))
        U_traj = np.zeros((T_steps, self.n_controls))
        F_traj[0] = F0

        for t in range(T_steps):
            e = F_traj[t] - (F_target if F_target is not None else np.zeros(self.n_states))
            u = -self.K @ e
            U_traj[t] = u
            # Euler integration: F_{t+1} = F_t + dt * (A F_t + B u_t)
            F_traj[t + 1] = F_traj[t] + self.dt * (self.A @ F_traj[t] + self.B @ u)

        return F_traj, U_traj

    # ------------------------------------------------------------------
    def prescribe(
        self,
        F_current: np.ndarray,
        modality_labels: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Translate u*(t) into a human-readable exercise prescription.

        Parameters
        ----------
        F_current : np.ndarray, shape (n_states,)
        modality_labels : list of str, optional
            Names for each control channel (e.g., ["aerobic_intensity",
            "resistance_load", "recovery_duration"]).

        Returns
        -------
        prescription : dict {modality: value}
        """
        u_star = self.optimal_control(F_current)
        labels = modality_labels or [f"modality_{i}" for i in range(self.n_controls)]
        return {label: float(val) for label, val in zip(labels, u_star)}


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def build_signaling_matrix(adjacency: np.ndarray, decay: float = 0.1) -> np.ndarray:
    """
    Construct a stable signaling dynamics matrix A from a cell adjacency.

    A = W − λ I  (neighbourhood spread minus self-decay)

    Parameters
    ----------
    adjacency : np.ndarray, shape (N, N)
    decay : float
        Self-decay rate λ (ensures stability when λ > spectral radius of W).

    Returns
    -------
    A : np.ndarray, shape (N, N)
    """
    W = adjacency / (adjacency.sum(axis=1, keepdims=True) + 1e-8)  # row-normalise
    return W - decay * np.eye(W.shape[0])


def default_actuation_matrix(
    n_states: int,
    n_controls: int,
    seed: int = 0,
) -> np.ndarray:
    """
    Return a default B matrix where controls uniformly affect all states.

    In practice B should be constructed from known exercise→exerkine mappings
    (e.g., aerobic exercise upregulates IL-6, BDNF via specific cell types).
    """
    rng = np.random.default_rng(seed)
    B = rng.normal(0, 0.1, size=(n_states, n_controls))
    return B
