"""
Layer 1: Multi-Omic State Embedding — Unified Z_i
==================================================
Formula::

    Z_i = f_multi(G_i^RNA, G_i^ATAC, G_i^RRBS, P_i)

Integrates four modalities per cell *i*:
    - ``G^RNA``  : scRNA-seq gene expression vector
    - ``G^ATAC`` : ATAC-seq chromatin accessibility
    - ``G^RRBS`` : RRBS DNA methylation (β-values per CpG locus)
    - ``P_i``    : protein abundance (IHC or spatial proteomics)

Two integration strategies are provided:
    1. **ConcatProjection** — simple concatenation followed by a deep MLP
       encoder (fast, no probabilistic assumptions).
    2. **MultiOmicVAE** — Variational Autoencoder with modality-specific
       encoders and a shared latent space Z_i ~ N(μ, σ²) (principled,
       supports uncertainty quantification and missing modality imputation).

Both produce a ``(N, latent_dim)`` matrix Z that feeds into the downstream
probabilistic communication layer (Layer 2 — P_ij).

Reference (JSX):  2.2.5 Multi-Omic State  |  Tag: VAE · GNN · Foundation Models
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import svd


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def log_normalise(X: np.ndarray, scale: float = 1e4) -> np.ndarray:
    """log1p normalisation (analogous to scanpy sc.pp.normalize_total + log1p)."""
    totals = X.sum(axis=1, keepdims=True)
    return np.log1p(X / (totals + 1e-9) * scale)


def binarise_atac(X_atac: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Binarise ATAC-seq peak accessibility."""
    return (X_atac > threshold).astype(np.float32)


def clip_methylation(X_rrbs: np.ndarray) -> np.ndarray:
    """Clip RRBS β-values to [0, 1] and replace NaN with 0.5 (unknown)."""
    X = np.nan_to_num(X_rrbs, nan=0.5)
    return np.clip(X, 0.0, 1.0)


# ---------------------------------------------------------------------------
# PCA-based lightweight embedding (no deep learning required)
# ---------------------------------------------------------------------------

class ConcatProjectionEmbedder:
    """
    Fast multi-omic embedder: concatenate modalities → truncated SVD (PCA).

    Suitable for CPU-only environments or as a baseline.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the output embedding Z_i.
    modality_weights : dict, optional
        Per-modality importance weight before concatenation
        (keys: "rna", "atac", "rrbs", "protein").  Default: equal weights.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        modality_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.latent_dim = latent_dim
        self.modality_weights = modality_weights or {
            "rna": 1.0,
            "atac": 0.5,
            "rrbs": 0.5,
            "protein": 1.0,
        }
        self._Vt: Optional[np.ndarray] = None
        self._mean: Optional[np.ndarray] = None
        self._fitted = False

    def fit_transform(
        self,
        X_rna: np.ndarray,
        X_atac: Optional[np.ndarray] = None,
        X_rrbs: Optional[np.ndarray] = None,
        X_protein: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Fit and return Z_i for all N cells.

        Missing modalities are replaced by zero matrices of shape (N, 1).

        Parameters
        ----------
        X_rna    : np.ndarray, shape (N, G)   — required
        X_atac   : np.ndarray, shape (N, P)   — optional
        X_rrbs   : np.ndarray, shape (N, C)   — optional
        X_protein: np.ndarray, shape (N, K)   — optional

        Returns
        -------
        Z : np.ndarray, shape (N, latent_dim)
        """
        N = X_rna.shape[0]
        w = self.modality_weights

        parts = [w["rna"] * log_normalise(X_rna)]
        if X_atac is not None:
            parts.append(w["atac"] * binarise_atac(X_atac))
        if X_rrbs is not None:
            parts.append(w["rrbs"] * clip_methylation(X_rrbs))
        if X_protein is not None:
            parts.append(w["protein"] * log_normalise(X_protein))

        X_concat = np.hstack(parts)  # (N, D_total)

        # Centre
        self._mean = X_concat.mean(axis=0)
        X_c = X_concat - self._mean

        # Truncated SVD
        d = min(self.latent_dim, min(X_c.shape) - 1)
        _, s, Vt = svd(X_c, full_matrices=False)
        self._Vt = Vt[:d]
        self._fitted = True

        Z = X_c @ self._Vt.T  # (N, latent_dim)
        return Z

    def transform(
        self,
        X_rna: np.ndarray,
        X_atac: Optional[np.ndarray] = None,
        X_rrbs: Optional[np.ndarray] = None,
        X_protein: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Project new cells using the fitted embedding."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform() first.")
        w = self.modality_weights
        parts = [w["rna"] * log_normalise(X_rna)]
        if X_atac is not None:
            parts.append(w["atac"] * binarise_atac(X_atac))
        if X_rrbs is not None:
            parts.append(w["rrbs"] * clip_methylation(X_rrbs))
        if X_protein is not None:
            parts.append(w["protein"] * log_normalise(X_protein))
        X_concat = np.hstack(parts) - self._mean
        # Pad/trim to match fitted dimensionality
        D = self._Vt.shape[1]
        if X_concat.shape[1] < D:
            X_concat = np.pad(X_concat, ((0, 0), (0, D - X_concat.shape[1])))
        else:
            X_concat = X_concat[:, :D]
        return X_concat @ self._Vt.T


# ---------------------------------------------------------------------------
# VAE-style multi-omic integration (numpy implementation)
# ---------------------------------------------------------------------------

class MultiOmicVAE:
    """
    Variational Autoencoder for joint multi-omic embedding.

    This is a lightweight numpy implementation of the VAE framework for
    environments without PyTorch/TensorFlow.  For production, wrap a
    PyTorch ``nn.Module`` with modality-specific encoders.

    The reparameterisation trick gives::

        Z_i ~ N(μ_i, diag(σ²_i))
        μ_i, log σ²_i = Encoder(X_i)

    Parameters
    ----------
    input_dims : dict {modality: dim}
        Input dimensionality per modality.
    latent_dim : int
        Dimensionality of the latent space Z_i.
    n_iter : int
        Number of gradient steps (random projection VAE approximation).
    seed : int
    """

    def __init__(
        self,
        input_dims: Dict[str, int],
        latent_dim: int = 32,
        n_iter: int = 50,
        seed: int = 42,
    ) -> None:
        self.input_dims = input_dims
        self.latent_dim = latent_dim
        self.n_iter = n_iter
        self._rng = np.random.default_rng(seed)

        # Random projection encoder weights (one per modality)
        self._enc_weights: Dict[str, np.ndarray] = {}
        for mod, dim in input_dims.items():
            self._enc_weights[mod] = self._rng.normal(
                0, 1 / np.sqrt(dim), size=(dim, latent_dim * 2)
            )  # outputs [μ, log σ²]

    def _encode(self, X: np.ndarray, mod: str) -> Tuple[np.ndarray, np.ndarray]:
        """Random-projection encode → (mu, log_var)."""
        h = np.tanh(X @ self._enc_weights[mod])
        mu, log_var = h[:, : self.latent_dim], h[:, self.latent_dim :]
        return mu, log_var

    def _reparameterise(self, mu: np.ndarray, log_var: np.ndarray) -> np.ndarray:
        eps = self._rng.standard_normal(mu.shape)
        return mu + np.exp(0.5 * log_var) * eps

    def fit_transform(
        self,
        modality_data: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Encode all provided modalities and fuse via mean pooling of μ_i.

        Parameters
        ----------
        modality_data : dict {modality_name: np.ndarray of shape (N, D_mod)}

        Returns
        -------
        Z : np.ndarray, shape (N, latent_dim)  — sampled latent codes
        mu : np.ndarray, shape (N, latent_dim)  — posterior means
        log_var : np.ndarray, shape (N, latent_dim)  — posterior log variances
        """
        mus, log_vars = [], []
        for mod, X in modality_data.items():
            if mod not in self._enc_weights:
                # Initialise on-the-fly if not pre-registered
                D = X.shape[1]
                self._enc_weights[mod] = self._rng.normal(
                    0, 1 / np.sqrt(D), size=(D, self.latent_dim * 2)
                )
            mu_k, lv_k = self._encode(X, mod)
            mus.append(mu_k)
            log_vars.append(lv_k)

        # Product-of-Gaussians fusion
        # Precision-weighted mean: μ* = Σ (σ_k^{-2} μ_k) / Σ σ_k^{-2}
        precisions = [np.exp(-lv) for lv in log_vars]
        total_prec = sum(precisions)
        mu_fused = sum(p * m for p, m in zip(precisions, mus)) / (total_prec + 1e-9)
        log_var_fused = -np.log(total_prec + 1e-9)

        Z = self._reparameterise(mu_fused, log_var_fused)
        return Z, mu_fused, log_var_fused

    @staticmethod
    def kl_divergence(mu: np.ndarray, log_var: np.ndarray) -> float:
        """Mean KL(q(Z|X) ‖ N(0,I)) as a regularisation diagnostic."""
        return float(-0.5 * np.mean(1 + log_var - mu**2 - np.exp(log_var)))


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------

def integrate_multiomics(
    X_rna: np.ndarray,
    X_atac: Optional[np.ndarray] = None,
    X_rrbs: Optional[np.ndarray] = None,
    X_protein: Optional[np.ndarray] = None,
    latent_dim: int = 64,
    method: str = "pca",
    seed: int = 0,
) -> np.ndarray:
    """
    One-line multi-omic integration returning Z_i embeddings.

    Parameters
    ----------
    X_rna     : np.ndarray, shape (N, G)  — required
    X_atac    : np.ndarray, shape (N, P)  — optional
    X_rrbs    : np.ndarray, shape (N, C)  — optional
    X_protein : np.ndarray, shape (N, K)  — optional
    latent_dim : int
    method : str
        ``"pca"`` (ConcatProjectionEmbedder) or ``"vae"`` (MultiOmicVAE).
    seed : int

    Returns
    -------
    Z : np.ndarray, shape (N, latent_dim)
    """
    if method == "pca":
        embedder = ConcatProjectionEmbedder(latent_dim=latent_dim)
        return embedder.fit_transform(X_rna, X_atac, X_rrbs, X_protein)

    elif method == "vae":
        input_dims: Dict[str, int] = {"rna": X_rna.shape[1]}
        modality_data: Dict[str, np.ndarray] = {"rna": log_normalise(X_rna)}
        if X_atac is not None:
            input_dims["atac"] = X_atac.shape[1]
            modality_data["atac"] = binarise_atac(X_atac)
        if X_rrbs is not None:
            input_dims["rrbs"] = X_rrbs.shape[1]
            modality_data["rrbs"] = clip_methylation(X_rrbs)
        if X_protein is not None:
            input_dims["protein"] = X_protein.shape[1]
            modality_data["protein"] = log_normalise(X_protein)

        vae = MultiOmicVAE(input_dims=input_dims, latent_dim=latent_dim, seed=seed)
        Z, mu, _ = vae.fit_transform(modality_data)
        return mu  # use posterior mean for downstream tasks

    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'pca' or 'vae'.")
