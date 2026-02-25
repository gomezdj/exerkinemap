# ExerkineMap **Virtual Physiological System v2.2** — a computational framework for exercise-induced exerkine signaling across cells, tissues, and organs.

ExerkineMap integrates nine mathematical layers — from multi-omic state embeddings through causal inference, optimal control theory, and generative biomolecule design — to model, simulate, and prescribe exercise interventions at single-cell resolution.

## Key Features
* Eight-Layer Mathematical Pipeline: Encodes unified cellular states through VAE and GNN multi-omic embeddings (RNA, ATAC, RRBS, proteomics), propagates exerkine signals via reaction-diffusion ODEs ($dF/dt = -\lambda F + \sum \omega_{ij} F_j(t) + u_i(t)$), and classifies exercise states across autocrine, paracrine, and endocrine signaling scales.

* Probabilistic Ligand-Receptor Communication: Scores ligand-receptor interactions using Bayesian Monte Carlo estimation with STRING and Reactome priors — resolving receptor affinity, pairwise activation probabilities ($P_{ij} = \sigma(\alpha \log x_i(l) + \gamma \log x_j(r) + \eta_{ij})$), and downstream pathway impact at single-cell resolution.

* Causal Inference & Optimal Exercise Control: Applies Pearl do-calculus structural causal models to quantify exercise intervention effects ($C_{ij} = \mathbb{E}[Y_i \mid do(F_j)] - \mathbb{E}[Y_i]$) and solves LQR optimal control problems to prescribe individualized exerkine secretion trajectories.

* Generative Biomolecule Design & Ensemble Prediction: Combines XGBoost, RandomForest, and LightGBM stacked ensembles over diffused exerkine features with a generative biomolecule optimizer to detect signaling gaps and propose pharmacological mimetics for disease contexts including cancer, cardiovascular, and neurological disorders.

* Multi-Consortia Spatial-Temporal Integration: Harmonizes multimodal spatial and single-cell data from MoTrPAC, HuBMAP, HTAN, and PsychENCODE using graph neural networks to resolve inter-organ exerkine communication networks and temporal diffusion trajectories.

# Installation 🧬

**Option 1 — pip (core)**
```bash
git clone https://github.com/gomezdj/exerkinemap.git
cd exerkinemap
pip install -r requirements.txt
```

**Option 2 — conda (full environment with spatial & single-cell dependencies)**
```bash
conda env create -f exerkinemap.yml
conda activate exerkinemap
pip install -e .
```

**Requirements**
| Dependency | Version | Notes |
|---|---|---|
| Python | ≥ 3.9 | |
| PyTorch | ≥ 2.7 | Spatial GNN layers |
| torch-geometric | ≥ 2.6 | `build_spatial_graph`, `SpatialOmicsModel` |
| scanpy / squidpy / anndata | see yml | Single-cell & spatial workflows |
| CUDA GPU | recommended | Signal diffusion & GNN acceleration |

# Quick Start Guide

## Layer-by-Layer Pipeline

```python
import numpy as np
import exerkinemap as em
from ensembl.tree_ensembl import ExerkineTreeEnsemble
from exerkinemap.optimal_control import solve_lqr, build_signaling_matrix, ExerkineLQRController
from exerkinemap.spatial_graph_lri import build_spatial_graph, infer_lri_pairs

# Shared inputs
EXERKINES = ['IL6', 'FGF21', 'TGFB', 'BDNF', 'IGF1', 'IL15']
LIGANDS   = ['IL6',  'FGF21', 'TGFB']
RECEPTORS = ['IL6R', 'FGFR1', 'TGFBR1']
ORGANS    = ['muscle', 'liver', 'brain', 'immune']

# ── Layer 1: Multi-Omic State Embedding ─────────────────────────────────────
# Z_i = f_multi(G^RNA, G^ATAC, G^RRBS, P_i)   [VAE | GNN | Foundation Models]
Z = em.integrate_multiomics(
    G_rna=rna_matrix,       # (N, G_rna)  log-normalised scRNA-seq counts
    G_atac=atac_matrix,     # (N, G_atac) binarised ATAC peak accessibility
    G_rrbs=rrbs_matrix,     # (N, G_cpg)  RRBS beta-values in [0, 1]
    P=protein_matrix,       # (N, G_prot) spatial proteomics abundances
    latent_dim=64,
)

# ── Layer 2: Probabilistic LRI Communication ────────────────────────────────
# P_ij = σ(α log x_i(l) + γ log x_j(r) + η_ij)   [Bayesian MC · Bernoulli prior]
f_0 = em.compute_initial_exerkine_profile(
    G_expr=rna_matrix,
    adjacency=adjacency,            # (N, N) k-NN or radius cell graph
    exerkines=EXERKINES,
    gene_names=gene_names,
)
lri_scores = em.compute_cell_lri_scores(
    G_expr=rna_matrix,
    ligands=LIGANDS,
    receptors=RECEPTORS,
    adjacency=adjacency,
    beta_pathway={'FGFR1': 1.4, 'IL6R': 1.2, 'TGFBR1': 0.9},
)
P_activation = em.compute_receptor_activation_scores(
    G_expr=rna_matrix,
    ligands=LIGANDS,
    receptors=RECEPTORS,
    adjacency=adjacency,
)

# ── Layer 3: Dynamic Exerkine Flux ──────────────────────────────────────────
# dF_i/dt = -λ F_i + Σ ω_ij F_j(t) + u_i(t)   [ODE · Reaction-Diffusion]
# F(t) = e^(-tL) · f_0
F_t = em.compute_diffused_signal(adjacency=adjacency, f_0=f_0, t=2.0)
F_series = em.compute_diffusion_time_series(
    adjacency=adjacency, f_0=f_0, time_points=[0.5, 1.0, 2.0, 5.0]
)                                   # (N, T) — paracrine → endocrine transition

# ── Layer 4: Causal Inference ───────────────────────────────────────────────
# C_ij = E[Y_i | do(F_j)] − E[Y_i]   [SCM · do-Calculus · Counterfactual]
scm = em.StructuralCausalModel(
    n_cells=rna_matrix.shape[0],
    n_exerkines=len(EXERKINES),
)
scm.fit(F=F_t, Y=outcome_matrix, adjacency=adjacency)
causal_scores = em.compute_causal_scores(scm, target_cells=target_idx)
intervention  = em.exercise_intervention_effect(scm, F_j=F_t, intervention_magnitude=1.5)

# ── Layer 5: Tree Ensemble Prediction ───────────────────────────────────────
# Ŷ = Σ α_k f_k(Z)   [XGBoost · RandomForest · LightGBM — CV-weighted]
ensemble = ExerkineTreeEnsemble()
X_features = np.hstack([Z, F_t, lri_scores])   # fuse latent state + flux + LRI
ensemble.train(X_features, y=labels)
Y_hat = ensemble.predict(X_features)            # tissue-level exercise state

# ── Layer 6: Optimal Control — Exercise Prescription ────────────────────────
# min_u ∫(F^T Q F + u^T R u) dt   s.t.  dF/dt = AF + Bu   [LQR · Riccati]
A = build_signaling_matrix(adjacency=adjacency, F=F_t)
n, m = A.shape[0], len(EXERKINES)
B = np.eye(n, m)                                # each exerkine is a controllable input
Q = np.eye(n)                                   # penalise pathological flux deviations
R = 0.1 * np.eye(m)                             # low control cost → allow intervention
K, P, eigvals = solve_lqr(A=A, B=B, Q=Q, R=R)

controller = ExerkineLQRController(A=A, B=B, Q=Q, R=R)
u_star = controller.compute_optimal_input(F_current=F_t.mean(axis=0))
print(f"Optimal u*(t) : {u_star}")              # prescribed exerkine secretion rates

# ── Layer 7: Generative Biomolecule Design ──────────────────────────────────
# θ* = argmax_θ U(G_ExerkineMap(θ))   [Gap detection → Candidate design → Optimise]
gaps       = em.detect_signaling_gaps(F=F_t, lri_scores=lri_scores)
design     = em.BiomoleculeDesignSpace()
candidates = em.generative_exerkine_design(gaps, design_space=design)
optimised  = em.GenerativeBiomoleculeOptimiser().optimise(
    candidates, forward_model=lambda theta: em.compute_diffused_signal(
        adjacency=adjacency, f_0=theta, t=2.0
    )
)

# ── Layer 8: Cross-Organ Transport ──────────────────────────────────────────
# F_o(t+1) = Σ_o' W_oo' F_o'(t)   [Organ graph · Vascular/lymphatic edges]
W_organ = np.array([                               # organ-organ transport weights
    [0.0, 0.6, 0.3, 0.1],   # muscle → liver, brain, immune
    [0.4, 0.0, 0.3, 0.3],   # liver  →
    [0.2, 0.2, 0.0, 0.6],   # brain  →
    [0.3, 0.3, 0.4, 0.0],   # immune →
])
F_organ = np.stack([F_t.mean(axis=0) for _ in ORGANS])
for _ in range(10):                                # simulate 10 inter-organ steps
    F_organ = W_organ @ F_organ                    # F_o(t+1) = Σ W_oo' F_o'(t)

# ── Validation Metric ───────────────────────────────────────────────────────
# ExerkineScore = (1/N) Σ P_activation(c_i)
exerkine_score = float(P_activation.mean())
print(f"ExerkineScore  : {exerkine_score:.4f}")
print(f"Causal Effect  : {causal_scores.mean():.4f}")
print(f"Ensemble R²    : {ensemble.weights}")
print(f"LQR Stability  : {'stable' if (eigvals.real < 0).all() else 'check gains'}")
```
