# EXERKINEMAP: Mathematical Model

## Definitions & Variables

- Cells: \( C = \{c_1, ..., c_N\} \)
- Exerkines: \( E \subset L \)
- Ligands: \( L = \{l_1, ..., l_k\} \)
- Receptors: \( R = \{r_1, ..., r_m\} \)
- Neighbors: \( N_i \)
- Gene expression matrix: \( G \in \mathbb{R}^{N \times G} \)

## Key Equations

- Initial exerkine profile:
  \[
  f_0(i) = \frac{x_i^{exerkine} + \sum_{j \in N_i} w_{ij} x_j^{exerkine}}{1 + \sum_{j \in N_i} w_{ij}}
  \]
- Ligand-receptor interaction:
  \[
  S_{i,j}^{(l_k, r_m)} = x_i(l_k) \cdot x_j(r_m)
  \]
- Signal propagation:
  \[
  F(t) = e^{-t L} \cdot f_0
  \]
- Pathway activation:
  \[
  P_{activation}(c_j | c_i) = \sum_{(l_k, r_m)} \delta_{l_k \in E} S_{ij}^{(l_k, r_m)} \beta_{r_m}
  \]

Pipeline Layers

## Multi-Omic State 
# Multi-Omic Z_i
# VAE GNN Foundational Mdodes
Z_i = f_multi(G_i^RNA, G_i^ATAC, G_i^RRBS, P_i)
"""
Unified cellular state embedding via VAE or GNN. Integrates RNA, chromatin, methylation, proteomics, into latent Z_i.
"""

##  Probabilistic Communcation
# P_ij Signal
# Bayesian Monte Carlo Bernoulli(n_lr)
P_ij = /sigma(a log x_i(l) + y log x_j(r) + /nu_ij)
"""
Stochastic ligand-receptor interaction score. Bernoulli prior /pi_lr from STRING/Reactome. Enables Bayesian inference & Monte Carlo sim.
"""

## Dynamic Exerkine Flux
# dF/dt ODE
# - ODE - Reaction-Diffusion -Temporal
dF_i/dt = \lambdaF_i + 
Sigma \omega_ij F_j(t) + u_i(t)
"""
Reaction-diffusion ODE. Degradation \lambda, neighborhood spread omega_ij, secretion  u_i(t) conditioned on genomic/metabolic/env state.
"""

## Causal Inference
# do-Calculus C_ij
# - SCM -do-Calculus - Counterfactual 
C_ij = E[Y_i | do(F_j)] - E[Y_i]

"""
Pearl do-calculus causal influence. Structural causal models, Bayesian networks, counterfactual simulation of exercise interventions.
"""

## Tree Ensemble
# Y_hat Prediction
# - XGBoost - RandomForest - LightGBM

Y = \Sigma \alpha_k f_k(Z) [XGM + RF + LightGBM]

"""
Stacked ensemble over diffused exerkine features. XGBoost sequential boosting, RF bagging, LightGBM histogram method.
"""

## Optimal Control
# LQR u*(t)
# - LQR - Control Theory - Optimal Rx
min_u \integral(F^TQF + + u^TRu)dt s.t. F=AF+Bu

"""
Linear-quadratic regulator over signaling dynamics. A=signaling matrix, B=intervention, u=exercise/therapy. Yields optimal prescription.
"""

## Generative Feedback
# \Theta* Biomolecule
- Diffusion - Transformer - ProteinDesign

\Theta* = argmax_\theta U(G_ExerkineMap(\theta))

"""
Generative loop: identify exerkine gaps --> design candidate biomolecules --> simulate signaling --> optimize therapeutic objective U.
"""

## Cross-Organ Transport
G_organ Graph
- Organ Graph - Transport -Multi-tissue
F_o(t+1) = \Sigma_o' W_oo' F_o'(t)

"""
Organ-level graph G=(O,E). Nodes: muscle, liver, brain, immune.
Edges: vascular/lymphatic. Enables muscle-brain, muscle-immune modeling.
"""

## Validation Metric

ExerkineScore = (1/N) \Sigma P_activation(c_i)