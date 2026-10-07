# EXERKINEMAP Mathematical Model

## 1. Overview

EXERKINEMAP models the exercise-responsive molecular communication system across single cells, spatial tissue organization, and molecular sequence space.

The framework integrates four information domains:

[
\boxed{
\text{Sequence}
+
\text{Single-Cell Omics}
+
\text{Spatial Omics}
+
\text{Exercise Response}
}
]

into a unified representation of **exerkines, ligands, receptors, cell–cell communication, signal propagation, and pathway activation**.

The model contains two molecular language-model branches:

* **GLM** — Genomic Language Model for DNA/RNA sequence representation.
* **PLM** — Protein Language Model for amino-acid sequence representation.

These molecular representations are integrated with single-cell expression and spatial information to construct the EXERKINEMAP communication graph.

---

# 2. Problem Definition

Given a multimodal biological dataset

[
\mathcal{D}
===========

\left{
\mathbf{A},
\mathbf{X},
\mathbf{S},
\mathbf{Y},
\mathcal{LR}
\right},
]

where:

* (\mathbf{A}) = nucleotide and protein sequences,
* (\mathbf{X}) = single-cell molecular expression matrix,
* (\mathbf{S}) = spatial coordinates,
* (\mathbf{Y}) = exercise conditions and biological phenotypes,
* (\mathcal{LR}) = ligand–receptor interaction database,

EXERKINEMAP seeks to learn:

[
\boxed{
\mathcal{F}*{EX}
:
\mathcal{D}
\rightarrow
\left{
E,
\mathbf{Z},
\mathcal{G}*{E},
\mathbf{F}(t),
\mathbf{A}_{P}
\right}
}
]

where:

* (E) = predicted exercise-responsive exerkines,
* (\mathbf{Z}) = molecular representations,
* (\mathcal{G}_{E}) = exerkine communication network,
* (\mathbf{F}(t)) = propagated exerkine signal,
* (\mathbf{A}_{P}) = pathway activation state.

The inverse problem is:

[
\boxed{
\left(
\mathbf{X},
\mathbf{S},
\mathbf{Y},
\mathcal{G}_{E}
\right)
\rightarrow
\left(
\hat{\mathbf{A}}^{N},
\hat{\mathbf{A}}^{P}
\right)
}
]

where (\hat{\mathbf{A}}^{N}) and (\hat{\mathbf{A}}^{P}) denote candidate nucleotide/RNA and protein sequences.

---

# 3. Mathematical Notation

## 3.1 Cellular Space

Let

[
C
=

\left{
c_1,c_2,\ldots,c_N
\right}
]

denote the set of (N) cells or spatially resolved cellular units.

Each cell is represented as:

[
c_i
===

\left(
\mathbf{x}_i,
\mathbf{s}_i,
\tau_i
\right),
]

where:

* (\mathbf{x}_i) = molecular expression profile,
* (\mathbf{s}_i\in\mathbb{R}^{d}) = spatial coordinate,
* (\tau_i) = cell type or cellular state.

The single-cell expression matrix is:

[
\mathbf{X}
\in
\mathbb{R}^{N\times G},
]

where (G) denotes the number of genes or molecular features.

---

## 3.2 Ligands and Receptors

Let

[
L
=

\left{
l_1,l_2,\ldots,l_K
\right}
]

denote candidate ligands.

Let

[
R
=

\left{
r_1,r_2,\ldots,r_M
\right}
]

denote receptors.

The known or predicted ligand–receptor interaction space is:

[
\mathcal{LR}
\subseteq
L\times R.
]

Exercise-responsive ligands are defined as:

[
E\subseteq L.
]

Thus:

[
l_k\in E
]

indicates that ligand (l_k) is classified as an exercise-responsive molecular signal, or **exerkine**.

---

# 4. Molecular Sequence Space

EXERKINEMAP represents both nucleotide and protein sequences.

## 4.1 Nucleotide Sequences

For molecule (q), define:

[
\mathbf{a}^{N}_q
================

\left(
a_1,a_2,\ldots,a_{n_q}
\right),
]

where:

[
a_i
\in
\mathcal{A}_{N},
]

and:

[
\mathcal{A}_{N}
===============

{A,C,G,T,U}.
]

The nucleotide sequence may represent genomic DNA, transcript RNA, regulatory sequence, or other nucleic-acid sequence.

---

## 4.2 Protein Sequences

For protein (q):

[
\mathbf{a}^{P}_q
================

\left(
a_1,a_2,\ldots,a_{m_q}
\right),
]

where:

[
a_i
\in
\mathcal{A}_{P}.
]

The canonical amino-acid alphabet is:

[
\mathcal{A}_{P}
===============

{
A,R,N,D,C,Q,E,G,H,I,
L,K,M,F,P,S,T,W,Y,V
}.
]

---

# 5. Definition 1 — Exercise-Responsive Molecular Representation

For each molecular entity (q), EXERKINEMAP generates contextual sequence representations using genomic and protein language models.

## 5.1 Genomic Language Model

Let:

[
\mathcal{T}*{N}
:
\mathcal{A}*{N}^{*}
\rightarrow
\mathcal{V}_{N}^{*}
]

denote the nucleotide/RNA tokenizer.

The tokenized sequence is:

[
\mathbf{t}^{N}_q
================

\mathcal{T}_{N}
\left(
\mathbf{a}^{N}_q
\right).
]

The GLM representation is:

[
\boxed{
\mathbf{z}^{N}_q
================

f_{\theta_N}
\left(
\mathbf{t}^{N}_q
\right)
}
]

where (f_{\theta_N}) is a BERT-style genomic language-model encoder.

---

## 5.2 Protein Language Model

Let:

[
\mathcal{T}*{P}
:
\mathcal{A}*{P}^{*}
\rightarrow
\mathcal{V}_{P}^{*}
]

denote the protein tokenizer.

The tokenized protein sequence is:

[
\mathbf{t}^{P}_q
================

\mathcal{T}_{P}
\left(
\mathbf{a}^{P}_q
\right).
]

The PLM representation is:

[
\boxed{
\mathbf{z}^{P}_q
================

f_{\theta_P}
\left(
\mathbf{t}^{P}_q
\right)
}
]

where (f_{\theta_P}) represents the protein language model.

Candidate implementations include:

* AminoBERT for contextual protein representation.
* ProGen2 for generative protein sequence modeling.

---

## 5.3 Multimodal Molecular Representation

The nucleotide and protein representations are combined:

[
\boxed{
\mathbf{z}_q
============

\Phi
\left(
\mathbf{z}^{N}_q,
\mathbf{z}^{P}_q
\right)
}
]

where (\Phi) is a multimodal fusion function.

A simple implementation is:

[
\mathbf{z}_q
============

W_N\mathbf{z}^{N}_q
+
W_P\mathbf{z}^{P}_q.
]

---

## 5.4 Exercise-Response Score

Define:

[
\rho_q
======

P(E_q=1
\mid
\mathbf{z}_q,
\mathbf{X},
\mathbf{Y}).
]

The predicted exerkine set is:

[
\boxed{
E
=

\left{
l_k\in L:
\rho_{l_k}>\theta_E
\right}.
}
]

This definition integrates sequence representation with observed exercise-responsive molecular behavior.

---

# 6. Sequence Tokenization and Representation

## 6.1 Multi-Scale Tokenization

EXERKINEMAP supports multiple genomic tokenization strategies:

[
\mathcal{T}_{N}
===============

{
\mathcal{T}*{char},
\mathcal{T}*{kmer},
\mathcal{T}_{BPE}
}.
]

### Character-level tokens

[
\mathcal{V}_{char}
==================

{A,C,G,T}.
]

### (k)-mer tokens

[
t_i
===

a_i a_{i+1}\ldots a_{i+k-1}.
]

### BPE/subword tokens

Starting from:

[
\mathcal{V}^{(0)}
=================

{A,C,G,T},
]

the most frequent neighboring pair is identified:

[
(u,v)^*
=======

\arg\max_{(u,v)}
\operatorname{Freq}(u,v),
]

and merged:

[
u\oplus v
\rightarrow
uv.
]

After (M) merges:

[
\boxed{
\mathcal{V}_{BPE}
=================

\mathcal{V}^{(M)}.
}
]

---

# 7. CBOW Genomic Embeddings

Before contextual BERT encoding, genomic tokens may be represented using CBOW.

For token (t_i), define the context:

[
\mathcal{C}_i
=============

\left{
t_{i-w},\ldots,t_{i-1},
t_{i+1},\ldots,t_{i+w}
\right}.
]

For (w=2):

[
\mathcal{C}_i
=============

{
t_{i-2},t_{i-1},t_{i+1},t_{i+2}
}.
]

CBOW minimizes:

[
\boxed{
\mathcal{L}_{CBOW}
==================

*

\sum_i
\log
P
\left(
t_i
\mid
\mathcal{C}_i
\right).
}
]

The resulting embedding:

[
\mathbf e_{t_i}
\in
\mathbb{R}^{d}
]

provides a contextual initialization for the GLM.

---

# 8. BERT-Style Genomic Language Model

The token embeddings are:

[
\mathbf{H}^{(0)}
================

\mathbf{E}_{CBOW}
+
\mathbf{P},
]

where (\mathbf P) represents positional information.

For Transformer layer (b):

[
\mathbf{H}^{(b)}
================

\operatorname{Transformer}_b
\left(
\mathbf{H}^{(b-1)}
\right).
]

The final representation is:

[
\boxed{
\mathbf{H}
==========

\mathbf{H}^{(B)}.
}
]

Self-attention is:

[
Q=HW_Q,
\qquad
K=HW_K,
\qquad
V=HW_V,
]

and:

[
\boxed{
\operatorname{Attention}(Q,K,V)
===============================

\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V.
}
]

---

# 9. Protein Language Model

For protein sequence representation:

[
\boxed{
\mathbf z^{P}_q
===============

f_{\theta_P}
\left(
\mathcal{T}_{P}
(\mathbf a_q^P)
\right).
}
]

A protein generative model such as ProGen2 defines:

[
\boxed{
P_{\mathrm{PLM}}
\left(
\mathbf a^P
\right)
=======

\prod_{t=1}^{m}
P
\left(
a_t
\mid
a_{<t}
\right).
}
]

This permits generation of candidate protein sequences:

[
\boxed{
\hat{\mathbf a}^{P}
\sim
P_{\mathrm{PLM}}
\left(
\mathbf a^{P}
\mid
\mathbf z_q,
\mathbf X,
\mathbf S,
\mathbf Y
\right).
}
]

---

# 10. Definition 2 — Spatial Ligand–Receptor Interaction

For sender cell (c_i), receiver cell (c_j), ligand (l_k), and receptor (r_m), define the base ligand–receptor interaction score:

[
\boxed{
S_{ij}^{(l_k,r_m)}
==================

x_i(l_k)
x_j(r_m).
}
]

This represents the expression-dependent interaction potential.

---

## 10.1 Molecular Compatibility

Let:

[
\Gamma_{km}
===========

g
\left(
\mathbf z_{l_k},
\mathbf z_{r_m}
\right)
]

represent molecular compatibility between ligand (l_k) and receptor (r_m).

The function (g) may incorporate:

* sequence similarity,
* learned molecular embeddings,
* known ligand–receptor relationships,
* protein language-model representations,
* biological priors.

---

## 10.2 Biological Prior

Let:

[
\alpha_{km}
]

denote the biological prior weight associated with ligand–receptor pair ((l_k,r_m)).

Examples include evidence derived from curated ligand–receptor databases or signaling resources.

---

## 10.3 Spatial Kernel

For cellular coordinates:

[
\mathbf s_i,\mathbf s_j
\in
\mathbb{R}^{d},
]

define:

[
\boxed{
K_{ij}^{S}
==========

\exp
\left(
-\frac{
|\mathbf s_i-\mathbf s_j|^2
}{
2\sigma_S^2
}
\right).
}
]

---

## 10.4 Spatially Informed Interaction

The complete interaction score becomes:

[
\boxed{
\widetilde S_{ij}^{(l_k,r_m)}
=============================

x_i(l_k)
x_j(r_m)
\alpha_{km}
\Gamma_{km}
K_{ij}^{S}.
}
]

This integrates:

[
\boxed{
\text{Expression}
+
\text{Molecular Compatibility}
+
\text{Biological Prior}
+
\text{Spatial Proximity}.
}
]

---

# 11. Definition 3 — EXERKINEMAP Exerkine Signaling Network

The EXERKINEMAP communication network is defined as:

[
\boxed{
\mathcal{G}_E
=============

(C,\mathcal{E}_E,\mathbf W_E).
}
]

The edge set is:

[
\mathcal{E}_E
=============

\left{
(c_i,c_j):
w_{ij}^{E}>0
\right}.
]

The exerkine-specific edge weight is:

[
\boxed{
w_{ij}^{E}
==========

\sum_{\substack{
(l_k,r_m)\in\mathcal{LR}\
l_k\in E
}}
\widetilde S_{ij}^{(l_k,r_m)}.
}
]

The adjacency matrix is:

[
\mathbf W_E
===========

[w_{ij}^{E}].
]

The corresponding degree matrix is:

[
D_{ii}
======

\sum_jw_{ij}^{E}.
]

The graph Laplacian is:

[
\boxed{
\mathcal{L}_E
=============

D-W_E.
}
]

---

# 12. Exerkine Secretion State

The initial exerkine secretion state of cell (c_i) is:

[
\boxed{
f_0(i)
======

\sum_{l_k\in E}
\rho_{l_k}
x_i(l_k).
}
]

Therefore:

[
\mathbf f_0
===========

\left[
f_0(1),
f_0(2),
\ldots,
f_0(N)
\right]^T.
]

---

# 13. Exerkine Signal Propagation

The exercise-responsive molecular signal propagates over the communication graph according to:

[
\boxed{
\mathbf F(t)
============

e^{-t\mathcal L_E}
\mathbf f_0.
}
]

where:

* (t) = propagation time or diffusion parameter,
* (\mathcal L_E) = EXERKINEMAP graph Laplacian,
* (\mathbf f_0) = initial exerkine secretion state,
* (\mathbf F(t)) = predicted signal distribution across cells.

This formulation is analogous to graph heat diffusion.

The model therefore represents:

[
\boxed{
\text{Exerkine secretion}
\rightarrow
\text{Cellular communication}
\rightarrow
\text{Spatial propagation}.
}
]

---

# 14. Pathway Activation

Let (P={p_1,\ldots,p_Q}) denote biological pathways.

Define:

[
\beta_{mp}
]

as the contribution of receptor (r_m) to pathway (p).

The pathway activation score in receiver cell (c_j) is:

[
\boxed{
A_j(p)
======

\sum_i
\sum_{\substack{
(l_k,r_m)\in\mathcal{LR}\
l_k\in E
}}
\widetilde S_{ij}^{(l_k,r_m)}
\beta_{mp}.
}
]

The pathway activation matrix is:

[
\mathbf A_P
\in
\mathbb R^{N\times Q}.
]

---

# 15. Interorgan Extension

Let:

[
O
=

{o_1,o_2,\ldots,o_H}
]

denote organs or tissues.

Each cell is assigned to an organ:

[
\omega:C\rightarrow O.
]

The cell-level communication graph can therefore be projected into an organ-level graph:

[
\boxed{
\mathcal G_O
============

(O,\mathcal E_O,\mathbf W_O).
}
]

The organ-level edge weight is:

[
\boxed{
W_{ab}^{O}
==========

\sum_{\substack{
i:\omega(c_i)=o_a\
j:\omega(c_j)=o_b
}}
w_{ij}^{E}.
}
]

This allows EXERKINEMAP to model:

[
\boxed{
\text{Cell}
\rightarrow
\text{Tissue}
\rightarrow
\text{Organ}
\rightarrow
\text{Interorgan communication}.
}
]

---

# 16. UMAP Integration

Let:

[
\mathbf Z
=========

[
\mathbf z_1,\ldots,\mathbf z_N
]
]

represent the integrated cellular molecular embeddings.

A nonlinear dimensionality reduction function:

[
\mathcal U:
\mathbb R^d
\rightarrow
\mathbb R^2
]

maps the high-dimensional representation to UMAP coordinates:

[
\boxed{
\mathbf u_i
===========

\mathcal U(\mathbf z_i).
}
]

The resulting coordinates can be used to visualize:

* cell type,
* exerkine expression,
* receptor expression,
* GLM embeddings,
* PLM embeddings,
* pathway activation,
* predicted communication strength.

---

# 17. Unified EXERKINEMAP Representation

The complete cellular representation is:

[
\boxed{
\mathbf h_i
===========

\Phi
\left(
\mathbf z_i^{N},
\mathbf z_i^{P},
\mathbf x_i,
\mathbf s_i,
\tau_i
\right).
}
]

Thus:

[
\boxed{
\text{Sequence}
+
\text{Protein}
+
\text{Cell}
+
\text{Space}
\rightarrow
\text{Unified Biological State}.
}
]

---

# 18. Complete EXERKINEMAP Pipeline

The mathematical workflow is:

[
\boxed{
\mathbf A
\rightarrow
\mathcal T
\rightarrow
\mathbf Z^{N},\mathbf Z^{P}
\rightarrow
\mathbf Z
}
]

followed by:

[
\boxed{
(\mathbf Z,\mathbf X,\mathbf S,\mathbf Y)
\rightarrow
E
\rightarrow
\mathcal G_E
\rightarrow
\mathbf F(t)
\rightarrow
\mathbf A_P.
}
]

The complete model is therefore:

[
\boxed{
\mathcal F_{EX}
:
(\mathbf A,\mathbf X,\mathbf S,\mathbf Y,\mathcal{LR})
\rightarrow
(E,\mathbf Z,\mathcal G_E,\mathbf F(t),\mathbf A_P).
}
]

---

# 19. Conceptual Architecture

```text
                         EXERCISE
                            │
                            ▼
              ┌─────────────────────────┐
              │ Reference Adaptation    │
              └────────────┬────────────┘
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
        DNA / RNA                    PROTEIN
             │                           │
             ▼                           ▼
       Tokenization                AA Tokenization
             │                           │
             ▼                           ▼
        CBOW / BPE                 Protein LM
             │                    ┌──────┴──────┐
             ▼                    │             │
        GLM / BERT            AminoBERT      ProGen2
             │                    │             │
             └────────────┬───────┘             │
                          ▼                     ▼
                  Molecular Embeddings    Candidate Sequences
                          │
                          ▼
              ┌─────────────────────────┐
              │ Single-Cell Omics       │
              │ Spatial Omics            │
              └────────────┬────────────┘
                           ▼
                  EXERKINE IDENTIFICATION
                           │
                           ▼
                 LIGAND–RECEPTOR MODEL
                           │
                           ▼
                  SPATIAL COMMUNICATION
                           │
                           ▼
                  GRAPH CONSTRUCTION
                           │
                           ▼
                  SIGNAL PROPAGATION
                           │
                           ▼
                   PATHWAY ACTIVATION
                           │
                           ▼
                  INTERORGAN NETWORK
```

---

# 20. Model Objective

The overall EXERKINEMAP objective can be represented as a weighted multimodal score:

[
\boxed{
\mathcal J(q)
=============

\lambda_N S_N(q)
+
\lambda_P S_P(q)
+
\lambda_E S_E(q)
+
\lambda_{LR}S_{LR}(q)
+
\lambda_{SP}S_{SP}(q)
+
\lambda_{SC}S_{SC}(q)
}
]

where:

* (S_N) = genomic/RNA sequence representation,
* (S_P) = protein sequence representation,
* (S_E) = exercise-response evidence,
* (S_{LR}) = ligand–receptor compatibility,
* (S_{SP}) = spatial communication,
* (S_{SC}) = single-cell molecular evidence,
* (\lambda_*) = model weighting parameters.

Candidate molecular entities can then be ranked according to:

[
\boxed{
q^*
===

\underset{q\in\mathcal Q}{\arg\max}
\mathcal J(q).
}
]

---

# 21. Summary

EXERKINEMAP establishes a unified mathematical framework in which:

[
\boxed{
\text{DNA/RNA}
\xrightarrow{\mathrm{GLM}}
\mathbf Z^N
}
]

[
\boxed{
\text{Protein}
\xrightarrow{\mathrm{PLM}}
\mathbf Z^P
}
]

[
\boxed{
(\mathbf Z^N,\mathbf Z^P,\mathbf X,\mathbf S)
\rightarrow
\mathcal G_E
}
]

and:

[
\boxed{
\mathcal G_E
\rightarrow
\mathbf F(t)
\rightarrow
\mathbf A_P.
}
]

The resulting framework connects **molecular sequence language, cellular expression, spatial organization, ligand–receptor signaling, graph diffusion, pathway activation, and interorgan communication** into a single computational representation of the exercise responsome.
