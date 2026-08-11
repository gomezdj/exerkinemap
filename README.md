# EXERKINEMAP

**EXERKINEMAP** (*EXERcise KINEmatics Multiomics single-cell Analysis and sPatial omics integration and maPping*) is a computational framework for mapping **exercise-responsive molecular signaling across cells, tissues, and organs**.

EXERKINEMAP integrates **single-cell omics, spatial omics, genomic and protein language models, molecular sequences, ligand–receptor biology, and graph-based signal propagation** to characterize the exercise responsome and identify candidate **exerkines, ligands, receptors, pathways, and intercellular communication networks**.

## Architecture

```text
                         EXERCISE RESPONSE
                                │
             ┌──────────────────┴──────────────────┐
             │                                     │
             ▼                                     ▼
      DNA / RNA SEQUENCES                    PROTEIN SEQUENCES
             │                                     │
             ▼                                     ▼
      ┌───────────────┐                    ┌────────────────┐
      │ GLM           │                    │ PLM            │
      │ Genomic       │                    │ Protein        │
      │ Language      │                    │ Language       │
      │ Model         │                    │ Model          │
      └───────┬───────┘                    └───────┬────────┘
              │                                    │
              ▼                                    ▼
      Genomic/RNA                           Protein/Amino-acid
      representations                       representations
              │                                    │
              └────────────────┬───────────────────┘
                               ▼
                    MULTIMODAL MOLECULAR
                       REPRESENTATION
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
   Single-Cell Omics      Spatial Omics      Exercise Phenotype
          │                    │                    │
          └────────────────────┼────────────────────┘
                               ▼
                    EXERKINE IDENTIFICATION
                               │
                               ▼
                  LIGAND–RECEPTOR NETWORK
                               │
                               ▼
                    SPATIAL COMMUNICATION
                               │
                               ▼
                    SIGNAL PROPAGATION
                               │
                               ▼
                     PATHWAY ACTIVATION
                               │
                               ▼
                   INTERORGAN RESPONSOME
```

## GLM — Genomic Language Model

The **GLM** represents DNA and RNA sequences as contextual biological language.

The sequence-processing pipeline is:

```text
DNA / RNA
    │
    ▼
Reference Adaptation
    │
    ▼
Multi-scale Tokenization
    │
    ├── Character tokens
    ├── k-mers
    └── BPE / subword tokens
    │
    ▼
CBOW / Word2Vec Embeddings
    │
    ▼
BERT-style Transformer Encoder
    │
    ▼
Contextual Genomic Representation
```

The GLM transforms a nucleotide sequence (\mathbf a_q^N) into a contextual representation:

[
\mathbf z_q^N
=============

f_{\theta_N}
\left(
\mathcal T_N(\mathbf a_q^N)
\right).
]

The representation can be used to characterize **regulatory sequence context, exercise-responsive genes, exerkine transcripts, and genomic relationships**.

## PLM — Protein Language Model

The **PLM** represents amino-acid sequences using protein language modeling.

EXERKINEMAP supports complementary protein-language-model functions:

* **AminoBERT** — contextual protein representation.
* **ProGen2** — generative protein sequence modeling.

The protein pipeline is:

```text
Protein Sequence
      │
      ▼
Amino-acid Tokenization
      │
      ▼
PLM
 ┌────┴───────────┐
 │                │
 ▼                ▼
AminoBERT       ProGen2
 │                │
 ▼                ▼
Protein          Candidate
Embedding        Protein Sequence
```

For a protein (q):

[
\mathbf z_q^P
=============

f_{\theta_P}
\left(
\mathcal T_P(\mathbf a_q^P)
\right).
]

ProGen2 provides a generative distribution:

[
P(\mathbf a_q^P)
================

\prod_{t=1}^{m}
P(a_t\mid a_{<t}),
]

enabling candidate protein-sequence generation conditioned on learned biological context.

## Multimodal Molecular Representation

GLM and PLM representations are integrated into a unified molecular representation:

[
\boxed{
\mathbf z_q
===========

\Phi
\left(
\mathbf z_q^N,
\mathbf z_q^P
\right)
}
]

where (\Phi) is a multimodal fusion function.

This creates a sequence-aware representation of exerkines and their receptors that can be integrated with cellular and spatial measurements.

## Single-Cell and Spatial Omics

EXERKINEMAP incorporates:

* scRNA-seq
* single-cell multiomics
* spatial transcriptomics
* spatial proteomics
* cell-type annotations
* ligand and receptor expression
* spatial coordinates
* exercise/control conditions

For sender cell (c_i) and receiver cell (c_j):

[
S_{ij}^{(l_k,r_m)}
==================

x_i(l_k)x_j(r_m).
]

Molecular compatibility and spatial proximity are incorporated into:

[
\widetilde S_{ij}^{(l_k,r_m)}
=============================

x_i(l_k)x_j(r_m)
\alpha_{km}
\Gamma_{km}
K_{ij}^{S}.
]

## Exerkine Communication Network

Exercise-responsive ligands are defined as:

[
E\subseteq L.
]

The resulting network is:

[
\mathcal G_E
============

(C,\mathcal E_E,\mathbf W_E),
]

where nodes represent cells or cellular populations and weighted directed edges represent predicted exerkine-mediated communication.

The graph Laplacian is:

[
\mathcal L_E=D-W_E.
]

Exercise-responsive signal propagation is modeled as:

[
\mathbf F(t)
============

e^{-t\mathcal L_E}\mathbf f_0.
]

This enables EXERKINEMAP to model the **spatial and temporal propagation of exercise-induced molecular signals**.

## Core Components

| Component                 | Function                                            |
| ------------------------- | --------------------------------------------------- |
| **GLM**                   | DNA/RNA sequence representation and genomic context |
| **CBOW / Word2Vec**       | Context-based genomic embeddings                    |
| **BERT Encoder**          | Contextual genomic representation                   |
| **PLM**                   | Protein sequence representation                     |
| **PROSE**                 | Protein sequence generation                         |
| **AminoBERT**             | Contextual amino-acid/protein embeddings            |
| **ProGen2**               | Generative protein sequence modeling                |
| **Single-cell Omics**     | Cellular expression and molecular state             |
| **Spatial Omics**         | Tissue localization and spatial context             |
| **Ligand–Receptor Model** | Cell–cell communication inference                   |
| **Graph Model**           | Communication network construction                  |
| **Graph Diffusion**       | Exerkine signal propagation                         |
| **Pathway Model**         | Downstream pathway activation                       |
| **Interorgan Mapping**    | Organ-to-organ exercise responsome                  |

## Research Objectives

EXERKINEMAP is designed to:

1. Identify exercise-responsive **exerkines, ligands, and receptors**.
2. Represent molecular sequences using **GLM and PLM embeddings**.
3. Integrate sequence information with **single-cell and spatial omics**.
4. Quantify ligand–receptor communication between cells.
5. Model spatial and temporal **exerkine signal propagation**.
6. Infer downstream pathway activation.
7. Construct **intraorgan and interorgan exercise-responsive communication maps**.
8. Explore candidate molecular sequences using generative language models.

## Applications

Potential applications include:

* Exercise physiology and the exercise responsome
* Cancer and tumor–host communication
* Aging and sarcopenia
* Metabolic disease
* Cardiovascular biology
* Neurobiology and neurodegeneration
* Placental biology
* Multi-organ systems biology

## Project Status

**Research / Development**

EXERKINEMAP is under active development. Current work focuses on integrating **GLM, PLM, single-cell, spatial omics, ligand–receptor inference, and graph-based modeling** into a scalable computational framework.

## Using EXERKINEMAP

### 1) Set up the environment
- Create and activate a Conda environment
- Install dependencies from requirements.txt

Example:
```bash
conda create -n exerkinemap python=3.11 -y
conda activate exerkinemap
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2) Prepare data
- Put raw inputs under raw
  - `data/raw/sequences/...`
  - `data/raw/metadata/...`
  - single-cell and spatial files as expected by the workflows

### 3) Run the preprocessing workflows
- Download and prepare reference data
```bash
python workflows/01_download_data.py
```
- Preprocess single-cell data
```bash
python workflows/02_preprocess_single_cell.py
```
- Preprocess spatial transcriptomics
```bash
python workflows/03_preprocess_spatial.py
```

### 4) Build sequence references
- Generate processed sequence references for GLM/PLM
```bash
python workflows/04_build_sequence_reference.py
```

### 5) Train tokenizer and GLM
- Train the BPE tokenizer and CBOW embeddings
```bash
python scripts/train_tokenizer.py
```
- Train the genomic language model
```bash
python scripts/train_glm.py
```

### 6) Build ligand-receptor database
```bash
python scripts/build_lr_database.py
```

### 7) Use the multimodal dataset
- The repository defines an `ExerkineDataset` container in dataset.py
- After preprocessing, multimodal data lives under:
  - `data/processed/sequences/`
  - `data/processed/...` for single-cell/spatial
  - models for tokenizers and trained embeddings

### 8) Evaluate or benchmark
- Run the benchmark summary
```bash
python scripts/benchmark.py
```

## Tutorial Notebooks
- 01-13


## Summary
- workflows prepares and processes data
- scripts trains models, builds references, and benchmarks
- dataset.py is the multimodal container concept for single-cell, spatial, sequence, metadata, and ligand-receptor data

Keep it simple: preprocess raw data, build sequence references, train the tokenizer/GLM, then use the resulting processed files and models for downstream analysis.

## License

See the repository license for terms of use.
