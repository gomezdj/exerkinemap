# ExerDynamix: EXERkine DYNamic Analysis, Multiomics Integration, and X-functional Clinical Mapping

# Introduction
ExerDynamix is an advanced computational framework designed to move beyond static mapping into the dynamics of exercise-induced molecular changes. By integrating multi-omics, single-cell, and spatial data, ExerDynamix uncovers the "how" and "when" of exerkine signaling—focusing on receptor affinity, dose-response relationships, and adaptive capacity to translate molecular knowledge into clinical applications like disease prevention and exercise mimetics.

# Table of Contents
1. [Key Features](#key-features)
2. [Installation](#installation)
3. [Quick Start Guide](#quick-start-guide)
4. **[Exerkine Dynamics & Clinical Translation](#spatial-mapping-data-analysis)**
   - [Receptor-Dependent Mechanisms](#receptor-dependent-mechanisms)
   - [Receptor-Independent Mechanisms](#receptor-independent-mechanisms)
   - [Dose-Response & Potency](#spatial-mapping)
   - [Adaptive Capacity Analysis](#adpative-capacity-analysis)
6. [Spatial & Single-Cell Integration](#spatial-single-cell-integration)
7. [Visualization & Benchmarking](#visualization-benchmarking)
8. [Contributing](#contributing)
9. [License](#license)
10. [Acknowledgements](#acknowledgements)

Key Features
* Dynamic Kinetic Modeling: Maps secretion triggers and distribution (autocrine/paracrine/endocrine) across tissues.

* Receptor-Specific Mapping: Identifies receptor localization and affinity to characterize downstream signaling cascades.

* Translational Pipeline: Designed to identify pharmacological targets for Exercise Mimetics.

* Multi-Consortia Integration: Harmonizes data from MoTrPAC, HuBMAP, HTAN, and PsychENCODE.

* Spatial-Temporal Trajectories: Infers molecular networks and cellular trajectories using CUDA-accelerated ML models.

# Installation 🧬
To install ExerDynamix, clone the repository and install the dependencies:
```
git clone https://github.com/gomezdj/ExerDynamix.git
cd ExerDynamix
pip install -r requirements.txt
```
Requirement: Python 3.9+ and CUDA-capable NVIDIA GPU (recommended for AMD Ryzen 9 9950X builds).

# Quick Start Guide
High-Level API
```
import exerdynamix as ed

# Load spatial and omics data
integrated_data = ed.load_and_integrate(transcriptomics="rna.csv", proteomics="protein.csv")

# Analyze Dynamics: Dose-Response & Receptor Affinity
dynamics_results = ed.analyze_dynamics(integrated_data, dose_column='intensity_min')

# Map to Clinical Targets (Mimetics)
targets = ed.identify_mimetics(dynamics_results, disease_context="neurological")

# Visualize the Exerkine Effect
ed.plot_dose_response(dynamics_results)
```

ONNEX Inference Pipeline
```
from exerdynamix.onnx_pipeline import ExerDynamixONNXPipeline

pipeline = ExerDynamixONNXPipeline(
    onnx_model_path="models/ExerDynamics_v1.onnx",
    apply_diffusion=True
)

# Predicting Exerkine effect based on spatial coordinates and receptor localization
results = pipeline.predict_from_spatial_data(
    spatial_coords=coords,
    G_expr=expression_matrix,
    ligands=['IL6', 'BDNF', 'CTNF'],
    receptors=['IL6R', 'NTRK2', 'LIFR']
)
```

# Exerkine Dynamics & Clinical Translation
- ExerDynamix transitions from "what is there" to "how it works" by focusing on:

- Receptor-Dependent Mechanisms: Distinguishing between receptor-mediated signaling and passive diffusion/EV-mediated transport.

- Dose-Response Relationship: Calculating exerkine potency and efficacy to determine the "onset of action" across different tissues.

- Adaptive Capacity: Factoring in inter-individual variability (diet, sleep, health status) to predict clinical outcomes in cancer, cardiovascular, and neurological diseases.

# Spatial & Single-Cell Integration
By leveraging Xenium and Stereo-seq data, ExerDynamix provides:
Cell-type-specific Exerkine Mapping: Precise cellular sources and targets. 
Ligand-Receptor Interaction Profiling: High-resolution spatial communication networks.
Barrier Permeability Analysis: Modeling exerkine distribution across the blood-brain barrier.

# Visualization
ExerDynamix includes interactive tools optimized for complex dynamics:
Kinetic Curves: Plasma concentration vs. Time ($AUC$).
Sigmoidal Response Plots: Tissue response (%) vs. Dose ($log_{10}$ scale).
Spatial Heatmaps: Visualizing the "Exerkine Infusion" effect across 3D tissue architectures.

# License
ExerDynamix is licensed under the MIT License.

# Acknowledgements
Special thanks to the Snyder Lab, the CSUEB & Stanford collaborative teams, and the developers of Scanpy, Squidpy, and PyTorch.

