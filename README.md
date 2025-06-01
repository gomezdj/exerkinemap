# EXERKINEMAPS: EXERcise KINEmatics Multiomics single-cell Analysis and spatial omics integration and maPping Schematics

## ExerkineMaps
ExerkineMaps is an advanced computational framework designed for the integration, analysis, and visualization of exercise-induced molecular changes. It combines multiomics data, single-cell analyses, and spatial data to provide comprehensive insights into the effects of exercise on various biological systems. By leveraging advanced machine learning techniques and state-of-the-art bioinformatics tools, ExerkineMap aims to unravel the complex molecular networks and cellular interactions modulated by physical activity.

# Table of Contents
1. [Introduction](#introduction)
2. [Key Features](#key-features)
3. [Installation](#installation)
4. [Quick Start Guide](#quick-start-guide)
5. **[Data Integration and Analysis](#data-integration-and-analysis)**
   - [Exerkine Identification](#exerkine-identification)
   - [Trajectory Inference and Network Analysis](#trajectory-inference-and-network-analysis)
   - [Spatial Mapping](#spatial-mapping)
   - [Single-Cell Analysis](#single-cell-analysis)
6. [Visualization](#visualization)
7. [Validation](#validation)
8. [Benchmarking and Performance](#benchmarking-and-performance)
9. [Contributing](#contributing)
10. [License](#license)
11. [Acknowledgements](#acknowledgements)

# Introduction
Exerkines are biomolecules released during physical exercise that mediate various physiological responses and health benefits. ExerkineMap integrates data from multiple omics domains (transcriptomics, proteomics, metabolomics, etc.), single-cell technologies, and spatial omics techniques to provide a holistic view of how exercise influences molecular and cellular processes.

# Key Features
* Comprehensive Data Integration: Harmonizes multiomics, single-cell, and spatial data.
* Exerkine Profiling: Identifies and characterizes exerkines and their regulatory mechanisms.
* Trajectory and Network Inference: Applies advanced methods to infer cellular trajectories and molecular networks.
* Spatial Analysis: Maps the spatial distribution of cells and molecules within tissues affected by exercise.
* Interactive Visualization: Provides tools for visualizing data through heatmaps, scatter plots, UMAP plots, and more.
* Machine Learning Integration: Utilizes state-of-the-art ML models for enhanced data analysis.
* Robust Validation and Benchmarking: Ensures accuracy and reliability through rigorous validation and benchmarking.

# Installation
To install ExerkineMap, clone the repository and install the required dependencies:
```
git clone https://github.com/username/ExerkineMap.git
cd ExerkineMap
pip install -r requirements.txt
```
Ensure you have Python 3.8 or higher.

# Quick Start Guide
1. Load Your Data: Ensure your multiomics, single-cell, and spatial data are formatted correctly.
2. Run Data Integration: Use the provided scripts to integrate your data.
3. Analyze and Visualize: Leverage ExerkineMap’s functionalities to perform analysis and generate visualizations.

Example Usage
```
import exerkine_map as em

# Load and preprocess your data
transcriptomics_data = em.load_data("transcriptomics.csv")
proteomics_data = em.load_data("proteomics.csv")
spatial_data = em.load_spatial_data("spatial_data.csv")

# Integrate multiomics data
integrated_data = em.integrate_data([transcriptomics_data, proteomics_data])

# Run analysis
exerkine_profiles = em.identify_exerkines(integrated_data)
trajectories = em.infer_trajectories(integrated_data)
spatial_map = em.map_spatial_relationships(spatial_data, integrated_data)

# Generate visualizations
em.plot_heatmap(exerkine_profiles)
em.plot_umap(trajectories)
em.plot_spatial_map(spatial_map)
```

# Data Integration and Analysis
# Exerkine Identification
ExerkineMap identifies and characterizes exerkines—molecules released during exercise that mediate physiological effects. It uses differential expression analysis and network-based approaches to pinpoint key exerkines and their regulatory pathways.

# Trajectory Inference and Network Analysis
ExerkineMap utilizes scVelo for RNA velocity analysis and trajectory inference. This involves the following steps:
* Preprocess the data to compute RNA velocity.
* Infer trajectories using the velocity information.
* Construct molecular networks to understand signaling pathways.

```
import scvelo as scv
import scanpy as sc

# Load the data into an AnnData object
adata = sc.read("integrated_data.h5ad")

# Preprocess for RNA velocity
scv.pp.filter_and_normalize(adata)
scv.pp.moments(adata)

# Compute RNA velocity
scv.tl.velocity(adata)
scv.tl.velocity_graph(adata)

# Visualize
scv.pl.velocity_embedding_stream(adata, basis='umap')

```

# Automated Cell Annotation (scRNA-seq data)
Cell type prediction (Python class `CellAssign`) for each cell is accomplish through a generative process that uses a neative binomial mixeture model to make cell-type predcitions. The inference is computed with the expectation maximization (EM) to optimize its parameters and provides a cell-type prediction for each cell. The logic implementing CellAssign can be found in [`scvi.external.cellassign.CellAssignModule`]. The implementation uses the same variable names as the math. The core logic is implemented in `scvi.external.cellassign.CellAssignModule.generative()`. In this method, the E step is taken and the log likelihood is computed for all cell types. In `scvi.external.cellassign.CellAssignModule.loss()` the full expected log likelihood is computed, as well as as the penalities corresponding to th epriors on pi and delta. CellAssign uses the standard `TrainingPlan`. 

# Spatial Mapping
By integrating spatial transcriptomics and proteomics data, ExerkineMap provides insights into the spatial organization of cells within tissues. This helps in understanding how exercise-induced changes alter tissue architecture and cell positioning.

# Single-Cell Analysis
ExerkineMap leverages single-cell RNA sequencing and other single-cell technologies to explore cellular heterogeneity and identify cell-type-specific responses to exercise. It incorporates tools like Seurat and Scanpy for clustering, differential expression, and functional analysis.

## 🧬 Single-cell Integration

EXERKINEMAP integrates single-cell RNA sequencing data to achieve cell-type-specific resolution, identifying precise cellular sources, receptors, signaling cascades, and communication networks involved in exerkine responses.

### Key Single-cell Functionalities:

- **Cell-type-specific Exerkine Mapping**
- **Ligand-Receptor Interaction Profiling**
- **Cell-cell Communication Analysis**
- **Spatial Transcriptomics Integration**

Explore these analyses in the [`notebooks/single-cell-analysis`](notebooks/single-cell-analysis/) directory.

# Visualization
ExerkineMap includes interactive visualization tools to explore and interpret data:
Heatmaps: For differential expression analysis.
UMAP Plots: For visualizing cellular trajectories and clustering.
Network Diagrams: To illustrate molecular interactions and regulatory networks.
Spatial Maps: To depict the spatial distribution of cells and molecules.

# Validation
ExerkineMap ensures the accuracy and reliability of analyses through rigorous validation methodologies, including cross-validation, permutation tests, and experimental verification (e.g., flow cytometry, immunohistochemistry).

# Benchmarking and Performance
ExerkineMap includes benchmarking tools to compare its performance with existing frameworks. Key metrics include accuracy, F1 score, ARI, and computational efficiency.

# Contributing
We welcome contributions from the community. To contribute:
Fork the repository.
Create a new branch for your feature or bugfix.
Commit your changes and submit a pull request.

# License
ExerkineMap is licensed under the MIT License. See LICENSE for more details.

# Acknowledgements
We thank the developers of the libraries and frameworks utilized in ExerkineMap, including STELLAR, MaxFuse, SPACEc, TensorFlow, PyTorch, Seurat, Scanpy, and others. Special thanks to the scientific community for providing valuable datasets and insights.

