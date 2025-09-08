# EXERKINEMAP: EXERcise KINEmatics Multiomics single-cell Analysis and spatial omics integration and maPping tissue schematics

## ExerkineMap
ExerkineMap is an advanced computational framework designed for the integration, analysis, and visualization of exercise-induced molecular changes. It combines multiomics data, single-cell analyses, and spatial data to provide comprehensive insights into the effects of exercise on various biological systems. By leveraging advanced machine learning techniques and state-of-the-art bioinformatics tools, ExerkineMaps aims to unravel the complex molecular networks and cellular interactions modulated by physical activity.

# Table of Contents
1. [Introduction](#introduction)
2. [Key Features](#key-features)
3. [Installation](#installation)
4. [Quick Start Guide](#quick-start-guide)
5. **[MoTrPAC and HuBMAP Spatial Mapping, Data Integration, and Analysis](#spatial-mapping-data-integration-and-analysis)**
   - [Exerkine Identification](#exerkine-identification)
   - [Spatial Mapping](#spatial-mapping)
   - [Single-Cell Spatial Proteomics Analysis](#single-cell-analysis)
6. [Visualization](#visualization)
7. [Validation](#validation)
8. [Benchmarking and Performance](#benchmarking-and-performance)
9. [Contributing](#contributing)
10. [License](#license)
11. [Acknowledgements](#acknowledgements)

# Introduction 
Exerkines are biomolecules released during physical exercise that mediate various physiological responses and health benefits. ExerkineMaps integrates data from MoTrPAC transcriptomics, global proteomics, and HuBMAP spatial omics technologies techniques to provide a detailed map of how exercise influences molecular and cellular processes.

# Key Features
* Comprehensive Data Integration: Harmonizes multiomics, single-cell, and spatial data.
* Exerkine Profiling: Identifies and characterizes exerkines and their regulatory mechanisms.
* Trajectory and Network Inference: Applies advanced methods to infer cellular trajectories and molecular networks.
* Spatial Analysis: Maps the spatial distribution of cells and molecules within tissues affected by exercise.
* Interactive Visualization: Provides tools for visualizing data through heatmaps, scatter plots, UMAP plots, and more.
* Machine Learning Integration: Utilizes state-of-the-art ML models for enhanced data analysis.
* Robust Validation and Benchmarking: Ensures accuracy and reliability through rigorous validation and benchmarking.

# Installation 🧬
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
import exerkinemap as em

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


```


# Spatial Mapping
By integrating spatial transcriptomics and proteomics data, ExerkineMap provides insights into the spatial organization of cells within tissues. This helps in understanding how exercise-induced changes alter tissue architecture and cell positioning.


##  Single-cell Integration

EXERKINEMAPS integrates single-cell RNA sequencing data to achieve cell-type-specific resolution, identifying precise cellular sources, receptors, signaling cascades, and communication networks involved in exerkine responses.

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
We thank the developers of the libraries and frameworks utilized in ExerkineMaps, including STELLAR, MaxFuse, SPACEc, TensorFlow, PyTorch, Seurat, Scanpy, and others. Special thanks to the scientific community for providing valuable datasets and insights.

