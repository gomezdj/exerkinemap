# EXERKINEMAP ONNX Integration Guide

## Overview

This guide explains how to integrate ONNX models into EXERKINEMAP for classifying exercise-induced spatial LRI patterns. The integration bridges your mathematical framework (LRI scores, signal diffusion, pathway analysis) with deep learning classification.

## Architecture

```
Spatial Single-Cell Data
         ↓
    LRI Computation (f_0, S_ij, P_activation)
         ↓
    Signal Diffusion (F(t) = e^(-tL) · f_0)
         ↓
    Image Encoding (28×28 tensor)
         ↓
    ONNX Model Inference
         ↓
    Exercise State Classification (10 classes)
```

## Installation

```bash
pip install onnxruntime scipy scikit-learn numpy
```

## Quick Start

### 1. Basic Prediction

```python
from exerkinemap.onnx_pipeline import ExerkineMapONNXPipeline

# Initialize pipeline
pipeline = ExerkineMapONNXPipeline(
    onnx_model_path="models/ExerciseTrainingModel.onnx",
    encoding_strategy='spatial_heatmap',
    apply_diffusion=True,
    diffusion_time=2.0
)

# Predict exercise state
results = pipeline.predict_from_spatial_data(
    spatial_coords=coords,      # Shape: (N, 2)
    G_expr=expression_matrix,   # Shape: (N, G)
    adjacency=adj_matrix,       # Shape: (N, N)
    ligands=['IL6', 'IL15', ...],
    receptors=['IL6R', 'IL15RA', ...],
    gene_names=all_genes,
    exerkines=['IL6', 'IL15', 'BDNF']
)

print(f"Exercise State: {results['class_label']}")
print(f"Confidence: {results['probabilities'].max():.3f}")
```

### 2. Multi-Organ Analysis

```python
# Prepare data for multiple organs
organ_data = {
    'muscle': {'spatial_coords': ..., 'G_expr': ..., 'adjacency': ...},
    'liver': {'spatial_coords': ..., 'G_expr': ..., 'adjacency': ...},
    'adipose': {'spatial_coords': ..., 'G_expr': ..., 'adjacency': ...}
}

# Analyze all organs
results = pipeline.predict_multi_organ(
    organ_data=organ_data,
    ligands=ligands,
    receptors=receptors,
    gene_names=gene_names
)

for organ, res in results.items():
    print(f"{organ}: {res['class_label']}")
```

### 3. Temporal Dynamics

```python
# Track how exercise response evolves over diffusion time
temporal_results = pipeline.analyze_temporal_dynamics(
    spatial_coords=coords,
    G_expr=expression_matrix,
    adjacency=adj_matrix,
    ligands=ligands,
    receptors=receptors,
    gene_names=gene_names,
    time_points=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# Visualize state transitions
for i, t in enumerate(temporal_results['time_points']):
    state = temporal_results['predicted_classes'][i]
    print(f"t={t}: {pipeline.get_class_labels()[state]}")
```

## Mathematical Framework Integration

### 1. Initial Exerkine Profile: f_0(i)

```python
from exerkinemap.cell_lri_scores import compute_initial_exerkine_profile

f_0 = compute_initial_exerkine_profile(
    G_expr=expression_matrix,
    adjacency=adj_matrix,
    exerkines=['IL6', 'IL15', 'BDNF', 'IGF1'],
    gene_names=gene_names,
    edge_weights=distance_weights  # Optional: w_ij = 1/(1 + d_ij)
)
```

**Formula:**
```
f_0(i) = (x_i^exerkine + Σ_{j∈N_i} w_ij · x_j^exerkine) / (1 + Σ_{j∈N_i} w_ij)

where x_i^exerkine = Σ_{l_k∈E} x_i(l_k)
```

### 2. Signal Diffusion: F(t)

```python
from exerkinemap.signal_diffusion import compute_diffused_signal

F_t = compute_diffused_signal(
    adjacency=adj_matrix,
    f_0=f_0,
    t=2.0  # Diffusion time parameter
)
```

**Formula:**
```
F(t) = e^(-tL) · f_0

where L is the graph Laplacian
```

**Interpretation:**
- `t=0.1-1.0`: Local paracrine signaling
- `t=1.0-5.0`: Regional diffusion
- `t=5.0-10.0`: Systemic endocrine effects

### 3. LRI Interaction Scores: S_ij

```python
from exerkinemap.cell_lri_scores import compute_cell_lri_scores

lri_scores = compute_cell_lri_scores(
    G_expr=expression_matrix,
    ligands=ligands,
    receptors=receptors,
    gene_names=gene_names,
    adjacency=adj_matrix,
    exerkines=exerkines,
    beta_pathway=pathway_impacts  # Receptor → pathway weight
)
```

**Formula:**
```
P_activation(c_i | c_j) = Σ_{(l_k, r_m)} δ_{l_k∈E} · S_ij^(l_k, r_m) · β_{r_m}

where:
- S_ij^(l_k, r_m) = x_i(l_k) · x_j(r_m)
- δ_{l_k∈E} = 1 if ligand is exerkine, else 0
- β_{r_m} = pathway impact score from databases
```

### 4. Spatial Distance Weighting: D_ij

```python
from exerkinemap.cell_lri_scores import apply_spatial_weighting

weighted_lri = apply_spatial_weighting(
    lri_matrix=lri_scores_matrix,
    spatial_coords=coords,
    sigma=10.0  # Spatial bandwidth
)
```

**Formula:**
```
ω_ij^spatial = D_ij · ω_ij

D_ij = exp(-||s_i - s_j||^2 / 2σ^2)
```

## Image Encoding Strategies

### 1. Spatial Heatmap (Recommended)

```python
from exerkinemap.lri_image_encoder import LRIImageEncoder

encoder = LRIImageEncoder(image_size=28)
image = encoder.spatial_heatmap_encoding(
    spatial_coords=coords,
    lri_scores=scores,
    sigma=1.0  # Gaussian smoothing
)
```

**Best for:** Visually interpretable spatial patterns

### 2. Adjacency Matrix

```python
image = encoder.adjacency_matrix_encoding(
    adjacency_matrix=adj_matrix,
    node_features=lri_scores  # Optional weighting
)
```

**Best for:** Network topology emphasis

### 3. Multi-Channel Composite

```python
lri_features = {
    'exerkine_signal': f_0,
    'receptor_activation': receptor_scores,
    'pathway_score': pathway_scores
}

image = encoder.multi_channel_composite_encoding(
    spatial_coords=coords,
    lri_feature_dict=lri_features,
    weights={'exerkine_signal': 0.4, 'receptor_activation': 0.3, 'pathway_score': 0.3}
)
```

**Best for:** Comprehensive multi-feature representation

### 4. Distance-Weighted

```python
image = encoder.distance_weighted_encoding(
    spatial_coords=coords,
    lri_scores=scores,
    distance_kernel='gaussian',
    kernel_sigma=10.0
)
```

**Best for:** Emphasizing spatial proximity

## Exercise State Classes

The ONNX model outputs 10 biologically meaningful classes:

| Class | Label | Description |
|-------|-------|-------------|
| 0 | Sedentary_Baseline | No exercise stimulus |
| 1 | Acute_Endurance_Response | IL-6, IL-15 signaling |
| 2 | Acute_Resistance_Response | IGF-1, FGF2 pathways |
| 3 | Chronic_Aerobic_Adaptation | VEGF, BDNF upregulation |
| 4 | Chronic_Strength_Adaptation | Myokine/hepatokine crosstalk |
| 5 | Recovery_Phase | Anti-inflammatory, tissue repair |
| 6 | Metabolic_Stress | Lactate, ROS signaling |
| 7 | Immune_Activation | Cytokine storm, inflammation |
| 8 | Angiogenic_Remodeling | Endothelial-muscle crosstalk |
| 9 | Neuromuscular_Plasticity | BDNF, neurotrophin signaling |

### Customizing Classes

```python
custom_classes = {
    0: "Your_Class_0",
    1: "Your_Class_1",
    # ... define your classes
}

classifier = ExerkineONNXClassifier(
    model_path="your_model.onnx",
    class_labels=custom_classes
)
```

## Pathway Impact Scores (β_r_m)

Define receptor-pathway relationships from databases:

```python
beta_pathway = {
    # Immune/Inflammatory
    'IL6R': 1.5,      # JAK/STAT, MAPK
    'IL15RA': 1.3,    # PI3K/AKT, mTOR
    'TNFRSF1A': 1.4,  # NF-κB
    
    # Growth/Angiogenesis
    'IGF1R': 1.8,     # PI3K/AKT, mTOR
    'FLT1': 1.2,      # VEGF pathway
    'TEK': 1.1,       # Angiopoietin
    
    # Neurotropic
    'NTRK2': 2.0,     # BDNF → neuroplasticity
    
    # Metabolic
    'INSR': 1.6,      # Insulin signaling
}
```

**Data sources:**
- KEGG Pathway Database
- Reactome
- STRING
- BioGRID
- PathBank

## Working with AnnData

For spatial transcriptomics in AnnData format:

```python
import scanpy as sc

# Load data
adata = sc.read_h5ad("spatial_data.h5ad")

# Extract components
spatial_coords = adata.obsm['spatial']
G_expr = adata.X.toarray()  # Convert sparse to dense
gene_names = adata.var_names.tolist()

# Build spatial graph
from sklearn.neighbors import NearestNeighbors
k = 10
nbrs = NearestNeighbors(n_neighbors=k+1).fit(spatial_coords)
distances, indices = nbrs.kneighbors(spatial_coords)

N = spatial_coords.shape[0]
adjacency = np.zeros((N, N))
edge_weights = np.zeros((N, N))

for i in range(N):
    for j_idx, j in enumerate(indices[i, 1:]):
        adjacency[i, j] = 1
        edge_weights[i, j] = 1.0 / (1.0 + distances[i, j_idx + 1])

# Run pipeline
results = pipeline.predict_from_spatial_data(
    spatial_coords=spatial_coords,
    G_expr=G_expr,
    adjacency=adjacency,
    ligands=your_ligands,
    receptors=your_receptors,
    gene_names=gene_names,
    edge_weights=edge_weights
)
```

## Performance Optimization

### For Large Datasets (N > 10,000 cells)

```python
# Use sparse operations
from exerkinemap.signal_diffusion import compute_diffused_signal

F_t = compute_diffused_signal(
    adjacency=adj_matrix,
    f_0=f_0,
    t=2.0,
    sparse=True  # Enable sparse matrix operations
)
```

### Batch Processing

```python
# Process multiple samples efficiently
results_list = pipeline.predict_batch(
    spatial_coords_list=[coords1, coords2, coords3],
    G_expr_list=[expr1, expr2, expr3],
    adjacency_list=[adj1, adj2, adj3],
    ligands=ligands,
    receptors=receptors,
    gene_names=gene_names
)
```

### GPU Acceleration

```python
# Use CUDA for ONNX inference
classifier = ExerkineONNXClassifier(
    model_path="model.onnx",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
```

## Validation & Interpretation

### Cross-Validation

```python
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True)

for train_idx, test_idx in kfold.split(spatial_coords):
    # Split data
    train_coords = spatial_coords[train_idx]
    test_coords = spatial_coords[test_idx]
    
    # Train/test workflow
    # ...
```

### Confidence Thresholding

```python
results = pipeline.predict_from_spatial_data(...)

confidence = results['probabilities'].max()

if confidence < 0.5:
    print("Low confidence prediction - manual review recommended")
elif confidence < 0.7:
    print("Moderate confidence")
else:
    print("High confidence prediction")
```

### Feature Importance

```python
# Analyze which LRI features drive classification
from exerkinemap.lri_image_encoder import LRIImageEncoder

encoder = LRIImageEncoder()

# Generate images with/without specific features
baseline_image = encoder.spatial_heatmap_encoding(coords, baseline_scores)
perturbed_image = encoder.spatial_heatmap_encoding(coords, perturbed_scores)

# Compare predictions
baseline_pred = pipeline.classifier.predict(baseline_image)
perturbed_pred = pipeline.classifier.predict(perturbed_image)

# Measure impact
impact = np.abs(baseline_pred[1] - perturbed_pred[1])
```

## Troubleshooting

### Common Issues

**1. Shape mismatch errors**
```python
# Ensure all arrays have correct dimensions
assert spatial_coords.shape == (N, 2)
assert G_expr.shape == (N, G)
assert adjacency.shape == (N, N)
```

**2. Missing genes**
```python
# Check gene availability
available_ligands = [l for l in ligands if l in gene_names]
available_receptors = [r for r in receptors if r in gene_names]

print(f"Using {len(available_ligands)}/{len(ligands)} ligands")
print(f"Using {len(available_receptors)}/{len(receptors)} receptors")
```

**3. Empty LRI scores**
```python
# Check expression thresholds
results = pipeline.predict_from_spatial_data(
    ...,
    expression_threshold=0.1  # Filter low expression
)
```

## Citation

If you use this integration in your research, please cite:

```bibtex
@software{exerkinemap_onnx,
  title={EXERKINEMAP ONNX Integration},
  author={Your Name},
  year={2026},
  url={https://github.com/gomezdj/exerkinemap}
}
```

## Support

For issues, questions, or feature requests:
- GitHub Issues: https://github.com/gomezdj/exerkinemap/issues
- Documentation: https://github.com/gomezdj/exerkinemap/wiki
