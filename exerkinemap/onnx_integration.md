# EXERKINEMAP ONNX Integration

## Overview

This module integrates ONNX models into EXERKINEMAP for classifying exercise-induced spatial ligand-receptor interaction (LRI) patterns. It implements the complete mathematical framework for:

1. **Initial exerkine profile computation**: f₀(i)
2. **Signal diffusion modeling**: F(t) = e^(-tL) · f₀
3. **LRI interaction scoring**: S_ij^(l_k, r_m) and P_activation
4. **Image tensor encoding**: 28×28 representations
5. **Exercise state classification**: 10-class ONNX inference

## Installation

```bash
# Install ONNX integration dependencies
pip install -r requirements_onnx.txt

# Or install individual packages
pip install onnxruntime scipy scikit-learn numpy
```

## Quick Start

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
    spatial_coords=coordinates,
    G_expr=expression_matrix,
    adjacency=adjacency_matrix,
    ligands=['IL6', 'IL15', 'BDNF', 'VEGFA'],
    receptors=['IL6R', 'IL15RA', 'NTRK2', 'FLT1'],
    gene_names=all_genes
)

print(f"Exercise State: {results['class_label']}")
print(f"Confidence: {results['probabilities'].max():.3f}")
```

## Architecture

```
┌─────────────────────────────────────┐
│   Spatial Single-Cell Data          │
│   - Coordinates (x, y)               │
│   - Gene Expression (N × G)          │
│   - Cell-Cell Graph                  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   LRI Score Computation              │
│   - f₀(i) exerkine profile          │
│   - S_ij interaction scores          │
│   - P_activation pathway scoring     │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   Signal Diffusion (Optional)        │
│   - F(t) = e^(-tL) · f₀             │
│   - Graph Laplacian dynamics         │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   Image Encoding (28×28)             │
│   - Spatial heatmap                  │
│   - Adjacency matrix                 │
│   - Multi-channel composite          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   ONNX Model Inference               │
│   - CNN architecture                 │
│   - 10-class softmax output          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   Exercise State Classification      │
│   - Baseline / Acute / Chronic       │
│   - Recovery / Stress / Remodeling   │
└─────────────────────────────────────┘
```

## Module Structure

```
exerkinemap/
├── onnx_classifier.py          # ONNX model wrapper & inference
├── lri_image_encoder.py        # LRI data → 28×28 image encoding
├── cell_lri_scores.py          # LRI score computation (S_ij, P_activation)
├── signal_diffusion.py         # Graph Laplacian diffusion (F(t))
└── onnx_pipeline.py            # End-to-end integration pipeline

examples/
└── onnx_integration_example.py # Usage examples

tests/
└── test_onnx_integration.py    # Unit tests

docs/
└── ONNX_INTEGRATION_GUIDE.md   # Comprehensive documentation
```

## Key Features

### 1. Mathematical Framework Implementation

- **Initial Exerkine Profile**:
  ```
  f₀(i) = (x_i^exerkine + Σ_{j∈N_i} w_ij · x_j^exerkine) / (1 + Σ_{j∈N_i} w_ij)
  ```

- **Signal Diffusion**:
  ```
  F(t) = e^(-tL) · f₀
  where L is the graph Laplacian
  ```

- **LRI Interaction Scores**:
  ```
  S_ij^(l_k, r_m) = x_i(l_k) · x_j(r_m)
  P_activation(c_i | c_j) = Σ δ_{l_k∈E} · S_ij^(l_k, r_m) · β_{r_m}
  ```

- **Spatial Distance Kernel**:
  ```
  D_ij = exp(-||s_i - s_j||² / 2σ²)
  ω_ij^spatial = D_ij · ω_ij
  ```

### 2. Multiple Encoding Strategies

- **Spatial Heatmap**: Interpolate LRI scores onto 28×28 grid
- **Adjacency Matrix**: Encode graph structure directly
- **Multi-Channel Composite**: Combine multiple LRI features
- **Distance-Weighted**: Emphasize spatial proximity

### 3. Exercise State Classes

| Class | State | Biology |
|-------|-------|---------|
| 0 | Sedentary Baseline | No stimulus |
| 1 | Acute Endurance Response | IL-6, IL-15 signaling |
| 2 | Acute Resistance Response | IGF-1, FGF2 pathways |
| 3 | Chronic Aerobic Adaptation | VEGF, BDNF upregulation |
| 4 | Chronic Strength Adaptation | Myokine crosstalk |
| 5 | Recovery Phase | Anti-inflammatory |
| 6 | Metabolic Stress | Lactate, ROS signaling |
| 7 | Immune Activation | Cytokine response |
| 8 | Angiogenic Remodeling | Endothelial signaling |
| 9 | Neuromuscular Plasticity | BDNF, neurotrophins |

### 4. Multi-Organ Analysis

```python
organ_data = {
    'muscle': {'spatial_coords': ..., 'G_expr': ..., 'adjacency': ...},
    'liver': {...},
    'adipose': {...}
}

results = pipeline.predict_multi_organ(
    organ_data=organ_data,
    ligands=ligands,
    receptors=receptors,
    gene_names=gene_names
)
```

### 5. Temporal Dynamics

```python
temporal_results = pipeline.analyze_temporal_dynamics(
    spatial_coords=coords,
    G_expr=expr,
    adjacency=adj,
    ligands=ligands,
    receptors=receptors,
    gene_names=genes,
    time_points=[0.1, 0.5, 1.0, 2.0, 5.0]
)
```

## Examples

### Basic Prediction

```python
from exerkinemap.onnx_pipeline import ExerkineMapONNXPipeline
import numpy as np

# Load data (from AnnData, H5AD, etc.)
spatial_coords = adata.obsm['spatial']
G_expr = adata.X.toarray()
gene_names = adata.var_names.tolist()

# Build spatial graph
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=11).fit(spatial_coords)
distances, indices = nbrs.kneighbors(spatial_coords)

N = spatial_coords.shape[0]
adjacency = np.zeros((N, N))
for i in range(N):
    for j in indices[i, 1:]:
        adjacency[i, j] = 1

# Define L-R pairs
ligands = ['IL6', 'IL15', 'VEGFA', 'BDNF']
receptors = ['IL6R', 'IL15RA', 'FLT1', 'NTRK2']
exerkines = ['IL6', 'IL15', 'BDNF']

# Initialize & predict
pipeline = ExerkineMapONNXPipeline(
    onnx_model_path="models/ExerciseTrainingModel.onnx"
)

results = pipeline.predict_from_spatial_data(
    spatial_coords=spatial_coords,
    G_expr=G_expr,
    adjacency=adjacency,
    ligands=ligands,
    receptors=receptors,
    gene_names=gene_names,
    exerkines=exerkines
)

print(f"Exercise State: {results['class_label']}")
```

### Pathway Impact Scores

```python
# Define from KEGG, Reactome, STRING, etc.
beta_pathway = {
    'IL6R': 1.5,      # JAK/STAT, MAPK
    'IL15RA': 1.3,    # PI3K/AKT, mTOR
    'NTRK2': 2.0,     # BDNF → neuroplasticity
    'IGF1R': 1.8,     # Growth signaling
    'FLT1': 1.2,      # VEGF pathway
}

results = pipeline.predict_from_spatial_data(
    ...,
    beta_pathway=beta_pathway
)
```

## Testing

```bash
# Run all tests
pytest tests/test_onnx_integration.py -v

# Run specific test
pytest tests/test_onnx_integration.py::TestLRIImageEncoder -v

# With coverage
pytest tests/test_onnx_integration.py --cov=exerkinemap --cov-report=html
```

## Documentation

- **Integration Guide**: `docs/ONNX_INTEGRATION_GUIDE.md`
- **Examples**: `examples/onnx_integration_example.py`
- **API Reference**: Run `sphinx-build -b html docs/ docs/_build/`

## Performance

### Optimization Tips

1. **Sparse Operations** (for large N):
   ```python
   F_t = compute_diffused_signal(adjacency, f_0, t=2.0, sparse=True)
   ```

2. **GPU Acceleration**:
   ```python
   classifier = ExerkineONNXClassifier(
       model_path="model.onnx",
       providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
   )
   ```

3. **Batch Processing**:
   ```python
   results = pipeline.predict_batch(
       spatial_coords_list=[...],
       G_expr_list=[...],
       adjacency_list=[...]
   )
   ```

## Citation

If you use this integration in your research:

```bibtex
@software{exerkinemap_onnx_2026,
  title={EXERKINEMAP: Exercise Kinematics Multiomics Analysis with ONNX Integration},
  author={Gomez, D.J. and Contributors},
  year={2026},
  url={https://github.com/gomezdj/exerkinemap}
}
```

## Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

See `LICENSE.txt` for details.

## Support

- **Issues**: https://github.com/gomezdj/exerkinemap/issues
- **Discussions**: https://github.com/gomezdj/exerkinemap/discussions
- **Email**: [your email]

## Acknowledgments

This work builds on:
- CellPhoneDB, LIANA for L-R databases
- Scanpy, Squidpy for spatial analysis
- ONNX Runtime for model inference
- Exercise biology research from multiple labs
