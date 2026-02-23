# EXERKINEMAP ONNX Integration - Complete File Structure

## New Files Added

```
exerkinemap/
├── onnx_classifier.py              ✅ NEW - ONNX model wrapper & inference engine
├── lri_image_encoder.py            ✅ NEW - LRI data → 28×28 image mapping
├── cell_lri_scores.py              ✅ NEW - Mathematical framework (f₀, S_ij, P_activation)
├── signal_diffusion.py             ✅ NEW - Graph Laplacian diffusion (F(t))
└── onnx_pipeline.py                ✅ NEW - End-to-end integration pipeline

examples/
└── onnx_integration_example.py     ✅ NEW - Comprehensive usage examples

tests/
└── test_onnx_integration.py        ✅ NEW - Unit tests for all modules

docs/
└── ONNX_INTEGRATION_GUIDE.md       ✅ NEW - Complete documentation

requirements_onnx.txt                ✅ NEW - ONNX integration dependencies
ONNX_INTEGRATION_README.md          ✅ NEW - Module overview & quick start
```

## Updated Project Structure

```
.
├── analysis/
│   ├── __init__.py
│   ├── interaction_scores.py
│   ├── pathways_analysis.py
│   └── tissue_specific_signaling.py
│
├── data/
│   ├── htan/
│   └── hubmap/
│       └── prepare_hubmap_lri_data.py
│
├── exerkinemap/
│   ├── __init__.py
│   ├── cell_lri_scores.py          ✅ NEW - LRI mathematical framework
│   ├── cluster.py
│   ├── cnn.py
│   ├── communication.py
│   ├── computational_steps.py
│   ├── datasets.py
│   ├── eb_classes.py
│   ├── embed.py
│   ├── exerkinemap.py
│   ├── gcn.py
│   ├── gnn.py
│   ├── graph.py
│   ├── implementation.py
│   ├── lri_image_encoder.py        ✅ NEW - Image encoding strategies
│   ├── map.py
│   ├── networks.py
│   ├── onnx_classifier.py          ✅ NEW - ONNX inference wrapper
│   ├── onnx_pipeline.py            ✅ NEW - Complete pipeline
│   ├── predictor.py
│   ├── signal_diffusion.py         ✅ NEW - Graph diffusion
│   ├── spatial_graph_lri.py
│   ├── spatial_omics_model.py
│   └── trainer.py
│
├── examples/                        ✅ NEW DIRECTORY
│   └── onnx_integration_example.py ✅ NEW - Usage examples
│
├── models/
│   └── spatial_lri_gnn.py
│
├── notebooks/
│   └── codex_spatial_proteomics_gnn.ipynb
│
├── tests/
│   ├── test_deep_learning.py
│   ├── test_dimensionality_reduction.py
│   ├── test_onnx_integration.py    ✅ NEW - ONNX tests
│   ├── test_preprocessing.py
│   ├── test_spatial_analysis.py
│   └── test_statistical_analysis.py
│
├── docs/                            ✅ NEW DIRECTORY
│   └── ONNX_INTEGRATION_GUIDE.md   ✅ NEW - Documentation
│
├── exerkinemap.yml
├── LICENSE.txt
├── README.md
├── ONNX_INTEGRATION_README.md      ✅ NEW - Module README
├── requirements.txt
├── requirements_onnx.txt            ✅ NEW - ONNX dependencies
└── setup.py
```

## Module Descriptions

### Core Modules

#### 1. `onnx_classifier.py`
**Purpose**: ONNX model wrapper for inference

**Key Classes**:
- `ExerkineONNXClassifier`: Main inference engine
- `EnsembleONNXClassifier`: Multi-model ensemble

**Key Functions**:
- `predict()`: Single image classification
- `predict_batch()`: Batch inference
- `get_top_k_predictions()`: Top-k results

**Features**:
- 10-class exercise state classification
- GPU acceleration support
- Ensemble predictions
- Confidence scoring

---

#### 2. `lri_image_encoder.py`
**Purpose**: Convert LRI data to 28×28 image tensors

**Key Class**:
- `LRIImageEncoder`: Multi-strategy encoding

**Encoding Strategies**:
1. **Spatial Heatmap**: Interpolate LRI scores onto grid
2. **Adjacency Matrix**: Encode graph structure
3. **Multi-Channel Composite**: Combine features
4. **Distance-Weighted**: Spatial proximity emphasis

**Key Functions**:
- `spatial_heatmap_encoding()`
- `adjacency_matrix_encoding()`
- `multi_channel_composite_encoding()`
- `distance_weighted_encoding()`
- `encode_time_series()`

---

#### 3. `cell_lri_scores.py`
**Purpose**: Implement mathematical framework for LRI computation

**Key Functions**:

```python
compute_initial_exerkine_profile()
# Formula: f₀(i) = (x_i^exerkine + Σ w_ij·x_j^exerkine) / (1 + Σ w_ij)

compute_cell_lri_scores()
# Formula: P_activation(c_i|c_j) = Σ δ·S_ij^(l_k,r_m)·β_{r_m}

compute_pairwise_lri_matrix()
# Formula: S_ij^(l_k,r_m) = x_i(l_k) · x_j(r_m)

compute_spatial_distance_kernel()
# Formula: D_ij = exp(-||s_i - s_j||² / 2σ²)

apply_spatial_weighting()
# Formula: ω_ij^spatial = D_ij · ω_ij
```

---

#### 4. `signal_diffusion.py`
**Purpose**: Model exerkine signal propagation via graph diffusion

**Key Functions**:

```python
compute_graph_laplacian()
# L = D - A (unnormalized) or L = I - D^(-1/2) A D^(-1/2) (normalized)

compute_diffused_signal()
# Formula: F(t) = e^(-tL) · f₀

compute_diffusion_time_series()
# F(t) at multiple timepoints

compute_steady_state_signal()
# Asymptotic behavior as t → ∞

analyze_signal_propagation_modes()
# Laplacian eigenanalysis
```

**Diffusion Time Interpretation**:
- `t = 0.1-1.0`: Paracrine (local) signaling
- `t = 1.0-5.0`: Regional diffusion
- `t = 5.0-10.0`: Endocrine (systemic) effects

---

#### 5. `onnx_pipeline.py`
**Purpose**: End-to-end integration pipeline

**Key Class**:
- `ExerkineMapONNXPipeline`: Complete workflow

**Key Methods**:
```python
predict_from_spatial_data()      # Single sample prediction
predict_batch()                  # Multiple samples
predict_multi_organ()            # Organ comparison
analyze_temporal_dynamics()      # Time evolution
```

**Workflow**:
1. Compute f₀(i) exerkine profile
2. Apply signal diffusion (optional)
3. Compute LRI scores
4. Encode to 28×28 image
5. ONNX inference
6. Return predictions + metadata

---

## Usage Workflows

### Workflow 1: Basic Prediction

```python
from exerkinemap.onnx_pipeline import ExerkineMapONNXPipeline

pipeline = ExerkineMapONNXPipeline(
    onnx_model_path="models/ExerciseTrainingModel.onnx",
    encoding_strategy='spatial_heatmap',
    apply_diffusion=True,
    diffusion_time=2.0
)

results = pipeline.predict_from_spatial_data(
    spatial_coords=coords,
    G_expr=expr_matrix,
    adjacency=adj_matrix,
    ligands=['IL6', 'IL15', 'BDNF'],
    receptors=['IL6R', 'IL15RA', 'NTRK2'],
    gene_names=genes
)
```

### Workflow 2: Multi-Organ Analysis

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

### Workflow 3: Temporal Dynamics

```python
temporal_results = pipeline.analyze_temporal_dynamics(
    spatial_coords=coords,
    G_expr=expr,
    adjacency=adj,
    ligands=ligands,
    receptors=receptors,
    gene_names=genes,
    time_points=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)
```

## Mathematical Framework Summary

### 1. Initial Exerkine Profile
```
f₀(i) = (x_i^exerkine + Σ_{j∈N_i} w_ij · x_j^exerkine) / (1 + Σ_{j∈N_i} w_ij)

where:
- x_i^exerkine = Σ_{l_k∈E} x_i(l_k)
- w_ij = edge weights (e.g., 1/(1+d_ij))
- N_i = neighbors of cell i
```

### 2. Signal Diffusion
```
F(t) = e^(-tL) · f₀

where:
- L = graph Laplacian (D - A)
- t = diffusion time parameter
- e^(-tL) = matrix exponential
```

### 3. LRI Interaction Score
```
S_ij^(l_k, r_m) = x_i(l_k) · x_j(r_m)

P_activation(c_i | c_j) = Σ_{(l_k, r_m)} δ_{l_k∈E} · S_ij^(l_k, r_m) · β_{r_m}

where:
- δ_{l_k∈E} = 1 if ligand is exerkine, else 0
- β_{r_m} = pathway impact score from databases
```

### 4. Spatial Distance Kernel
```
D_ij = exp(-||s_i - s_j||² / 2σ²)

ω_ij^spatial = D_ij · ω_ij
```

## Exercise State Classes

| ID | Class Label | Biological Interpretation |
|----|-------------|---------------------------|
| 0 | Sedentary_Baseline | No exercise stimulus |
| 1 | Acute_Endurance_Response | IL-6, IL-15 signaling |
| 2 | Acute_Resistance_Response | IGF-1, FGF2 pathways |
| 3 | Chronic_Aerobic_Adaptation | VEGF, BDNF upregulation |
| 4 | Chronic_Strength_Adaptation | Myokine/hepatokine crosstalk |
| 5 | Recovery_Phase | Anti-inflammatory, repair |
| 6 | Metabolic_Stress | Lactate, ROS signaling |
| 7 | Immune_Activation | Cytokine storm |
| 8 | Angiogenic_Remodeling | Endothelial signaling |
| 9 | Neuromuscular_Plasticity | BDNF, neurotrophins |

## Testing

Run all tests:
```bash
pytest tests/test_onnx_integration.py -v
```

Test coverage:
```bash
pytest tests/test_onnx_integration.py --cov=exerkinemap --cov-report=html
```

## Documentation

1. **Quick Start**: `ONNX_INTEGRATION_README.md`
2. **Comprehensive Guide**: `docs/ONNX_INTEGRATION_GUIDE.md`
3. **Examples**: `examples/onnx_integration_example.py`
4. **API Tests**: `tests/test_onnx_integration.py`

## Dependencies

Core:
- `numpy>=1.21.0`
- `scipy>=1.7.0`
- `scikit-learn>=1.0.0`
- `onnxruntime>=1.12.0`

Single-cell:
- `scanpy>=1.9.0`
- `anndata>=0.8.0`
- `squidpy>=1.2.0`

Optional:
- `onnxruntime-gpu` (GPU acceleration)
- `liana` (L-R databases)

## Next Steps

1. **Train ONNX Model**: Use your CNN to create `.onnx` file
2. **Prepare Data**: Format spatial transcriptomics data
3. **Define L-R Pairs**: Curate exerkines, ligands, receptors
4. **Set β Pathways**: Assign receptor pathway impacts
5. **Run Pipeline**: Execute end-to-end workflow
6. **Validate**: Cross-validate predictions
7. **Visualize**: Plot spatial patterns & temporal dynamics

## Support & Contribution

- **GitHub**: https://github.com/gomezdj/exerkinemap
- **Issues**: Submit bugs/feature requests
- **Discussions**: Ask questions
- **PRs**: Contributions welcome!
