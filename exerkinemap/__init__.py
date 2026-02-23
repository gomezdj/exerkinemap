"""
ExerkineMap — Virtual Physiological System v2.2
================================================
Eight-layer pipeline for exercise-induced exerkine signaling:

  Layer 1  multiomics_integration   Z_i = f_multi(RNA, ATAC, RRBS, Protein)
  Layer 2  cell_lri_scores          P_ij = σ(α log x_i(l) + γ log x_j(r) + η_ij)
  Layer 3  signal_diffusion         dF/dt = -λF + Σ ω_ij F_j(t) + u_i(t)
  Layer 4  causal_inference         C_ij = E[Y_i | do(F_j)] - E[Y_i]
  Layer 5  ensembl.tree_ensembl     Ŷ = Σ α_k f_k(Z)  [XGB + RF + LightGBM]
  Layer 6  optimal_control          min_u ∫(F^TQF + u^TRu)dt
  Layer 7  generative_feedback      θ* = argmax_θ U(G_ExerkineMap(θ))
  Layer 8  analysis.pathways_analysis  F_o(t+1) = Σ_o' W_oo' F_o'(t)
"""

# ── Layer 1: Multi-Omic State
from .multiomics_integration import (
    integrate_multiomics,
    ConcatProjectionEmbedder,
    MultiOmicVAE,
)

# ── Layer 2: Probabilistic Communication
from .cell_lri_scores import (
    compute_initial_exerkine_profile,
    compute_cell_lri_scores,
    compute_pairwise_lri_matrix,
    compute_receptor_activation_scores,
)

# ── Layer 3: Dynamic Exerkine Flux
from .signal_diffusion import (
    compute_graph_laplacian,
    compute_diffused_signal,
    compute_diffusion_time_series,
)

# ── Layer 4: Causal Inference
from .causal_inference import (
    StructuralCausalModel,
    compute_causal_scores,
    exercise_intervention_effect,
)

# ── Layer 6: Optimal Control
from .optimal_control import (
    ExerkineLQRController,
    solve_lqr,
    build_signaling_matrix,
)

# ── Layer 7: Generative Feedback
from .generative_feedback import (
    GenerativeBiomoleculeOptimiser,
    BiomoleculeDesignSpace,
    generative_exerkine_design,
    detect_signaling_gaps,
)

# ── Spatial graph & models (require torch + torch_geometric)
try:
    from .spatial_graph_lri import build_spatial_graph, infer_lri_pairs
    from .spatial_omics_model import SpatialOmicsModel
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# ── ONNX pipeline (requires onnxruntime)
try:
    from .onnx_pipeline import ExerkineMapONNXPipeline
    _ONNX_PIPELINE_AVAILABLE = True
except ImportError:
    _ONNX_PIPELINE_AVAILABLE = False

# ── Exercise-biology class taxonomy
from .eb_classes import EXERKINEMAP_CLASSES

__version__ = "2.2.0"
__all__ = [
    # Layer 1
    "integrate_multiomics", "ConcatProjectionEmbedder", "MultiOmicVAE",
    # Layer 2
    "compute_initial_exerkine_profile", "compute_cell_lri_scores",
    "compute_pairwise_lri_matrix", "compute_receptor_activation_scores",
    # Layer 3
    "compute_graph_laplacian", "compute_diffused_signal", "compute_diffusion_time_series",
    # Layer 4
    "StructuralCausalModel", "compute_causal_scores", "exercise_intervention_effect",
    # Layer 6
    "ExerkineLQRController", "solve_lqr", "build_signaling_matrix",
    # Layer 7
    "GenerativeBiomoleculeOptimiser", "BiomoleculeDesignSpace",
    "generative_exerkine_design", "detect_signaling_gaps",
    # Spatial & models
    "build_spatial_graph", "infer_lri_pairs", "SpatialOmicsModel",
    # ONNX
    "ExerkineMapONNXPipeline",
    # Classes
    "EXERKINEMAP_CLASSES",
]
