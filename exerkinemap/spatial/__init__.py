"""Spatial analysis primitives for EXERKINEMAP."""

from .coordinates import SpatialCoordinates, normalize_coordinates
from .distance import pairwise_distance, compute_distance_matrix
from .kernels import GaussianKernel, compute_kernel_matrix
from .neighborhoods import NeighborhoodBuilder, build_neighborhoods
from .spatial_interactions import SpatialInteractionAnalyzer, analyze_spatial_interactions

__all__ = [
    "SpatialCoordinates",
    "normalize_coordinates",
    "pairwise_distance",
    "compute_distance_matrix",
    "GaussianKernel",
    "compute_kernel_matrix",
    "NeighborhoodBuilder",
    "build_neighborhoods",
    "SpatialInteractionAnalyzer",
    "analyze_spatial_interactions",
]
