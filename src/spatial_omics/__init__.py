"""Spatial omics analysis primitives for EXERKINEMAP."""

from .xenium import XeniumProcessor, process_xenium_data
from .coordinates import SpatialCoordinates, normalize_coordinates
from .integration import SpatialIntegrationAnalyzer, integrate_spatial_data

__all__ = [
    "XeniumProcessor",
    "process_xenium_data",
    "SpatialCoordinates",
    "normalize_coordinates",
    "SpatialIntegrationAnalyzer",
    "integrate_spatial_data",
]
