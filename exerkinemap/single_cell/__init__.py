"""Single-cell analysis primitives for EXERKINEMAP."""

from .expression import ExpressionAnalyzer, analyze_expression
from .cell_type import CellTypeAnnotator, annotate_cell_types
from .clustering import ClusterAnalyzer, analyze_clusters
from .differential_expression import DifferentialExpressionAnalyzer, analyze_differential_expression
from .annotations import AnnotationManager, manage_annotations

__all__ = [
    "ExpressionAnalyzer",
    "analyze_expression",
    "CellTypeAnnotator",
    "annotate_cell_types",
    "ClusterAnalyzer",
    "analyze_clusters",
    "DifferentialExpressionAnalyzer",
    "analyze_differential_expression",
    "AnnotationManager",
    "manage_annotations",
]
