"""
exerkinemap/tasks/__init__.py

Exposes the specialized downstream task classes for the EXERKINEMAP framework.
"""

from .biomarker_discovery import BiomarkerDiscoverer
from .variant_prediction import VariantPredictor
from .personalized_medicine import PersonalizedMedicineMapper

__all__ = [
    "BiomarkerDiscoverer", 
    "VariantPredictor", 
    "PersonalizedMedicineMapper"
]