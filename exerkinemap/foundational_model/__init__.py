"""
exerkinemap/foundation_model/__init__.py

Exposes the core classes for the three-stage Omics Foundation Model (FM) pipeline 
and multimodal latent space integration.
"""

from .pretraining import OmicsPretrainer
from .finetuning import MoTrPACFineTuner
from .prompting import OmicsPrompter
from .embeddings import OmicsEmbedder

__all__ = [
    "OmicsPretrainer",
    "MoTrPACFineTuner",
    "OmicsPrompter",
    "OmicsEmbedder"
]