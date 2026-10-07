"""
workflows/04_build_sequence_reference.py

Tokenizes reference datasets (coding and non-coding regulatory elements) 
and projects them into the unified multimodal latent space for spatial alignment.
"""

import torch
from exerkinemap.foundation_model.embeddings import OmicsEmbedder

def build_references():
    print("Initializing Omics Embedder...")
    embedder = OmicsEmbedder()
    print(f"Hardware acceleration active on: {embedder.device}")
    
    # Example raw sequences (In production, these are loaded from data/raw/sequences)
    sample_proteins = ["MGFVLRRDWR", "MKVLLILACLVALALA"]
    sample_rnas = ["ATGCGTACGTAGCTAG", "CGCGCATATATCGCG"]
    
    print("Generating context-aware embeddings...")
    unified_batch = embedder.build_multimodal_batch(sample_proteins, sample_rnas)
    
    # Save the resulting tensor embeddings for downstream modality fusion and spatial mapping
    torch.save(unified_batch, "data/processed/spatial_maps/sequence_references.pt")
    print("Sequence reference libraries built and projected to latent space.")

if __name__ == "__main__":
    build_references()