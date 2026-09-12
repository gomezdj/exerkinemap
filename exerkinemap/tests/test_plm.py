import pytest
import torch
from workflows.plm import ProteinLanguageModel

@pytest.fixture
def plm_instance():
    return ProteinLanguageModel(model_name="esm2_t33_650M_UR50D")

def test_amino_acid_motif_extraction(plm_instance):
    # Mock protein sequence
    sequence = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF"
    
    # Extract representation for a specific motif range (e.g., residues 10-15)
    motif_embedding = plm_instance.extract_motif_embedding(sequence, start_idx=10, end_idx=15)
    
    # Ensure correct sequence length slicing in the tensor
    assert motif_embedding.shape[0] == 5 # 15 - 10
    assert motif_embedding.shape[1] == plm_instance.hidden_dim

def test_cls_token_representation(plm_instance):
    sequence = "MVLSPAD"
    cls_embedding = plm_instance.get_sequence_embedding(sequence)
    
    # Should return a single vector per sequence
    assert cls_embedding.dim() == 1
    assert len(cls_embedding) == plm_instance.hidden_dim