import pytest
import torch
from workflows.glm import GenomicLanguageModel

@pytest.fixture
def glm_instance():
    # Initialize with a lightweight configuration for testing
    return GenomicLanguageModel(model_name="BioBERT", hidden_dim=768)

def test_glm_embedding_dimensions(glm_instance):
    # Mock tokenized RNA sequence batch [batch_size, seq_len]
    mock_tokens = torch.randint(0, 50000, (4, 128)) 
    mock_attention_mask = torch.ones((4, 128))
    
    embeddings = glm_instance.get_embeddings(
        input_ids=mock_tokens, 
        attention_mask=mock_attention_mask
    )
    
    # Check [batch_size, seq_len, hidden_dim]
    assert embeddings.shape == (4, 128, 768)
    assert not torch.isnan(embeddings).any(), "GLM output contains NaN values"

def test_glm_sequence_generation(glm_instance):
    prompt_tokens = torch.tensor([[101, 45, 23]]) # Mock start tokens
    generated = glm_instance.generate_sequence(prompt_tokens, max_length=50)
    
    assert generated.shape[1] <= 50, "Generated sequence exceeds max_length constraint"