# test_tokenization.py
import pytest
from workflows.tokenization import SequenceTokenizer

def test_rna_tokenization():
    tokenizer = SequenceTokenizer(model_type="omni-dna")
    sequence = "AUGGCCAUUGUAA"
    tokens = tokenizer.tokenize(sequence)
    assert len(tokens) > 0
    assert tokenizer.decode(tokens) == sequence

# test_plm.py
import pytest
from workflows.plm import ProteinLanguageModel

def test_plm_embedding_generation():
    plm = ProteinLanguageModel(model_name="esm2")
    embedding = plm.get_embeddings(["MVLSP"])
    assert embedding.dim() == 3 # [batch, seq_len, hidden_dim]

# test_glm.py
import pytest
from workflows.glm import GenomicLanguageModel

def test_glm_sequence_generation():
    glm = GenomicLanguageModel()
    generated = glm.generate_sequence(prompt_sequence="ATGC", max_length=50)
    assert len(generated) == 50
    assert all(n in "ATGC" for n in generated)