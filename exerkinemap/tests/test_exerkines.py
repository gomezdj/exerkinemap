# test_exerkines.py
import pytest
from workflows.exerkines import ExerkineProcessor

@pytest.fixture
def mock_motrpac_exerkine_data():
    return [{"protein_id": "P12345", "sequence": "MVLSPADKTNVK", "tissue_source": "skeletal_muscle"}]

def test_exerkine_representation_generation(mock_motrpac_exerkine_data):
    processor = ExerkineProcessor(data=mock_motrpac_exerkine_data)
    embeddings = processor.generate_representations()
    assert embeddings.shape[1] > 0, "Exerkine embeddings should not be empty"
    assert processor.is_validated, "Exerkine sequences must pass validation"

# test_ligand_receptor.py
import pytest
from workflows.ligand_receptor import LigandReceptorMapper

def test_lr_mapping():
    mapper = LigandReceptorMapper()
    interactions = mapper.map_interactions(ligand="IL6", receptor="IL6R")
    assert "binding_affinity" in interactions
    assert interactions["binding_affinity"] >= 0.0