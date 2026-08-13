import pytest
import pandas as pd
from workflows.ligand_receptor import LigandReceptorMapper

@pytest.fixture
def mock_lr_database():
    return pd.DataFrame({
        "ligand": ["IL6", "TNF", "CXCL8"],
        "receptor": ["IL6R", "TNFRSF1A", "CXCR1"],
        "affinity_score": [0.92, 0.85, 0.45]
    })

def test_interaction_mapping(mock_lr_database):
    mapper = LigandReceptorMapper(database=mock_lr_database)
    interactions = mapper.get_interactions(ligands=["IL6", "TNF"])
    
    assert len(interactions) == 2
    assert "IL6R" in interactions["receptor"].values
    
def test_affinity_thresholding(mock_lr_database):
    mapper = LigandReceptorMapper(database=mock_lr_database, threshold=0.80)
    filtered_interactions = mapper.get_significant_interactions()
    
    # CXCL8 should be filtered out
    assert "CXCL8" not in filtered_interactions["ligand"].values
    assert all(filtered_interactions["affinity_score"] >= 0.80)