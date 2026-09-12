import pytest
import numpy as np
from workflows.pathways import PathwayEnrichment

@pytest.fixture
def enrichment_analyzer():
    return PathwayEnrichment(database="kegg_motrpac_subset")

def test_hypergeometric_enrichment(enrichment_analyzer):
    mock_gene_list = ["AKT1", "MTOR", "AMPK", "IL6"]
    background_size = 20000
    
    results = enrichment_analyzer.run_hypergeometric_test(
        target_genes=mock_gene_list, 
        background_n=background_size
    )
    
    assert "p_value" in results.columns
    assert "adj_p_value" in results.columns
    # Ensure Benjamini-Hochberg correction never decreases the p-value
    assert all(results["adj_p_value"] >= results["p_value"])

def test_empty_gene_list(enrichment_analyzer):
    with pytest.raises(ValueError, match="Gene list cannot be empty"):
        enrichment_analyzer.run_hypergeometric_test(target_genes=[], background_n=20000)