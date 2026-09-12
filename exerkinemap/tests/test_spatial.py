# test_spatial.py
import pytest
import numpy as np
from workflows.spatial import SpatialMapper

@pytest.fixture
def mock_spatial_anndata():
    import anndata as ad
    X = np.random.rand(100, 50) # 100 cells, 50 genes
    spatial_coords = np.random.rand(100, 2)
    obs = {"cell_type": ["myocyte"] * 100}
    return ad.AnnData(X=X, obs=obs, obsm={"spatial": spatial_coords})

def test_spatial_map_generation(mock_spatial_anndata):
    mapper = SpatialMapper()
    spatial_graph = mapper.build_neighborhood_graph(mock_spatial_anndata, n_neighbors=5)
    assert "connectivities" in spatial_graph.obsp
    assert spatial_graph.obsp["connectivities"].shape == (100, 100)