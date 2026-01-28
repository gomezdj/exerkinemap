"""
Unit Tests for EXERKINEMAP ONNX Integration
"""

import pytest
import numpy as np
from exerkinemap.onnx_classifier import ExerkineONNXClassifier
from exerkinemap.lri_image_encoder import LRIImageEncoder
from exerkinemap.cell_lri_scores import (
    compute_initial_exerkine_profile,
    compute_cell_lri_scores,
    compute_pairwise_lri_matrix
)
from exerkinemap.signal_diffusion import (
    compute_graph_laplacian,
    compute_diffused_signal
)


class TestLRIImageEncoder:
    """Test image encoding from LRI data"""
    
    def test_spatial_heatmap_encoding(self):
        encoder = LRIImageEncoder(image_size=28)
        
        # Create test data
        N = 50
        spatial_coords = np.random.rand(N, 2) * 100
        lri_scores = np.random.rand(N)
        
        # Encode
        image = encoder.spatial_heatmap_encoding(spatial_coords, lri_scores)
        
        # Check shape
        assert image.shape == (28, 28, 1)
        assert image.dtype == np.float32
        
        # Check normalization
        assert image.min() >= 0.0
        assert image.max() <= 1.0
    
    def test_adjacency_matrix_encoding(self):
        encoder = LRIImageEncoder(image_size=28)
        
        # Create test adjacency
        N = 50
        adjacency = np.random.rand(N, N)
        adjacency = (adjacency + adjacency.T) / 2  # Make symmetric
        
        # Encode
        image = encoder.adjacency_matrix_encoding(adjacency)
        
        assert image.shape == (28, 28, 1)
        assert image.dtype == np.float32
    
    def test_multi_channel_encoding(self):
        encoder = LRIImageEncoder(image_size=28)
        
        N = 50
        spatial_coords = np.random.rand(N, 2) * 100
        
        lri_features = {
            'feature1': np.random.rand(N),
            'feature2': np.random.rand(N),
            'feature3': np.random.rand(N)
        }
        
        weights = {'feature1': 0.5, 'feature2': 0.3, 'feature3': 0.2}
        
        image = encoder.multi_channel_composite_encoding(
            spatial_coords, lri_features, weights=weights
        )
        
        assert image.shape == (28, 28, 1)
    
    def test_empty_data_handling(self):
        encoder = LRIImageEncoder(image_size=28)
        
        # Empty arrays
        empty_coords = np.array([]).reshape(0, 2)
        empty_scores = np.array([])
        
        image = encoder.spatial_heatmap_encoding(empty_coords, empty_scores)
        
        # Should return zero image
        assert image.shape == (28, 28, 1)
        assert np.all(image == 0)


class TestCellLRIScores:
    """Test LRI score computation"""
    
    def test_initial_exerkine_profile(self):
        N, G = 100, 50
        G_expr = np.random.rand(N, G)
        adjacency = np.random.randint(0, 2, (N, N))
        
        exerkines = ['GENE_0', 'GENE_1', 'GENE_2']
        gene_names = [f'GENE_{i}' for i in range(G)]
        
        f_0 = compute_initial_exerkine_profile(
            G_expr, adjacency, exerkines, gene_names
        )
        
        assert f_0.shape == (N,)
        assert np.all(f_0 >= 0)
    
    def test_cell_lri_scores(self):
        N, G = 50, 30
        G_expr = np.random.rand(N, G)
        adjacency = np.random.randint(0, 2, (N, N))
        
        ligands = ['GENE_0', 'GENE_1']
        receptors = ['GENE_5', 'GENE_6']
        gene_names = [f'GENE_{i}' for i in range(G)]
        
        scores = compute_cell_lri_scores(
            G_expr, ligands, receptors, gene_names, adjacency
        )
        
        assert scores.shape == (N,)
        assert np.all(scores >= 0)
    
    def test_pairwise_lri_matrix(self):
        N, G = 30, 20
        G_expr = np.random.rand(N, G)
        
        ligands = ['GENE_0', 'GENE_1']
        receptors = ['GENE_5', 'GENE_6']
        gene_names = [f'GENE_{i}' for i in range(G)]
        
        lri_matrix = compute_pairwise_lri_matrix(
            G_expr, ligands, receptors, gene_names
        )
        
        assert lri_matrix.shape == (N, N)
        assert np.all(lri_matrix >= 0)
    
    def test_expression_threshold(self):
        N, G = 50, 30
        G_expr = np.random.rand(N, G) * 0.1  # Low expression
        adjacency = np.random.randint(0, 2, (N, N))
        
        ligands = ['GENE_0', 'GENE_1']
        receptors = ['GENE_5', 'GENE_6']
        gene_names = [f'GENE_{i}' for i in range(G)]
        
        # With threshold
        scores_thresh = compute_cell_lri_scores(
            G_expr, ligands, receptors, gene_names, adjacency,
            expression_threshold=0.5
        )
        
        # Without threshold
        scores_no_thresh = compute_cell_lri_scores(
            G_expr, ligands, receptors, gene_names, adjacency,
            expression_threshold=0.0
        )
        
        # Scores with threshold should be lower
        assert scores_thresh.sum() <= scores_no_thresh.sum()


class TestSignalDiffusion:
    """Test signal diffusion computation"""
    
    def test_graph_laplacian(self):
        N = 20
        adjacency = np.random.rand(N, N)
        adjacency = (adjacency + adjacency.T) / 2  # Symmetric
        
        L = compute_graph_laplacian(adjacency, normalized=True)
        
        assert L.shape == (N, N)
        
        # Check Laplacian properties
        # 1. Symmetric
        assert np.allclose(L, L.T)
        
        # 2. Row sums should be zero (for unnormalized)
        L_unnorm = compute_graph_laplacian(adjacency, normalized=False)
        row_sums = L_unnorm.sum(axis=1)
        assert np.allclose(row_sums, 0, atol=1e-10)
    
    def test_diffused_signal(self):
        N = 30
        adjacency = np.random.rand(N, N)
        adjacency = (adjacency + adjacency.T) / 2
        
        f_0 = np.random.rand(N)
        
        # Test different time points
        for t in [0.1, 1.0, 5.0]:
            F_t = compute_diffused_signal(adjacency, f_0, t=t)
            
            assert F_t.shape == (N,)
            assert np.all(np.isfinite(F_t))
    
    def test_diffusion_conservation(self):
        """Signal mass should be approximately conserved"""
        N = 20
        adjacency = np.random.rand(N, N)
        adjacency = (adjacency + adjacency.T) / 2
        
        f_0 = np.ones(N)
        
        F_t = compute_diffused_signal(adjacency, f_0, t=1.0)
        
        # Total signal should be approximately conserved
        assert np.abs(F_t.sum() - f_0.sum()) < 1.0


class TestONNXClassifier:
    """Test ONNX model wrapper (requires actual model file)"""
    
    @pytest.mark.skipif(True, reason="Requires actual ONNX model file")
    def test_classifier_initialization(self):
        # This test would require an actual ONNX model
        model_path = "tests/fixtures/test_model.onnx"
        classifier = ExerkineONNXClassifier(model_path)
        
        assert classifier.session is not None
        assert len(classifier.class_labels) == 10
    
    @pytest.mark.skipif(True, reason="Requires actual ONNX model file")
    def test_prediction(self):
        model_path = "tests/fixtures/test_model.onnx"
        classifier = ExerkineONNXClassifier(model_path)
        
        # Create test image
        image = np.random.rand(28, 28, 1).astype(np.float32)
        
        pred_class, probs, label = classifier.predict(image)
        
        assert isinstance(pred_class, int)
        assert 0 <= pred_class < 10
        assert probs.shape == (10,)
        assert np.isclose(probs.sum(), 1.0)
        assert isinstance(label, str)


class TestEndToEndPipeline:
    """Test complete pipeline workflow"""
    
    def test_pipeline_flow(self):
        """Test data flow through entire pipeline"""
        
        # Generate synthetic data
        N, G = 100, 50
        spatial_coords = np.random.rand(N, 2) * 100
        G_expr = np.random.rand(N, G)
        
        # Create adjacency
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=6).fit(spatial_coords)
        distances, indices = nbrs.kneighbors(spatial_coords)
        
        adjacency = np.zeros((N, N))
        for i in range(N):
            for j in indices[i, 1:]:
                adjacency[i, j] = 1
        
        # Define genes
        ligands = ['GENE_0', 'GENE_1', 'GENE_2']
        receptors = ['GENE_10', 'GENE_11', 'GENE_12']
        gene_names = [f'GENE_{i}' for i in range(G)]
        exerkines = ['GENE_0', 'GENE_1']
        
        # Step 1: Compute f_0
        f_0 = compute_initial_exerkine_profile(
            G_expr, adjacency, exerkines, gene_names
        )
        assert f_0.shape == (N,)
        
        # Step 2: Diffuse signal
        F_t = compute_diffused_signal(adjacency, f_0, t=2.0)
        assert F_t.shape == (N,)
        
        # Step 3: Compute LRI scores
        lri_scores = compute_cell_lri_scores(
            G_expr, ligands, receptors, gene_names, adjacency
        )
        assert lri_scores.shape == (N,)
        
        # Step 4: Encode to image
        encoder = LRIImageEncoder(image_size=28)
        image = encoder.spatial_heatmap_encoding(spatial_coords, lri_scores)
        assert image.shape == (28, 28, 1)
        
        # Step 5: Would do ONNX inference here
        # (skipped without actual model)
        
        print("✓ End-to-end pipeline test passed")


def test_integration_with_anndata_structure():
    """Test compatibility with AnnData-like structures"""
    
    # Simulate AnnData structure
    N, G = 200, 100
    
    # Create mock adata structure
    class MockAnnData:
        def __init__(self):
            self.obsm = {'spatial': np.random.rand(N, 2) * 100}
            self.X = np.random.rand(N, G)
            self.var_names = [f'GENE_{i}' for i in range(G)]
    
    adata = MockAnnData()
    
    # Extract data (as would be done in practice)
    spatial_coords = adata.obsm['spatial']
    G_expr = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
    gene_names = adata.var_names
    
    # Verify structure
    assert spatial_coords.shape == (N, 2)
    assert G_expr.shape == (N, G)
    assert len(gene_names) == G
    
    print("✓ AnnData compatibility test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
