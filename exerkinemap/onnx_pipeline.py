"""
EXERKINEMAP ONNX Integration Pipeline
Main entry point for LRI data → ONNX model → Exercise state classification
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import warnings

from .onnx_classifier import ExerkineONNXClassifier, EnsembleONNXClassifier
from .lri_image_encoder import LRIImageEncoder, create_image_from_exerkinemap_data
from .cell_lri_scores import (
    compute_initial_exerkine_profile,
    compute_cell_lri_scores,
    compute_pairwise_lri_matrix,
    compute_receptor_activation_scores,
    apply_spatial_weighting
)
from .signal_diffusion import (
    compute_diffused_signal,
    compute_diffusion_time_series
)


class ExerkineMapONNXPipeline:
    """
    End-to-end pipeline: EXERKINEMAP data → ONNX classification
    
    Workflow:
    1. Compute LRI scores from expression + graph
    2. Apply signal diffusion (optional)
    3. Encode to 28×28 image
    4. ONNX inference
    5. Return exercise state predictions
    """
    
    def __init__(
        self,
        onnx_model_path: str,
        encoding_strategy: str = 'spatial_heatmap',
        apply_diffusion: bool = True,
        diffusion_time: float = 2.0,
        image_size: int = 28
    ):
        """
        Initialize pipeline
        
        Parameters:
        -----------
        onnx_model_path : str
            Path to trained ONNX model
        encoding_strategy : str
            'spatial_heatmap', 'adjacency', 'multi_channel', or 'distance_weighted'
        apply_diffusion : bool
            Whether to apply signal diffusion before encoding
        diffusion_time : float
            Time parameter for F(t) diffusion
        image_size : int
            Target image dimension (default 28)
        """
        self.classifier = ExerkineONNXClassifier(onnx_model_path)
        self.encoder = LRIImageEncoder(image_size=image_size)
        self.encoding_strategy = encoding_strategy
        self.apply_diffusion = apply_diffusion
        self.diffusion_time = diffusion_time
        
        print(f"✓ Initialized EXERKINEMAP-ONNX pipeline")
        print(f"  Encoding: {encoding_strategy}")
        print(f"  Diffusion: {'enabled' if apply_diffusion else 'disabled'}")
    
    def predict_from_spatial_data(
        self,
        spatial_coords: np.ndarray,
        G_expr: np.ndarray,
        adjacency: np.ndarray,
        ligands: List[str],
        receptors: List[str],
        gene_names: List[str],
        exerkines: Optional[List[str]] = None,
        beta_pathway: Optional[Dict[str, float]] = None,
        edge_weights: Optional[np.ndarray] = None,
        return_image: bool = False
    ) -> Dict[str, Any]:
        """
        Predict exercise state from spatial single-cell data
        
        Parameters:
        -----------
        spatial_coords : np.ndarray, shape (N, 2)
            Cell spatial coordinates s_i = (x_i, y_i)
        G_expr : np.ndarray, shape (N, G)
            Gene expression matrix
        adjacency : np.ndarray, shape (N, N)
            Cell-cell adjacency from k-NN or radius graph
        ligands : list of str
            Ligand gene names
        receptors : list of str
            Receptor gene names
        gene_names : list of str
            All gene names
        exerkines : list of str, optional
            Exerkine subset
        beta_pathway : dict, optional
            Receptor → pathway impact scores
        edge_weights : np.ndarray, optional
            Edge weights for adjacency
        return_image : bool
            Whether to return the encoded image tensor
        
        Returns:
        --------
        results : dict
            {
                'predicted_class': int,
                'class_label': str,
                'probabilities': np.ndarray,
                'lri_scores': np.ndarray,
                'image': np.ndarray (if return_image=True)
            }
        """
        # Step 1: Compute initial exerkine profile f_0(i)
        if exerkines is None:
            exerkines = ligands
        
        f_0 = compute_initial_exerkine_profile(
            G_expr=G_expr,
            adjacency=adjacency,
            exerkines=exerkines,
            gene_names=gene_names,
            edge_weights=edge_weights
        )
        
        # Step 2: Apply signal diffusion (optional)
        if self.apply_diffusion:
            lri_scores = compute_diffused_signal(
                adjacency=adjacency,
                f_0=f_0,
                t=self.diffusion_time
            )
        else:
            lri_scores = f_0
        
        # Step 3: Compute comprehensive LRI scores
        lri_scores_full = compute_cell_lri_scores(
            G_expr=G_expr,
            ligands=ligands,
            receptors=receptors,
            gene_names=gene_names,
            adjacency=adjacency,
            exerkines=exerkines,
            beta_pathway=beta_pathway
        )
        
        # Combine diffused signal with full LRI scores (weighted average)
        combined_scores = 0.5 * lri_scores + 0.5 * lri_scores_full
        
        # Step 4: Encode to 28×28 image
        image = self._encode_to_image(
            spatial_coords=spatial_coords,
            lri_scores=combined_scores,
            adjacency=adjacency
        )
        
        # Step 5: ONNX inference
        predicted_class, probabilities, class_label = self.classifier.predict(image)
        
        # Prepare results
        results = {
            'predicted_class': predicted_class,
            'class_label': class_label,
            'probabilities': probabilities,
            'lri_scores': combined_scores,
            'f_0': f_0
        }
        
        if return_image:
            results['image'] = image
        
        return results
    
    def predict_batch(
        self,
        spatial_coords_list: List[np.ndarray],
        G_expr_list: List[np.ndarray],
        adjacency_list: List[np.ndarray],
        ligands: List[str],
        receptors: List[str],
        gene_names: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Batch prediction for multiple samples
        
        Parameters:
        -----------
        spatial_coords_list : list of np.ndarray
            List of spatial coordinate arrays
        G_expr_list : list of np.ndarray
            List of expression matrices
        adjacency_list : list of np.ndarray
            List of adjacency matrices
        ligands, receptors, gene_names : as above
        **kwargs : additional parameters for predict_from_spatial_data
        
        Returns:
        --------
        results_list : list of dict
        """
        results_list = []
        
        for i, (coords, expr, adj) in enumerate(zip(
            spatial_coords_list, G_expr_list, adjacency_list
        )):
            try:
                results = self.predict_from_spatial_data(
                    spatial_coords=coords,
                    G_expr=expr,
                    adjacency=adj,
                    ligands=ligands,
                    receptors=receptors,
                    gene_names=gene_names,
                    **kwargs
                )
                results['sample_id'] = i
                results_list.append(results)
            except Exception as e:
                warnings.warn(f"Failed to process sample {i}: {str(e)}")
                continue
        
        return results_list
    
    def predict_multi_organ(
        self,
        organ_data: Dict[str, Dict[str, np.ndarray]],
        ligands: List[str],
        receptors: List[str],
        gene_names: List[str],
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Predict exercise states across multiple organs
        
        Parameters:
        -----------
        organ_data : dict
            {
                'muscle': {'spatial_coords': ..., 'G_expr': ..., 'adjacency': ...},
                'liver': {...},
                'adipose': {...},
                ...
            }
        
        Returns:
        --------
        organ_results : dict
            {organ_name: prediction_results}
        """
        organ_results = {}
        
        for organ_name, data in organ_data.items():
            print(f"Processing {organ_name}...")
            
            try:
                results = self.predict_from_spatial_data(
                    spatial_coords=data['spatial_coords'],
                    G_expr=data['G_expr'],
                    adjacency=data['adjacency'],
                    ligands=ligands,
                    receptors=receptors,
                    gene_names=gene_names,
                    **kwargs
                )
                organ_results[organ_name] = results
            except Exception as e:
                warnings.warn(f"Failed to process {organ_name}: {str(e)}")
                continue
        
        return organ_results
    
    def _encode_to_image(
        self,
        spatial_coords: np.ndarray,
        lri_scores: np.ndarray,
        adjacency: np.ndarray
    ) -> np.ndarray:
        """Internal helper to encode LRI data to image"""
        
        if self.encoding_strategy == 'spatial_heatmap':
            image = self.encoder.spatial_heatmap_encoding(
                spatial_coords, lri_scores
            )
        
        elif self.encoding_strategy == 'adjacency':
            image = self.encoder.adjacency_matrix_encoding(
                adjacency, node_features=lri_scores
            )
        
        elif self.encoding_strategy == 'multi_channel':
            lri_dict = {'lri_scores': lri_scores}
            image = self.encoder.multi_channel_composite_encoding(
                spatial_coords, lri_dict
            )
        
        elif self.encoding_strategy == 'distance_weighted':
            image = self.encoder.distance_weighted_encoding(
                spatial_coords, lri_scores
            )
        
        else:
            raise ValueError(f"Unknown encoding strategy: {self.encoding_strategy}")
        
        return image
    
    def analyze_temporal_dynamics(
        self,
        spatial_coords: np.ndarray,
        G_expr: np.ndarray,
        adjacency: np.ndarray,
        ligands: List[str],
        receptors: List[str],
        gene_names: List[str],
        time_points: List[float],
        exerkines: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze how exercise state changes over diffusion time
        
        Parameters:
        -----------
        time_points : list of float
            Diffusion times to evaluate
        
        Returns:
        --------
        temporal_results : dict
            {
                'time_points': list,
                'predicted_classes': list,
                'probabilities': np.ndarray (T, 10),
                'lri_time_series': np.ndarray (N, T)
            }
        """
        if exerkines is None:
            exerkines = ligands
        
        # Compute f_0
        f_0 = compute_initial_exerkine_profile(
            G_expr=G_expr,
            adjacency=adjacency,
            exerkines=exerkines,
            gene_names=gene_names
        )
        
        # Compute diffusion time series
        F_time_series = compute_diffusion_time_series(
            adjacency=adjacency,
            f_0=f_0,
            time_points=time_points
        )
        
        # Predict at each timepoint
        predicted_classes = []
        all_probabilities = []
        
        for t_idx, t in enumerate(time_points):
            lri_scores = F_time_series[:, t_idx]
            
            image = self._encode_to_image(
                spatial_coords=spatial_coords,
                lri_scores=lri_scores,
                adjacency=adjacency
            )
            
            pred_class, probs, _ = self.classifier.predict(image)
            predicted_classes.append(pred_class)
            all_probabilities.append(probs)
        
        return {
            'time_points': time_points,
            'predicted_classes': predicted_classes,
            'probabilities': np.array(all_probabilities),
            'lri_time_series': F_time_series
        }
    
    def get_class_labels(self) -> Dict[int, str]:
        """Get exercise state class labels"""
        return self.classifier.class_labels


def create_pipeline_from_config(config: Dict) -> ExerkineMapONNXPipeline:
    """
    Factory function to create pipeline from configuration
    
    Parameters:
    -----------
    config : dict
        {
            'onnx_model_path': str,
            'encoding_strategy': str,
            'apply_diffusion': bool,
            'diffusion_time': float,
            ...
        }
    
    Returns:
    --------
    pipeline : ExerkineMapONNXPipeline
    """
    return ExerkineMapONNXPipeline(**config)
