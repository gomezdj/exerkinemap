"""Spatial coordinate helpers."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


class SpatialCoordinates:
    """Simple container for spatial coordinates."""

    def __init__(self, coordinates: Sequence[Tuple[float, float]] | None = None):
        self.coordinates = list(coordinates or [])

    def append(self, point: Tuple[float, float]) -> None:
        self.coordinates.append(tuple(point))

    def __len__(self) -> int:
        return len(self.coordinates)

    def __iter__(self):
        return iter(self.coordinates)


def normalize_coordinates(coordinates: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    return [tuple(point) for point in coordinates]

import numpy as np
from typing import Dict, Any, Optional

class CCFRegistrationEngine:
    """
    Registers spatial omics data to the HuBMAP Common Coordinate Framework (CCF)
    and initializes a Celcomen Agent-Based Model (ABM) for virtual tissue simulation.
    """
    def __init__(self, ccf_reference_path: str, atlas_resolution: float = 1.0):
        self.ccf_reference_path = ccf_reference_path
        self.atlas_resolution = atlas_resolution
        self.celcomen_env = None
        self.transformation_matrix = None
        
    def register_to_ccf(self, spatial_coords: SpatialCoordinates, tissue_type: str) -> np.ndarray:
        """
        Aligns local spatial omics coordinates (from SpatialCoordinates container) 
        to the 3D CCF atlas space using anatomical landmarks.
        """
        # TODO: Implement actual rigid/affine transformation logic (e.g., via SimpleITK or ANTs)
        # mapping local cell centroids to the HuBMAP macro-anatomical grid.
        if self.transformation_matrix is None:
            # Default to an identity matrix matching the dimension + 1 for homogeneous coords
            self.transformation_matrix = np.eye(coords_array.shape[1] + 1)
            # Apply the atlas resolution scaling to the diagonal (excluding the homogeneous scale factor)
            np.fill_diagonal(self.transformation_matrix, self.atlas_resolution)
            self.transformation_matrix[-1, -1] = 1.0

        # Convert to homogeneous coordinates by adding a column of 1s
        ones_column = np.ones((coords_array.shape[0], 1))
        homogeneous_coords = np.hstack([coords_array, ones_column])
        
        # Apply transformation matrix using dot product
        # Matrix math: transformed_coords = X_homogeneous * T^T
        transformed_homogeneous = np.dot(homogeneous_coords, self.transformation_matrix.T)
        
        # Drop the homogeneous dimension (the 1s column) to return to standard coordinates
        registered_coords = transformed_homogeneous[:, :-1]

    def initialize_celcomen_abm(self, registered_coords: np.ndarray, microenvironment_params: Dict[str, Any]):
        """
        Initializes the Celcomen virtual tissue environment using the 
        CCF-registered physical boundaries and sets up morphogen physics.
        """
        try:
            import celcomen
        except ImportError:
            raise ImportError("Please install 'celcomen' to enable ABM tissue simulations.")
            
        # Define physical tissue boundaries based on CCF space
        min_bounds = np.min(registered_coords, axis=0)
        max_bounds = np.max(registered_coords, axis=0)
        
        # Initialize Celcomen physical grid mapped to the CCF bounds
        self.celcomen_env = celcomen.Environment(
            dimensions=(min_bounds, max_bounds),
            resolution=self.atlas_resolution
        )
        
        # Bridge the Reaction-Diffusion ODE outputs to the Celcomen physics engine
        if "morphogens" in microenvironment_params:
            for morphogen, params in microenvironment_params["morphogens"].items():
                self.celcomen_env.add_diffusion_field(
                    name=morphogen, 
                    diffusion_coefficient=params.get('diffusion_rate', 1.0),
                    decay_rate=params.get('decay_rate', 0.1)
                )
                
        return self.celcomen_env
        
    def step_developmental_simulation(self, dt: float):
        """
        Steps the Celcomen ABM forward in time, calculating physical cell migration, 
        division, and morphogen diffusion across the CCF space.
        """
        if not self.celcomen_env:
            raise ValueError("Celcomen environment not initialized. Run initialize_celcomen_abm first.")
        
        self.celcomen_env.step(dt)
