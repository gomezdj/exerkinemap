"""
data/dataset.py

Defines the standardized multimodal ExerkineDataset object for managing 
single-cell omics, spatial transcriptomics, sequences, metadata, and 
ligand-receptor databases within the EXERKINEMAP framework.
"""
import logging
import scanpy as sc
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

class ExerkineDataset:
    """
    Standardized container class for EXERKINEMAP multimodal inputs.
    """
   
    def __init__(
        self,
        single_cell: sc.AnnData,
        spatial: sc.AnnData,
        sequences: dict,
        metadata: pd.DataFrame,
        ligand_receptor: pd.DataFrame
    ):
        self.single_cell = single_cell
        self.spatial = spatial
        self.sequences = sequences
        self.metadata = metadata
        self.ligand_receptor = ligand_receptor
        
        logger.info("ExerkineDataset successfully initialized with multimodal modalities.")

    def summary(self):
        """Prints summary statistics of the dataset components."""
        print("=== EXERKINEMAP Dataset Summary ===")
        print(f"Single-Cell Cells (n_obs): {self.single_cell.n_obs if self.single_cell else 'N/A'}")
        print(f"Spatial Spots (n_obs): {self.spatial.n_obs if self.spatial else 'N/A'}")
        print(f"Loaded Sequence Types: {list(self.sequences.keys()) if self.sequences else 'N/A'}")
        print(f"Metadata Records: {len(self.metadata) if self.metadata is not None else 'N/A'}")
        print(f"Ligand-Receptor Pairs: {len(self.ligand_receptor) if self.ligand_receptor is not None else 'N/A'}")


# A standardized dataset object:
dataset = ExerkineDataset(
    single_cell=adata,
    spatial=spatial_data,
    sequences=sequence_data,
    metadata=metadata,
    ligand_receptor=lr_database
)


