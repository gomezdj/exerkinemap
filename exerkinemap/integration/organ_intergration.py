# exerkinemap/integration/organ_intergration.py
"""Organ-level integration scaffolding."""

import pandas as pd
from typing import Dict

class MoTrPACRatCrosstalkBridge:
    """
    Ingests 6-month-old rat endurance multi-omics bundles to map
    source-to-target exerkine trajectories across all available tissues.
    """
    def __init__(self, bundle_dir: str):
        # Load the differential expression datasets from all MoTrPAC rat tissue bundles
        self.tissues = {
            "gastrocnemius": pd.read_csv(f"{bundle_dir}/gastrocnemius_transcriptomics.csv"),
            "heart": pd.read_csv(f"{bundle_dir}/heart_transcriptomics.csv"),
            "liver": pd.read_csv(f"{bundle_dir}/liver_transcriptomics.csv"),
            "lung": pd.read_csv(f"{bundle_dir}/lung_transcriptomics.csv"),
            "kidney": pd.read_csv(f"{bundle_dir}/kidney_transcriptomics.csv"),
            "brown_adipose": pd.read_csv(f"{bundle_dir}/brown_adipose_transcriptomics.csv"),
            "white_adipose": pd.read_csv(f"{bundle_dir}/white_adipose_transcriptomics.csv"),
            "blood_rna": pd.read_csv(f"{bundle_dir}/blood_rna_transcriptomics.csv")
        }
        
    def extract_ligand_receptor_pair(
        self, 
        ligand: str, 
        receptor: str, 
        timepoint: str, 
        source_tissue: str = "gastrocnemius", 
        target_tissue: str = "liver"
    ) -> Dict:
        """
        Maps a ligand secreted from the source tissue to its receptor expression 
        in a target tissue at identical timepoints.
        """
        if source_tissue not in self.tissues or target_tissue not in self.tissues:
            raise ValueError(f"Invalid tissue. Available tissues: {list(self.tissues.keys())}")

        source_df = self.tissues[source_tissue]
        target_df = self.tissues[target_tissue]
        
        # Source Tissue Expression
        source_data = source_df[
            (source_df['gene_symbol'] == ligand) & 
            (source_df['timepoint'] == timepoint)
        ]
        
        # Target Tissue Expression
        target_data = target_df[
            (target_df['gene_symbol'] == receptor) & 
            (target_df['timepoint'] == timepoint)
        ]
        
        return {
            "ligand": ligand,
            "source_tissue": source_tissue,
            "source_logFC": source_data['logFC'].values[0] if not source_data.empty else 0.0,
            "receptor": receptor,
            "target_tissue": target_tissue,
            "target_logFC": target_data['logFC'].values[0] if not target_data.empty else 0.0,
            "timepoint": timepoint
        }