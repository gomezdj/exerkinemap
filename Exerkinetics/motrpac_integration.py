import pandas as pd
import numpy as np
from pathlib import Path

class MoTrPACDataLoader:
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        
    def load_tissue_baseline(self, tissue: str, assay: str) -> pd.DataFrame:
        """
        Loads MoTrPAC multi-omic baseline data for a specified tissue and assay 
        (e.g., transcriptomics, proteomics, metabolomics).
        """
        file_path = self.data_dir / f"motrpac_{tissue}_{assay}_baseline.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"MoTrPAC baseline file not found at {file_path}")
        
        df = pd.read_parquet(file_path)
        return df

    def get_exerkine_markers(self, tissue: str, target_exerkines: list[str]) -> pd.DataFrame:
        """Extracts specific myokine/osteokine signature profiles from MoTrPAC datasets."""
        df = self.load_tissue_baseline(tissue, "proteomics")
        filtered_df = df[df['gene_symbol'].isin(target_exerkines)]
        return filtered_df