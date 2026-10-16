import pandas as pd

class InterOrganCrossTalkMapper:
    def __init__(self, ligand_receptor_db: pd.DataFrame):
        self.lr_db = ligand_receptor_db

    def map_secretory_signaling(self, predicted_exerkines: dict[str, float], target_tissues: list[str]) -> pd.DataFrame:
        """Maps secreted myokines/osteokines to target tissue receptor expression."""
        active_ligands = list(predicted_exerkines.keys())
        network_edges = self.lr_db[
            self.lr_db['ligand'].isin(active_ligands) & 
            self.lr_db['target_tissue'].isin(target_tissues)
        ].copy()
        
        network_edges['flux_score'] = network_edges['ligand'].map(predicted_exerkines)
        return network_edges