"""
exerkinemap/tasks/biomarker_discovery.py

Extracts digital biomarkers from the mapped latent space and cross-references
these computationally derived molecular signatures with the Exerkine Atlas
for physiological validation.
"""

import pandas as pd

class BiomarkerDiscoverer:
    def __init__(self, prompter, atlas_reference_path: str):
        """
        Initializes the discoverer with the trained OmicsPrompter and the Exerkine Atlas reference.
        """
        self.prompter = prompter
        self.atlas_reference = self._load_exerkine_atlas(atlas_reference_path)

    def _load_exerkine_atlas(self, path: str) -> set:
        """Loads the Exerkine Atlas reference library of cataloged physiological molecules."""
        # Mock load: assumes a CSV with a 'sequence' column of validated exerkines
        try:
            df = pd.read_csv(path)
            return set(df['sequence'].tolist())
        except FileNotFoundError:
            print(f"Warning: Exerkine Atlas not found at {path}. Validation will return False.")
            return set()

    def extract_and_validate(self, physiological_context: str) -> list:
        """
        Uses the fine-tuned LLM to extract candidate biomarker sequences, 
        then benchmarks them against the Exerkine Atlas.
        """
        # 1. Prompt the foundation model for candidate sequences
        candidates = self.prompter.discover_biomarkers(physiological_context)
        
        # 2. Validate against the Exerkine Atlas
        validated_biomarkers = []
        for seq in candidates:
            is_validated = seq in self.atlas_reference
            validated_biomarkers.append({
                "sequence": seq,
                "in_exerkine_atlas": is_validated,
                "context": physiological_context
            })
            
        return validated_biomarkers

if __name__ == "__main__":
    # Example Usage
    from exerkinemap.foundation_model.prompting import OmicsPrompter
    
    fm_prompter = OmicsPrompter("results/finetuned_motrpac_fm")
    discoverer = BiomarkerDiscoverer(fm_prompter, "data/raw/metadata/exerkine_atlas.csv")
    
    results = discoverer.extract_and_validate("Skeletal muscle post-endurance training adaptation")
    print(results)