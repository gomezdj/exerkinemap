"""
workflows/05_prompt_specialized_tasks.py

Executes downstream agentic tasks utilizing the MoTrPAC-anchored LLM:
- Biomarker Discovery (Validated against the Exerkine Atlas)
- Variant Prediction (Spatial ligand-receptor affinity)
- Virtual Cell Mapping (Personalized medicine transition modeling)
"""

from exerkinemap.foundation_model.prompting import OmicsPrompter
from exerkinemap.tasks.biomarker_discovery import BiomarkerDiscoverer
from exerkinemap.tasks.variant_prediction import VariantPredictor
from exerkinemap.tasks.virtual_cell_mapping import VirtualCellMapper

def execute_tasks():
    print("Loading specialized Omics Prompter...")
    prompter = OmicsPrompter(finetuned_model_path="results/finetuned_motrpac_fm")
    
    # Task 1: Biomarker Discovery
    print("\n--- Executing Task 1: Biomarker Discovery ---")
    discoverer = BiomarkerDiscoverer(prompter, atlas_reference_path="data/raw/metadata/exerkine_atlas.csv")
    biomarkers = discoverer.extract_and_validate(
        physiological_context="High-intensity interval training response in skeletal muscle."
    )
    for b in biomarkers:
        print(f"Candidate: {b['sequence']} | Validated in Atlas: {b['in_exerkine_atlas']}")
        
    # Task 2: Variant Prediction & Affinity Optimization
    print("\n--- Executing Task 2: Variant Prediction ---")
    predictor = VariantPredictor(prompter)
    
    # Propose a novel exerkine for a specific condition
    novel_seq = predictor.generate_novel_exerkine("Gestational Diabetes Mellitus (GDM) placental interface")
    print(f"Proposed Exerkine Sequence: {novel_seq}")
    
    # Predict binding affinity for a spatial communication network edge
    affinity = predictor.predict_ligand_receptor_affinity(ligand_seq=novel_seq, receptor_seq="MKVLLILACL")
    print(f"Predicted Spatial Binding Affinity: {affinity}")

    # Task 3: Virtual Cell Mapping
    print("\n--- Executing Task 3: Virtual Cell Mapping ---")
    cell_mapper = VirtualCellMapper(prompter, sequence_reference_path="data/processed/spatial_maps/sequence_references.pt")
    virtual_map = cell_mapper.map_virtual_cells(
        physiological_context="High-intensity interval training response in skeletal muscle."
    )
    print(f"Generated Virtual Cell Map: {virtual_map}")

if __name__ == "__main__":
    execute_tasks()
