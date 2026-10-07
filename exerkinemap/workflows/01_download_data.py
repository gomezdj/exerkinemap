"""
workflows/01_download_data.py

Initializes local directories and orchestrates the ingestion of primary datasets:
- Sequence References (GENCODE, UniProt) via public FTPs.
- MoTrPAC (Temporal exercise-response omics) via local staging.
- HuBMAP / Human Cell Atlas (High-resolution spatial references) via local staging.
- Exerkine Atlas (Cataloged physiological biomarker validation).
"""

import sys
import shutil
import argparse
import requests
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Define standardized model architecture directories
TARGET_DIRS = {
    "sequences_rna": Path("data/raw/sequences/rna"),
    "sequences_protein": Path("data/raw/sequences/protein"),
    "metadata": Path("data/raw/metadata"),
    "single_cell": Path("data/raw/single_cell"),
    "spatial": Path("data/processed/spatial_maps"),
    "pretrained_fm": Path("results/pretrained_fm"),
    "finetuned_fm": Path("results/finetuned_motrpac_fm")
}

def setup_directories():
    """Creates the required data directory structure for EXERKINEMAP."""
    for name, path in TARGET_DIRS.items():
        path.mkdir(parents=True, exist_ok=True)
    logger.info("Local directory structure initialized.")

def download_reference_file(url: str, output_path: Path):
    """Download public molecular sequence references."""
    if output_path.exists():
        logger.info(f"Reference already exists: {output_path.name}. Skipping.")
        return

    logger.info(f"Downloading {url} to {output_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"Successfully downloaded {output_path.name}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download {url}: {e}")
        sys.exit(1)

def fetch_exerkine_atlas():
    """
    Connects to Exerkine Atlas for the physiological validation library.
    Note: In a production environment, this interfaces with specific API endpoints.
    """
    atlas_path = TARGET_DIRS["metadata"] / "exerkine_atlas.csv"
    if not atlas_path.exists():
        logger.info("Connecting to Exerkine Atlas API to build validation library...")
        # Placeholder for actual Exerkine Atlas API pull
        # e.g., requests.get("https://api.exerkineatlas.org/v1/export")
        
        # Creating a mock file for pipeline continuation
        with open(atlas_path, "w") as f:
            f.write("sequence,name,tissue_origin,exercise_modality\n")
        logger.info("Exerkine Atlas placeholder generated.")
    else:
        logger.info("Exerkine Atlas reference already exists. Skipping.")

def ingest_consortium_data(staging_dir: Path, use_symlinks: bool = False):
    """
    Transfer or symlink MoTrPAC and HuBMAP data from a staging directory 
    into the model's standardized data/raw/ structure.
    """
    if not staging_dir.exists():
        logger.error(f"Staging directory not found: {staging_dir}")
        sys.exit(1)

    logger.info(f"Scanning staging directory {staging_dir} for consortium data...")
    
    # Define mapping rules: (file_extension/keyword) -> target_directory
    for file_path in staging_dir.rglob("*"):
        if file_path.is_file():
            target_path = None
            filename = file_path.name.lower()

            if "metadata" in filename and filename.endswith(".csv"):
                target_path = TARGET_DIRS["metadata"] / file_path.name
            elif "spatial" in filename and filename.endswith(".h5ad"):
                target_path = TARGET_DIRS["spatial"] / file_path.name
            elif filename.endswith(".h5ad") or filename.endswith(".csv"):
                target_path = TARGET_DIRS["single_cell"] / file_path.name

            if target_path:
                if target_path.exists():
                    logger.info(f"File already in model: {target_path.name}. Skipping.")
                    continue

                if use_symlinks:
                    logger.info(f"Symlinking {file_path.name} -> {target_path}")
                    target_path.symlink_to(file_path)
                else:
                    logger.info(f"Copying {file_path.name} -> {target_path}")
                    shutil.copy2(file_path, target_path)

def main():
    parser = argparse.ArgumentParser(description="Ingest raw data into the EXERKINEMAP architecture.")
    parser.add_argument("--staging-dir", type=str, help="Path to the directory containing downloaded MoTrPAC/HuBMAP data.", required=False)
    parser.add_argument("--symlink", action="store_true", help="Use symlinks instead of copying large consortium files.")
    args = parser.parse_args()

    logger.info("Initializing EXERKINEMAP data ingestion workflow...")
    setup_directories()

    # 1. Download Public Reference Sequences (Omics FM Pre-training baselines)
    references = [
        {"url": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.transcripts.fa.gz", 
         "path": TARGET_DIRS["sequences_rna"] / "gencode.v44.transcripts.fa.gz"},
        {"url": "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Eukaryota/UP000005640/UP000005640_9606.fasta.gz", 
         "path": TARGET_DIRS["sequences_protein"] / "uniprot_human_proteome.fasta.gz"}
    ]

    for ref in references:
        download_reference_file(ref["url"], ref["path"])

    # 2. Fetch Physiological Validation Library
    fetch_exerkine_atlas()

    # 3. Ingest pre-downloaded empirical consortium data (MoTrPAC / HuBMAP)
    if args.staging_dir:
        ingest_consortium_data(Path(args.staging_dir), use_symlinks=args.symlink)
    else:
        logger.info("No staging directory provided. Skipping consortium data ingestion. Run with --staging-dir to import MoTrPAC/HuBMAP empirical data.")

    logger.info("Workflow 01_download_data complete. Pipeline ready for pre-training.")

if __name__ == "__main__":
    main()