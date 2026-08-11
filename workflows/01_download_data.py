"""
01_download_data.py

This script prepares the raw data architecture for the computational framework.
It automatically downloads sequence references (GENCODE, UniProt) from public FTPs,
and ingests (copies or symlinks) pre-downloaded consortium data (MoTrPAC, HuBMAP) 
from a local staging directory into the model's standardized data structure.
"""
import sys
import shutil
import argparse
import requests
import logging
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Resolve repository root and set target raw data paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

TARGET_DIRS = {
    "single_cell": DATA_DIR / "single_cell",
    "spatial": DATA_DIR / "spatial",
    "sequences_dna": DATA_DIR / "sequences" / "dna",
    "sequences_rna": DATA_DIR / "sequences" / "rna",
    "sequences_protein": DATA_DIR / "sequences" / "protein",
    "metadata": DATA_DIR / "metadata"
}

def create_directories():
    """Ensure all target data directories exist in the repository."""
    for name, path in TARGET_DIRS.items():
        path.mkdir(parents=True, exist_ok=True)
    logger.info("Verified all data/raw/ subdirectories.")

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
    parser = argparse.ArgumentParser(description="Ingest raw data into the model architecture.")
    parser.add_argument("--staging-dir", type=str, help="Path to the directory containing downloaded MoTrPAC/HuBMAP data.", required=False)
    parser.add_argument("--symlink", action="store_true", help="Use symlinks instead of copying large consortium files.")
    args = parser.parse_args()

    logger.info("Initializing data ingestion workflow...")
    create_directories()

    # 1. Download Public Reference Sequences
    references = [
        {"url": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.transcripts.fa.gz", 
         "path": TARGET_DIRS["sequences_rna"] / "gencode.v44.transcripts.fa.gz"},
        {"url": "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Eukaryota/UP000005640/UP000005640_9606.fasta.gz", 
         "path": TARGET_DIRS["sequences_protein"] / "uniprot_human_proteome.fasta.gz"}
    ]

    for ref in references:
        download_reference_file(ref["url"], ref["path"])

    # 2. Ingest pre-downloaded consortium data if staging directory is provided
    if args.staging_dir:
        ingest_consortium_data(Path(args.staging_dir), use_symlinks=args.symlink)
    else:
        logger.info("No staging directory provided. Skipping consortium data ingestion. Run with --staging-dir to import MoTrPAC/HuBMAP data.")

    logger.info("Workflow 01_download_data complete.")

if __name__ == "__main__":
    main()
