"""
download_references.py

This script programmatically downloads pathway and protein-protein interaction (PPI) 
databases (STRING, BioGRID, Reactome) required for the MoTrPAC signal propagation 
and pathway activation models.
"""
import sys
import requests
import logging
import tarfile
import zipfile
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Resolve repository root and set reference data paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REF_DIR = PROJECT_ROOT / "data" / "references"

TARGET_DIRS = {
    "pathways": REF_DIR / "pathways",
    "networks": REF_DIR / "networks",
}

def create_directories():
    """Ensure all reference directories exist."""
    for path in TARGET_DIRS.values():
        path.mkdir(parents=True, exist_ok=True)
    logger.info("Verified reference subdirectories.")

def download_and_extract(url: str, output_path: Path, extract: bool = False):
    """Download a file, with optional extraction for zip/tar archives."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        logger.info(f"File already exists: {output_path.name}. Skipping.")
        return

    logger.info(f"Downloading {url} to {output_path}...")
    try:
        with requests.Session() as session:
            response = session.get(url, stream=True, timeout=30)
            response.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        logger.info(f"Successfully downloaded {output_path.name}")

        if extract:
            logger.info(f"Extracting {output_path.name}...")
            if output_path.suffix == ".zip":
                with zipfile.ZipFile(output_path, "r") as zip_ref:
                    zip_ref.extractall(output_path.parent)
            elif output_path.suffix in [".tgz", ".tar", ".tar.gz"]:
                with tarfile.open(output_path, "r:*") as tar_ref:
                    tar_ref.extractall(output_path.parent)
            else:
                logger.warning(f"Unsupported extraction format for {output_path.name}. Skipping extraction.")
            logger.info("Extraction complete.")

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download {url}: {e}")
        sys.exit(1)

def main():
    logger.info("Initializing reference download workflow for MoTrPAC...")
    create_directories()

    # Define the core interaction and pathway databases
    references = [
        # STRING DB - Homo sapiens protein links (v12.0)
        {"url": "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz", 
         "path": TARGET_DIRS["networks"] / "9606.string.links.v12.0.txt.gz",
         "extract": False},
        
        # Reactome - Complete pathway mapping
        {"url": "https://reactome.org/download/current/ReactomePathways.txt", 
         "path": TARGET_DIRS["pathways"] / "ReactomePathways.txt",
         "extract": False},
         
        # Reactome - Ensembl to Pathway mapping
        {"url": "https://reactome.org/download/current/Ensembl2Reactome.txt", 
         "path": TARGET_DIRS["pathways"] / "Ensembl2Reactome.txt",
         "extract": False},

        # BioGRID - Latest Release (Update version number as needed, currently v4.4.226)
        {"url": "https://downloads.thebiogrid.org/Download/BioGRID/Release-Archive/BIOGRID-4.4.226/BIOGRID-ALL-4.4.226.tab3.zip", 
         "path": TARGET_DIRS["networks"] / "BIOGRID-ALL-latest.tab3.zip",
         "extract": True}
    ]

    for ref in references:
        download_and_extract(ref["url"], ref["path"], extract=ref["extract"])
        
    logger.info("Note: KEGG API requires specific REST queries or licensing for bulk downloads. Use BioServices or KEGG API directly for specific maps.")
    logger.info("Workflow download_references complete.")

if __name__ == "__main__":
    main()
