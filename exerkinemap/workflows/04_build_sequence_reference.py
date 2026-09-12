"""
04_build_sequence_reference.py

This script processes the raw molecular sequence references (GENCODE, UniProt).
It parses the FASTA files, cleans the sequences, and builds structured 
reference indices for the Genomic Language Model (GLM) and Protein Language Model (PLM) 
within the EXERKINEMAP architecture.
"""
import gzip
import logging
import pandas as pd
from Bio import SeqIO
from pathlib import Path
from datasets import Dataset, DatasetDict
from huggingface_hub import login

# Configure logging
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_SEQ_DIR = PROJECT_ROOT / "data" / "raw" / "sequences"
PROCESSED_SEQ_DIR = PROJECT_ROOT / "data" / "processed" / "sequences"

# Hugging Face Repository settings
HF_REPO_ID = os.getenv("HF_REPO_ID", "gomezdj/exerkine-sequences")
HF_TOKEN = os.getenv("HF_TOKEN") # Ensure your token is set in your environment variables

def create_directories():
    """Ensure processed sequence directories exist."""
    (PROCESSED_SEQ_DIR / "dna").mkdir(parents=True, exist_ok=True)
    (PROCESSED_SEQ_DIR / "rna").mkdir(parents=True, exist_ok=True)
    (PROCESSED_SEQ_DIR / "protein").mkdir(parents=True, exist_ok=True)

def parse_fasta(file_path, file_type="fasta"):
    """Generator to parse gzipped FASTA files iteratively to save memory."""
    with gzip.open(file_path, "rt") as handle:
        for record in SeqIO.parse(handle, file_type):
            yield record

def build_rna_reference():
    """Process GENCODE RNA transcripts."""
    rna_file = RAW_SEQ_DIR / "rna" / "gencode.v44.transcripts.fa.gz"
    if not rna_file.exists():
        logger.warning(f"RNA reference not found: {rna_file}. Ensure 01_download_data.py was run.")
        return
    
    logger.info("Processing RNA sequences (GENCODE)...")
    records = []
    for record in parse_fasta(rna_file):
        # Extract Ensembl ID without version for easier matching
        header_parts = record.id.split('|')
        transcript_id = header_parts[0].split('.')[0]
        gene_id = header_parts[1].split('.')[0] if len(header_parts) > 1 else "Unknown"
        
        records.append({
            "transcript_id": transcript_id,
            "gene_id": gene_id,
            "sequence": str(record.seq),
            "length": len(record.seq)
        })
        
    df = pd.DataFrame(records)
    output_path = PROCESSED_SEQ_DIR / "rna" / "transcript_reference.parquet"
    
    # Save as Parquet for highly optimized reading during GLM training
    df.to_parquet(output_path, index=False)
    logger.info(f"RNA reference built: {len(df)} transcripts saved to {output_path}")

def build_protein_reference():
    """Process UniProt amino acid sequences."""
    protein_file = RAW_SEQ_DIR / "protein" / "uniprot_human_proteome.fasta.gz"
    if not protein_file.exists():
        logger.warning(f"Protein reference not found: {protein_file}. Ensure 01_download_data.py was run.")
        return
        
    logger.info("Processing Protein sequences (UniProt)...")
    records = []
    for record in parse_fasta(protein_file):
        # UniProt ID extraction (e.g., sp|P12345|NAME)
        parts = record.id.split('|')
        uniprot_id = parts[1] if len(parts) > 1 else record.id
        
        records.append({
            "uniprot_id": uniprot_id,
            "description": record.description,
            "sequence": str(record.seq),
            "length": len(record.seq)
        })

    df = pd.DataFrame(records)
    output_path = PROCESSED_SEQ_DIR / "protein" / "protein_reference.parquet"
    df.to_parquet(output_path, index=False)
    logger.info(f"Protein reference built: {len(df)} proteins saved to {output_path}")

def main():
    logger.info("Initializing 04_build_sequence_reference workflow...")
    create_directories()
    
    # Process sequences and generate Hugging Face Dataset objects
    rna_dataset = build_rna_reference()
    protein_dataset = build_protein_reference()
    
    build_rna_reference()
    build_protein_reference()
    
    # Push to Hugging Face Hub if configured
    if HF_TOKEN and (rna_dataset or protein_dataset):
        logger.info(f"Authenticating with Hugging Face and pushing to {HF_REPO_ID}...")
        login(token=HF_TOKEN)
        
        # Combine into a single DatasetDict with multiple splits/configs
        dataset_dict = DatasetDict()
        if rna_dataset:
            dataset_dict["rna_transcripts"] = rna_dataset
        if protein_dataset:
            dataset_dict["proteins"] = protein_dataset
            
        dataset_dict.push_to_hub(HF_REPO_ID, private=True)
        logger.info("Successfully pushed references to the Hugging Face Hub.")
    else:
        logger.info("Skipping Hugging Face upload (HF_TOKEN not set). Data saved locally.")
    
    logger.info("Workflow 04_build_sequence_reference complete.")

if __name__ == "__main__":
    main()
