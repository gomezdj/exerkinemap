"""
sequence_io.py
FASTA file readers and writers for nucleotide and protein sequences.
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_fasta(filepath: Path) -> dict:
    """Loads a FASTA file into a dictionary mapping sequence header to sequence string."""
    filepath = Path(filepath)
    if not filepath.exists():
        logger.error(f"FASTA file not found at {filepath}")
        raise FileNotFoundError(f"FASTA file not found at {filepath}")
        
    logger.info(f"Loading FASTA sequences from {filepath}")
    sequences = {}
    current_header = None
    current_seq = []
    
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_header:
                    sequences[current_header] = "".join(current_seq)
                current_header = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_header:
            sequences[current_header] = "".join(current_seq)
            
    logger.info(f"Loaded {len(sequences)} sequences from FASTA.")
    return sequences

def save_fasta(sequences: dict, filepath: Path) -> None:
    """Saves a dictionary of sequences (header -> string) to a FASTA file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving {len(sequences)} sequences to FASTA at {filepath}")
    
    with open(filepath, "w") as f:
        for header, seq in sequences.items():
            f.write(f">{header}\n{seq}\n")
    logger.info("FASTA file successfully written.")
