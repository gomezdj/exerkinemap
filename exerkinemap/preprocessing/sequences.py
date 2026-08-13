"""Helpers for loading and cleaning biological sequence data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

def process_dna_sequence(sequence, chunk_size=6, k=3, stopwords=None):
    if stopwords is None:
        stopwords = set()

    # Step 1: Fragment into chunks
    chunks = [sequence[i:i + chunk_size] for i in range(0, len(sequence), chunk_size)]
    
    cleaned_chunks = []
    all_kmers = []

    for chunk in chunks:
        # Step 2: Generate 3-mers using a sliding window
        chunk_kmers = [chunk[i:i + k] for i in range(len(chunk) - k + 1)]
        
        # Step 3: Data Cleaning (Stopword removal)
        filtered_kmers = [kmer for kmer in chunk_kmers if kmer not in stopwords]
        
        cleaned_chunks.append(filtered_kmers)
        all_kmers.extend(filtered_kmers)

    return chunks, cleaned_chunks, all_kmers


def load_fasta_sequences(path: Any, *, lowercase: bool = True) -> List[dict]:
    """Load a FASTA file into a list of dictionaries if Biopython is available."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        from Bio import SeqIO
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError("biopython is required to load FASTA files") from exc

    records = []
    for record in SeqIO.parse(path, "fasta"):
        records.append(
            {
                "id": record.id,
                "description": record.description,
                "sequence": clean_sequence(str(record.seq), lowercase=lowercase),
            }
        )
    return records
