"""Helpers for loading and cleaning biological sequence data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional


def clean_sequence(sequence: str, lowercase: bool = True, remove_non_acgt: bool = False) -> str:
    """Clean a nucleotide/protein sequence string for downstream modeling."""
    if sequence is None:
        return ""
    cleaned = str(sequence).strip()
    if lowercase:
        cleaned = cleaned.lower()
    if remove_non_acgt:
        cleaned = "".join(ch for ch in cleaned if ch in set("acgt"))
    return cleaned


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
