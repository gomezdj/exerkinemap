"""Build a typed human/rat candidate catalog with curated NCBI RefSeq mRNAs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = PROJECT_ROOT / "data" / "reference" / "gdm_exerkine_candidates.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "refseq_sequence_manifest.csv"
NCBI_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
SPECIES = {
    "human": "Homo sapiens",
    "rat": "Rattus norvegicus",
}
NUCLEOTIDE_ALPHABET = frozenset("ACGTUN")


@dataclass(frozen=True)
class Candidate:
    """A curated entity from the supplied EXERKINEMAP candidate list."""

    entity_id: str
    display_name: str
    gene_symbol: str
    entity_type: str
    resolution_status: str
    aliases: str
    categories: str
    notes: str


class RefSeqClient(Protocol):
    """Interface needed to resolve and fetch RefSeq mRNA records."""

    def resolve_mrna_accession(self, gene_symbol: str, organism: str) -> str | None:
        """Resolve an accession for a gene and organism."""

    def fetch_sequences(self, accessions: Iterable[str]) -> dict[str, str]:
        """Fetch sequences indexed by accession."""


def load_candidates(catalog_path: Path) -> list[Candidate]:
    """Load and validate the curated candidate catalog."""
    with catalog_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected_columns = set(Candidate.__dataclass_fields__)
        if reader.fieldnames is None or set(reader.fieldnames) != expected_columns:
            raise ValueError(
                f"{catalog_path} must contain exactly {sorted(expected_columns)} columns."
            )
        candidates = [Candidate(**row) for row in reader]
    entity_ids = [candidate.entity_id for candidate in candidates]
    if len(entity_ids) != len(set(entity_ids)):
        raise ValueError("Candidate catalog contains duplicate entity_id values.")
    return candidates


class NCBIClient:
    """Small retrying NCBI E-utilities client limited to three requests per second."""

    def __init__(self, email: str, delay_seconds: float = 0.35) -> None:
        if not email:
            raise ValueError(
                "Provide --email or set NCBI_EMAIL before retrieving RefSeq records."
            )
        self.email = email
        self.delay_seconds = delay_seconds
        self._last_request_at = 0.0

    def _request(self, endpoint: str, **parameters: str) -> str:
        parameters.update({"tool": "EXERKINEMAP", "email": self.email})
        url = NCBI_BASE_URL + endpoint + "?" + urllib.parse.urlencode(parameters)
        for attempt in range(3):
            wait_seconds = self.delay_seconds - (time.monotonic() - self._last_request_at)
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            try:
                with urllib.request.urlopen(url, timeout=30) as response:
                    self._last_request_at = time.monotonic()
                    return response.read().decode("utf-8")
            except urllib.error.URLError:
                if attempt == 2:
                    raise
                time.sleep(2**attempt)
        raise AssertionError("Unreachable retry state")

    def _json(self, endpoint: str, **parameters: str) -> dict[str, object]:
        return json.loads(self._request(endpoint, retmode="json", **parameters))

    def resolve_mrna_accession(self, gene_symbol: str, organism: str) -> str | None:
        """Return a canonical curated NM_ accession for a gene and organism."""
        term = (
            f"{gene_symbol}[Gene Name] AND {organism}[Organism] "
            "AND refseq[filter] AND biomol_mrna[PROP]"
        )
        search = self._json("esearch.fcgi", db="nuccore", term=term, retmax="200")
        identifiers = search["esearchresult"]["idlist"]  # type: ignore[index]
        if not identifiers:
            return None
        summary = self._json(
            "esummary.fcgi", db="nuccore", id=",".join(identifiers)
        )["result"]  # type: ignore[index]
        candidates = [
            summary[identifier]  # type: ignore[index]
            for identifier in identifiers
            if summary[identifier]["accessionversion"].startswith("NM_")  # type: ignore[index]
            and f"({gene_symbol.upper()})"
            in summary[identifier]["title"].upper()  # type: ignore[index]
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda record: (
                0 if "transcript variant 1" in record["title"].lower() else 1,
                1 if "transcript variant" in record["title"].lower() else 0,
                record["accessionversion"],
            ),
        )["accessionversion"]

    def fetch_sequences(self, accessions: Iterable[str]) -> dict[str, str]:
        """Fetch FASTA sequences for RefSeq accessions in batches."""
        sequences: dict[str, str] = {}
        accession_list = list(accessions)
        for start in range(0, len(accession_list), 100):
            fasta = self._request(
                "efetch.fcgi",
                db="nuccore",
                id=",".join(accession_list[start : start + 100]),
                rettype="fasta",
                retmode="text",
            )
            header: str | None = None
            sequence_parts: list[str] = []
            for line in fasta.splitlines():
                if line.startswith(">"):
                    if header is not None:
                        sequences[header.split()[0]] = "".join(sequence_parts)
                    header = line[1:]
                    sequence_parts = []
                else:
                    sequence_parts.append(line.strip())
            if header is not None:
                sequences[header.split()[0]] = "".join(sequence_parts)
        return sequences


def build_manifest_rows(
    candidates: Iterable[Candidate], client: RefSeqClient
) -> list[dict[str, str | int]]:
    """Resolve every eligible candidate and retain unresolved entities explicitly."""
    rows: list[dict[str, str | int]] = []
    pending: list[tuple[Candidate, str, str, str]] = []
    for candidate in candidates:
        if candidate.resolution_status != "resolve_refseq_mrna":
            rows.append(_unresolved_row(candidate))
            continue
        for species, organism in SPECIES.items():
            accession = client.resolve_mrna_accession(candidate.gene_symbol, organism)
            pending.append((candidate, species, organism, accession or ""))

    sequences = client.fetch_sequences(
        accession for _, _, _, accession in pending if accession
    )
    for candidate, species, _, accession in pending:
        sequence = sequences.get(accession, "")
        status = "resolved_refseq_mrna"
        if not accession:
            status = "not_found_refseq_mrna"
        elif not sequence or set(sequence) - NUCLEOTIDE_ALPHABET:
            raise ValueError(f"NCBI returned an invalid FASTA sequence for {accession}.")
        rows.append(
            _row(
                candidate,
                species=species,
                resolution_status=status,
                accession=accession,
                sequence=sequence,
            )
        )
    return rows


def _unresolved_row(candidate: Candidate) -> dict[str, str | int]:
    return _row(
        candidate,
        species="not_applicable",
        resolution_status=candidate.resolution_status,
        accession="",
        sequence="",
    )


def _row(
    candidate: Candidate,
    *,
    species: str,
    resolution_status: str,
    accession: str,
    sequence: str,
) -> dict[str, str | int]:
    return {
        "entity_id": candidate.entity_id,
        "display_name": candidate.display_name,
        "gene_or_transcript_id": candidate.gene_symbol,
        "species": species,
        "label": 1,
        "label_definition": "candidate_reference_membership_not_empirical_benchmark",
        "label_source": "user_supplied_candidate_list",
        "entity_type": candidate.entity_type,
        "resolution_status": resolution_status,
        "aliases": candidate.aliases,
        "categories": candidate.categories,
        "notes": candidate.notes,
        "refseq_accession": accession,
        "sequence": sequence,
        "source_url": (
            f"https://www.ncbi.nlm.nih.gov/nuccore/{accession}" if accession else ""
        ),
    }


def write_manifest(rows: Iterable[dict[str, str | int]], output_path: Path) -> None:
    """Write the manifest atomically after all records have been resolved."""
    row_list = list(rows)
    if not row_list:
        raise ValueError("Cannot write an empty RefSeq manifest.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row_list[0]))
        writer.writeheader()
        writer.writerows(row_list)
    temporary_path.replace(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh the typed human/rat EXERKINEMAP RefSeq candidate manifest."
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--email", default=os.environ.get("NCBI_EMAIL", ""))
    args = parser.parse_args()

    candidates = load_candidates(args.catalog)
    rows = build_manifest_rows(candidates, NCBIClient(args.email))
    write_manifest(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
