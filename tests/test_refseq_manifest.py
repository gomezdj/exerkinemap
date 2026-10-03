from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from exerkinemap.scripts.refresh_refseq_manifest import (
    Candidate,
    DEFAULT_CATALOG,
    build_manifest_rows,
    load_candidates,
    write_manifest,
)


class FakeRefSeqClient:
    def resolve_mrna_accession(self, gene_symbol: str, organism: str) -> str | None:
        if gene_symbol == "MISSING":
            return None
        return f"NM_{gene_symbol}_{organism.split()[0]}"

    def fetch_sequences(self, accessions: Iterable[str]) -> dict[str, str]:
        return {accession: "ATGC" for accession in accessions}


def test_catalog_preserves_corrected_and_unresolved_user_terms() -> None:
    candidates = {candidate.entity_id: candidate for candidate in load_candidates(DEFAULT_CATALOG)}

    assert {
        "IL6",
        "FGF19",
        "RARRES2",
        "LEP",
        "ADIPOQ",
        "FNDC5",
        "CFD",
        "FGF21",
        "FGF23",
    } <= set(candidates)
    assert "HPS" in candidates["FGL1"].aliases
    assert "ADRPIN" in candidates["ENHO"].aliases
    assert candidates["FGFR21"].resolution_status == "ambiguous_or_invalid_symbol"
    assert candidates["AMPK"].gene_symbol == ""


def test_manifest_rows_keep_non_gene_entities_blank() -> None:
    candidates = [
        Candidate(
            entity_id="GENE",
            display_name="Gene",
            gene_symbol="GENE",
            entity_type="protein_coding_gene",
            resolution_status="resolve_refseq_mrna",
            aliases="",
            categories="",
            notes="",
        ),
        Candidate(
            entity_id="METABOLITE",
            display_name="Metabolite",
            gene_symbol="",
            entity_type="metabolite",
            resolution_status="non_gene_entity",
            aliases="",
            categories="",
            notes="",
        ),
    ]

    rows = build_manifest_rows(candidates, FakeRefSeqClient())

    gene_rows = [row for row in rows if row["entity_id"] == "GENE"]
    metabolite_row = next(row for row in rows if row["entity_id"] == "METABOLITE")
    assert {row["species"] for row in gene_rows} == {"human", "rat"}
    assert all(row["sequence"] == "ATGC" for row in gene_rows)
    assert metabolite_row["species"] == "not_applicable"
    assert metabolite_row["refseq_accession"] == ""
    assert metabolite_row["sequence"] == ""


def test_write_manifest_retains_typed_columns(tmp_path: Path) -> None:
    output_path = tmp_path / "manifest.csv"
    rows = build_manifest_rows(
        [
            Candidate(
                entity_id="GENE",
                display_name="Gene",
                gene_symbol="GENE",
                entity_type="protein_coding_gene",
                resolution_status="resolve_refseq_mrna",
                aliases="",
                categories="",
                notes="",
            )
        ],
        FakeRefSeqClient(),
    )

    write_manifest(rows, output_path)

    with output_path.open(newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    assert len(manifest_rows) == 2
    assert {"entity_id", "entity_type", "resolution_status", "sequence"} <= set(
        manifest_rows[0]
    )
