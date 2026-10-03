"""Build an explicit binary benchmark manifest from candidates and raw controls."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Sequence


CONTROL_COLUMNS = (
    "control_id",
    "species",
    "sequence",
    "refseq_accession",
    "source_url",
    "control_definition",
)
NUCLEOTIDE_ALPHABET = frozenset("ACGTUN")


def load_raw_controls(path: Path) -> list[dict[str, str]]:
    """Load immutable raw controls without changing their provenance."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Raw control manifest was not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Raw control manifest is empty.")
        missing = set(CONTROL_COLUMNS) - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Raw control manifest is missing columns: {sorted(missing)}"
            )
        controls = [{column: (row[column] or "").strip() for column in CONTROL_COLUMNS} for row in reader]
    if not controls:
        raise ValueError(
            "Raw control manifest has no controls. Add controls with an explicit "
            "control_definition before building a benchmark."
        )
    identifiers = [control["control_id"] for control in controls]
    if any(not identifier for identifier in identifiers):
        raise ValueError("Every raw control must have a non-empty control_id.")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Raw control manifest contains duplicate control_id values.")
    for control in controls:
        sequence = control["sequence"].upper()
        if not sequence:
            raise ValueError(
                f"Raw control {control['control_id']} has no nucleotide sequence."
            )
        if set(sequence) - NUCLEOTIDE_ALPHABET:
            raise ValueError(
                f"Raw control {control['control_id']} contains unsupported nucleotide symbols."
            )
        if not control["control_definition"]:
            raise ValueError(
                f"Raw control {control['control_id']} lacks a control_definition."
            )
        control["sequence"] = sequence
    return controls


def load_candidate_positives(
    path: Path, entity_ids: Sequence[str]
) -> list[dict[str, str]]:
    """Select RefSeq-backed candidate records by explicit entity identifier."""
    selected_ids = set(entity_ids)
    if not selected_ids:
        raise ValueError("Provide at least one --positive-entity-id.")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Candidate manifest was not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "entity_id",
            "species",
            "resolution_status",
            "refseq_accession",
            "sequence",
            "source_url",
        }
        if reader.fieldnames is None:
            raise ValueError("Candidate manifest is empty.")
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Candidate manifest is missing columns: {sorted(missing)}"
            )
        candidates = [
            row
            for row in reader
            if row["entity_id"] in selected_ids
            and row["resolution_status"] == "resolved_refseq_mrna"
            and row["sequence"]
        ]
    found_ids = {candidate["entity_id"] for candidate in candidates}
    missing_ids = selected_ids - found_ids
    if missing_ids:
        raise ValueError(
            "No RefSeq-backed candidate records were found for: "
            f"{sorted(missing_ids)}"
        )
    return candidates


def build_benchmark_rows(
    candidates: Iterable[dict[str, str]], controls: Iterable[dict[str, str]]
) -> list[dict[str, str | int]]:
    """Construct labeled rows while retaining distinct source provenance."""
    candidate_rows = list(candidates)
    control_rows = list(controls)
    if len(candidate_rows) < 3 or len(control_rows) < 3:
        raise ValueError(
            "A benchmark manifest requires at least three positives and three controls."
        )

    positive_sequences = {candidate["sequence"].upper() for candidate in candidate_rows}
    control_sequences = {control["sequence"].upper() for control in control_rows}
    overlap = positive_sequences & control_sequences
    if overlap:
        raise ValueError(
            "Raw controls duplicate selected positive sequences and cannot define a "
            f"binary benchmark ({len(overlap)} duplicate sequence(s))."
        )

    rows: list[dict[str, str | int]] = []
    for candidate in candidate_rows:
        rows.append(
            {
                "record_id": f"{candidate['entity_id']}:{candidate['species']}:{candidate['refseq_accession']}",
                "entity_id": candidate["entity_id"],
                "species": candidate["species"],
                "label": 1,
                "label_definition": "selected_reference_candidate_vs_control",
                "label_source": "selected_refseq_candidate",
                "refseq_accession": candidate["refseq_accession"],
                "sequence": candidate["sequence"].upper(),
                "source_url": candidate["source_url"],
                "control_definition": "",
            }
        )
    for control in control_rows:
        rows.append(
            {
                "record_id": control["control_id"],
                "entity_id": "",
                "species": control["species"],
                "label": 0,
                "label_definition": "selected_reference_candidate_vs_control",
                "label_source": "raw_control",
                "refseq_accession": control["refseq_accession"],
                "sequence": control["sequence"].upper(),
                "source_url": control["source_url"],
                "control_definition": control["control_definition"],
            }
        )
    return rows


def write_benchmark_manifest(
    rows: Iterable[dict[str, str | int]], output_path: Path
) -> None:
    """Atomically write a processed benchmark manifest."""
    row_list = list(rows)
    if not row_list:
        raise ValueError("Cannot write an empty benchmark manifest.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row_list[0]))
        writer.writeheader()
        writer.writerows(row_list)
    temporary_path.replace(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a processed binary sequence benchmark from raw controls."
    )
    parser.add_argument(
        "--controls",
        type=Path,
        default=Path("data/raw/benchmarking/sequence_controls.csv"),
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("data/processed/refseq_sequence_manifest.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/benchmarks/gdm_sequence_benchmark.csv"),
    )
    parser.add_argument(
        "--positive-entity-id",
        action="append",
        dest="positive_entity_ids",
        required=True,
        help="Candidate entity_id to include as a positive; repeat for each ID.",
    )
    args = parser.parse_args()

    controls = load_raw_controls(args.controls)
    candidates = load_candidate_positives(args.candidates, args.positive_entity_ids)
    rows = build_benchmark_rows(candidates, controls)
    write_benchmark_manifest(rows, args.output)
    print(
        f"Wrote {len(rows)} rows ({len(candidates)} positives and {len(controls)} controls) "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
