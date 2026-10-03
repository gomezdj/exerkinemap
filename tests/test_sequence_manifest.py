from __future__ import annotations

import csv
from pathlib import Path

import pytest

from exerkinemap.benchmarks.sequence_manifest import (
    build_benchmark_rows,
    load_raw_controls,
    write_benchmark_manifest,
)


def _control(control_id: str, sequence: str) -> dict[str, str]:
    return {
        "control_id": control_id,
        "species": "human",
        "sequence": sequence,
        "refseq_accession": "",
        "source_url": "",
        "control_definition": "prespecified non-exerkine control",
    }


def _candidate(entity_id: str, sequence: str) -> dict[str, str]:
    return {
        "entity_id": entity_id,
        "species": "human",
        "refseq_accession": f"NM_{entity_id}",
        "sequence": sequence,
        "source_url": f"https://example.org/{entity_id}",
    }


def test_raw_controls_require_empirical_definition(tmp_path: Path) -> None:
    path = tmp_path / "controls.csv"
    path.write_text(
        "control_id,species,sequence,refseq_accession,source_url,control_definition\n"
        "control-1,human,ATGC,,, \n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="lacks a control_definition"):
        load_raw_controls(path)


def test_benchmark_rows_preserve_raw_control_provenance() -> None:
    rows = build_benchmark_rows(
        [_candidate("FGF21", "AAAA"), _candidate("FGF23", "AAAT"), _candidate("FNDC5", "AATA")],
        [_control("control-1", "CCCC"), _control("control-2", "CCCG"), _control("control-3", "CCCT")],
    )

    assert [row["label"] for row in rows] == [1, 1, 1, 0, 0, 0]
    assert rows[-1]["label_source"] == "raw_control"
    assert rows[-1]["control_definition"] == "prespecified non-exerkine control"


def test_duplicate_control_sequence_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate selected positive sequences"):
        build_benchmark_rows(
            [_candidate("FGF21", "AAAA"), _candidate("FGF23", "AAAT"), _candidate("FNDC5", "AATA")],
            [_control("control-1", "AAAA"), _control("control-2", "CCCG"), _control("control-3", "CCCT")],
        )


def test_processed_benchmark_manifest_is_written_atomically(tmp_path: Path) -> None:
    rows = build_benchmark_rows(
        [_candidate("FGF21", "AAAA"), _candidate("FGF23", "AAAT"), _candidate("FNDC5", "AATA")],
        [_control("control-1", "CCCC"), _control("control-2", "CCCG"), _control("control-3", "CCCT")],
    )
    output_path = tmp_path / "benchmark.csv"

    write_benchmark_manifest(rows, output_path)

    with output_path.open(newline="", encoding="utf-8") as handle:
        written_rows = list(csv.DictReader(handle))
    assert len(written_rows) == 6
    assert {row["label"] for row in written_rows} == {"0", "1"}
