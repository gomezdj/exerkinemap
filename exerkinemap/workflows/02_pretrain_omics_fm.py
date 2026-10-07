"""Continue masked-language-model training on the downloaded UniProt protein FASTA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exerkinemap.foundational_model.pretraining import DEFAULT_MODEL, OmicsPretrainer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data/raw/sequences/protein/uniprot_human_proteome.fasta.gz",
        help="Plain or gzipped protein FASTA corpus (default: downloaded human UniProt reference).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results/pretrained_fm/esm2_t6_8M_uniprot",
    )
    parser.add_argument("--model-checkpoint", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-records", type=int, help="Optional limit for a small local run.")
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing files in output-dir.")
    args = parser.parse_args(argv)

    pretrainer = OmicsPretrainer(model_checkpoint=args.model_checkpoint)
    output_directory = pretrainer.execute_pretraining(
        dataset_path=args.input,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_records=args.max_records,
        learning_rate=args.learning_rate,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"Protein MLM continued-training complete: {output_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
