"""Validate required data folders and sample files for EXERKINEMAP workflows."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
required_paths = [
    PROJECT_ROOT / "data" / "raw" / "exerkines" / "sequences",
    PROJECT_ROOT / "data" / "raw" / "exerkines" / "metadata",
    PROJECT_ROOT / "data" / "snRNAseq",
    PROJECT_ROOT / "data" / "xenium",
    PROJECT_ROOT / "data" / "codex",
]

missing = []
for p in required_paths:
    if not p.exists():
        missing.append(str(p))

if missing:
    print("Missing expected data directories:")
    for m in missing:
        print(f" - {m}")
    print("\nCreate these directories and add the expected files. See data/*/README.md for conventions.")
    sys.exit(1)

print("All expected data directories exist. Please populate them with your raw files as required by the workflows.")
