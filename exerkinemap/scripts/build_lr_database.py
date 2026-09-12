"""Build the EXERKINEMAP ligand-receptor interaction network from FANTOM5 data."""

import runpy
import sys
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    workflow_path = repo_root / "workflows" / "08_build_ligand_receptor_network.py"
    if not workflow_path.exists():
        raise FileNotFoundError(f"Workflow script not found: {workflow_path}")

    print("Building ligand-receptor network using the FANTOM5-backed workflow...")
    runpy.run_path(str(workflow_path), run_name="__main__")


if __name__ == "__main__":
    main()
