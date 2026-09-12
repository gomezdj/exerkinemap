"""
train_glm.py

Wrapper script for the EXERKINEMAP GLM training workflow.
This script executes workflows/05_train_glm.py from the repository root.
"""

import runpy
import sys
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    workflow_path = repo_root / "workflows" / "05_train_glm.py"
    if not workflow_path.exists():
        raise FileNotFoundError(f"Workflow script not found: {workflow_path}")

    runpy.run_path(str(workflow_path), run_name="__main__")


if __name__ == "__main__":
    main()
