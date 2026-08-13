"""
writers.py
General file writing utilities for tabular and JSON data formats.
"""
import logging
import json
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

def write_csv(df: pd.DataFrame, filepath: Path, index: bool = False, **kwargs) -> None:
    """Writes a pandas DataFrame to a CSV file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing DataFrame to CSV at {filepath}")
    df.to_csv(filepath, index=index, **kwargs)

def write_json(data: dict, filepath: Path, indent: int = 4) -> None:
    """Writes a dictionary to a JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing dictionary to JSON at {filepath}")
    with open(filepath, "w") as f:
        json.dump(data, f, indent=indent)
