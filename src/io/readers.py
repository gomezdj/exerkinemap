"""
readers.py
General file reading utilities for CSV, tabular, and JSON data formats.
"""
import logging
import json
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

def read_csv(filepath: Path, **kwargs) -> pd.DataFrame:
    """Reads a CSV or tabular file into a pandas DataFrame."""
    filepath = Path(filepath)
    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"File not found at {filepath}")
    logger.info(f"Reading CSV file from {filepath}")
    return pd.read_csv(filepath, **kwargs)

def read_json(filepath: Path) -> dict:
    """Reads a JSON file into a Python dictionary."""
    filepath = Path(filepath)
    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"File not found at {filepath}")
    logger.info(f"Reading JSON file from {filepath}")
    with open(filepath, "r") as f:
        data = json.load(f)
    return data
