"""
metadata.py
Utilities for loading and validating experimental and sample metadata manifests.
"""
import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

def load_metadata(filepath: Path) -> pd.DataFrame:
    """Loads experimental metadata manifest."""
    filepath = Path(filepath)
    if not filepath.exists():
        logger.error(f"Metadata file not found at {filepath}")
        raise FileNotFoundError(f"Metadata file not found at {filepath}")
    logger.info(f"Loading metadata from {filepath}")
    return pd.read_csv(filepath)

def validate_metadata(df: pd.DataFrame, required_columns: list) -> bool:
    """Validates that all mandatory columns are present in the metadata dataframe."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        logger.error(f"Metadata validation failed. Missing required columns: {missing}")
        return False
    logger.info("Metadata validation passed successfully.")
    return True
