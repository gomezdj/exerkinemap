"""Ingest a DHDR-style signal CSV and derive causal window features."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from exerkinemap.wearables.dhdr_tools import load_dhdr_signal_csv
from exerkinemap.wearables.flirt_features import causal_window_features


LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess a DHDR-style wearable signal into causal windows."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", default="DHDR_CUSTOM")
    parser.add_argument("--subject-column", default="subject_id")
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--value-column", default="value")
    parser.add_argument("--signal", required=True)
    parser.add_argument("--unit")
    parser.add_argument("--window", default="30min")
    parser.add_argument("--min-periods", type=int, default=1)
    args = parser.parse_args()

    observations = load_dhdr_signal_csv(
        args.input,
        source=args.source,
        subject_column=args.subject_column,
        timestamp_column=args.timestamp_column,
        value_column=args.value_column,
        signal=args.signal,
        unit=args.unit,
    )
    features = causal_window_features(
        observations, window=args.window, min_periods=args.min_periods
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)
    LOGGER.info("Wrote %d causal feature windows to %s", len(features), args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()