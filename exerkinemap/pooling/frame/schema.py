"""Compatibility re-exports for the wearable observation schema."""

from exerkinemap.wearables.schema import (
    CANONICAL_WEARABLE_COLUMNS,
    WearableSchemaError,
    normalize_wearable_frame,
    validate_wearable_frame,
)

__all__ = [
    "CANONICAL_WEARABLE_COLUMNS",
    "WearableSchemaError",
    "normalize_wearable_frame",
    "validate_wearable_frame",
]
