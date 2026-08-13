"""Configuration helpers."""

from __future__ import annotations

from typing import Any, Dict


def get_config() -> Dict[str, Any]:
    return {}


def load_config(path: str | None = None) -> Dict[str, Any]:
    return get_config()
