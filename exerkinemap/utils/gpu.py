"""GPU helpers."""

from __future__ import annotations


def is_gpu_available() -> bool:
    return False


def get_gpu_device() -> str:
    return "cpu"
