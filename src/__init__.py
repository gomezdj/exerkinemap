"""Top-level EXERKINEMAP package.

This package exposes a small public API for notebooks and scripts.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("exerkinemap")
except PackageNotFoundError:  # pragma: no cover - local editable install fallback
    __version__ = "0.1.0"

__all__ = ["__version__"]