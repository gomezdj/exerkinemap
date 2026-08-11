"""Compatibility shim for importing exerkinemap.io."""

import importlib


def _build_stub(name: str):
    def _stub(*args, **kwargs):
        raise ImportError(
            f"Optional dependency missing while importing exerkinemap.io.{name}; "
            "install the scientific dependencies declared in requirements.txt"
        )

    _stub.__name__ = name
    return _stub


try:
    module = importlib.import_module("src.io")
    globals().update({name: getattr(module, name) for name in dir(module) if not name.startswith("_")})
    __all__ = [name for name in dir(module) if not name.startswith("_")]
except Exception:
    read_csv = _build_stub("read_csv")
    read_json = _build_stub("read_json")
    write_csv = _build_stub("write_csv")
    write_json = _build_stub("write_json")
    load_anndata = _build_stub("load_anndata")
    save_anndata = _build_stub("save_anndata")
    load_fasta = _build_stub("load_fasta")
    save_fasta = _build_stub("save_fasta")
    load_metadata = _build_stub("load_metadata")
    validate_metadata = _build_stub("validate_metadata")
    __all__ = [
        "read_csv",
        "read_json",
        "write_csv",
        "write_json",
        "load_anndata",
        "save_anndata",
        "load_fasta",
        "save_fasta",
        "load_metadata",
        "validate_metadata",
    ]
