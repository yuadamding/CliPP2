"""CliPP2: observed-data pairwise fusion for multi-region subclonal reconstruction."""

from __future__ import annotations

from importlib import import_module

from ._version import __version__

_EXPORTS = {
    "FitConfig": ".config",
    "TumorData": ".io.data",
    "process_tumor": ".api",
    "load_tumor_txt": ".io.tumor_txt",
    "resolve_fit_config": ".config",
}

__all__ = [*_EXPORTS, "__version__"]


def __getattr__(name: str):
    """Load fitting dependencies only when their public operation is used."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
