"""CliPP2: observed-data pairwise fusion for multi-region subclonal reconstruction."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from ._version import __version__

if TYPE_CHECKING:
    from .api import fit_fixed_objective
    from .config import CheckpointRequest, FitConfig, RunConfig, resolve_fit_config
    from .core.fusion.types import RawFit
    from .io.data import EmissionPaths, ExclusionCode, TumorData
    from .io.tumor_txt import load_tumor_txt

_LAZY_EXPORTS = {
    "CheckpointRequest": (".config", "CheckpointRequest"),
    "EmissionPaths": (".io.data", "EmissionPaths"),
    "ExclusionCode": (".io.data", "ExclusionCode"),
    "FitConfig": (".config", "FitConfig"),
    "RawFit": (".core.fusion.types", "RawFit"),
    "RunConfig": (".config", "RunConfig"),
    "TumorData": (".io.data", "TumorData"),
    "fit_fixed_objective": (".api", "fit_fixed_objective"),
    "load_tumor_txt": (".io.tumor_txt", "load_tumor_txt"),
    "resolve_fit_config": (".config", "resolve_fit_config"),
}


def __getattr__(name: str) -> Any:
    """Load public API objects only when callers request them."""

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

__all__ = [
    "CheckpointRequest",
    "EmissionPaths",
    "ExclusionCode",
    "FitConfig",
    "RawFit",
    "RunConfig",
    "TumorData",
    "__version__",
    "fit_fixed_objective",
    "load_tumor_txt",
    "resolve_fit_config",
]
