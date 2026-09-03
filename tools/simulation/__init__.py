"""Evolution-grounded tumor simulation and benchmark-truth generation."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import CopyNumberEvolutionConfig, TumorSimulationConfig
    from .generator import simulate_tumor
    from .output import validate_generated_tumor_directory

_LAZY_EXPORTS = {
    "CopyNumberEvolutionConfig": (".config", "CopyNumberEvolutionConfig"),
    "TumorSimulationConfig": (".config", "TumorSimulationConfig"),
    "simulate_tumor": (".generator", "simulate_tumor"),
    "validate_generated_tumor_directory": (
        ".output",
        "validate_generated_tumor_directory",
    ),
}


def __getattr__(name: str) -> Any:
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
    "CopyNumberEvolutionConfig",
    "TumorSimulationConfig",
    "simulate_tumor",
    "validate_generated_tumor_directory",
]
