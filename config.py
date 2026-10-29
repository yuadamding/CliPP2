"""One immutable, fully resolved configuration for a CliPP2 fit."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Final

DEFAULT_MAX_MAJOR_CN: Final = 10
DEFAULT_MAX_CLUSTERS: Final = 20
# Preserved mixed-CN input bounds use the historical four-copy support cap;
# the tree model gives matched single-state CN its full 1..major support.
MAX_MULTIPLICITY: Final = 4
DEFAULT_DEVICE: Final = "cuda"


def validate_max_major_cn(value: int) -> int:
    """Require an explicit positive integer input-eligibility cutoff."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError("max_major_cn must be a positive integer.")
    return int(value)


@dataclass(frozen=True, slots=True)
class FitConfig:
    """Public execution and CN eligibility options; numerical policy is fixed."""

    device: str = DEFAULT_DEVICE
    verbose: bool = False
    max_major_cn: int = DEFAULT_MAX_MAJOR_CN
    max_clusters: int = DEFAULT_MAX_CLUSTERS
    grid_cache_mb: int = 64
    device_cache_mb: int = 256
    workspace_mb: int = 64
    native_host_memory_mb: int | None = None
    device_memory_mb: int | None = None
    cpu_threads: int | None = None
    execution_policy: str = "auto"

    def __post_init__(self) -> None:
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda.")
        if self.execution_policy not in ("auto", "reference", "candidate"):
            raise ValueError("execution_policy must be auto, reference or candidate.")
        if self.execution_policy == "candidate" and self.device != "cuda":
            raise ValueError("The candidate execution policy requires device=cuda.")
        if not isinstance(self.verbose, bool):
            raise TypeError("verbose must be a boolean.")
        object.__setattr__(self, "max_major_cn", validate_max_major_cn(self.max_major_cn))
        if type(self.max_clusters) is not int or self.max_clusters < 1:
            raise ValueError("max_clusters must be a positive integer.")
        for name in ("grid_cache_mb", "device_cache_mb", "workspace_mb"):
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name == "workspace_mb" else 0):
                raise ValueError(f"{name} must be a nonnegative integer (workspace_mb must be positive).")
        for name in ("native_host_memory_mb", "device_memory_mb", "cpu_threads"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or not 1 <= value <= 2**31 - 1):
                raise ValueError(f"{name} must be a positive int32 or None (auto).")


def resolve_fit_config(
    *, device: str = DEFAULT_DEVICE, verbose: bool = False,
    max_major_cn: int = DEFAULT_MAX_MAJOR_CN,
    max_clusters: int = DEFAULT_MAX_CLUSTERS,
    grid_cache_mb: int = 64, device_cache_mb: int = 256, workspace_mb: int = 64,
    native_host_memory_mb: int | None = None, device_memory_mb: int | None = None,
    cpu_threads: int | None = None,
    execution_policy: str = "auto",
) -> FitConfig:
    """Construct the only public fit configuration; removed knobs raise TypeError."""
    return FitConfig(device=device, verbose=verbose, max_major_cn=max_major_cn,
                     max_clusters=max_clusters, grid_cache_mb=grid_cache_mb,
                     device_cache_mb=device_cache_mb, workspace_mb=workspace_mb,
                     native_host_memory_mb=native_host_memory_mb,
                     device_memory_mb=device_memory_mb, cpu_threads=cpu_threads,
                     execution_policy=execution_policy)


__all__ = ["FitConfig", "resolve_fit_config"]
