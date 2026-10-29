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
    max_major_cn: int = DEFAULT_MAX_MAJOR_CN
    max_clusters: int = DEFAULT_MAX_CLUSTERS
    cpu_threads: int | None = None

    def __post_init__(self) -> None:
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda.")
        object.__setattr__(self, "max_major_cn", validate_max_major_cn(self.max_major_cn))
        if type(self.max_clusters) is not int or self.max_clusters < 1:
            raise ValueError("max_clusters must be a positive integer.")
        if self.cpu_threads is not None and (
                type(self.cpu_threads) is not int or not 1 <= self.cpu_threads <= 2**31 - 1):
            raise ValueError("cpu_threads must be a positive int32 or None (auto).")


def resolve_fit_config(
    *, device: str = DEFAULT_DEVICE,
    max_major_cn: int = DEFAULT_MAX_MAJOR_CN,
    max_clusters: int = DEFAULT_MAX_CLUSTERS,
    cpu_threads: int | None = None,
) -> FitConfig:
    """Construct the only public fit configuration; removed knobs raise TypeError."""
    return FitConfig(device=device, max_major_cn=max_major_cn,
                     max_clusters=max_clusters, cpu_threads=cpu_threads)


__all__ = ["FitConfig", "resolve_fit_config"]
