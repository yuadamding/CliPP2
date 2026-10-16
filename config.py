"""One immutable, fully resolved configuration for a CliPP2 fit."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Final

DEFAULT_MAX_MAJOR_CN: Final = 4
DEFAULT_DEVICE: Final = "cuda"
DEFAULT_MAX_CLUSTERS: Final = 10
MAX_CLUSTERS: Final = 50
EXHAUSTIVE_MAX_CLUSTERS: Final = 10


@dataclass(frozen=True, slots=True)
class SearchPolicy:
    """Search allowances change coverage, never the statistical scoring rule."""
    mode: str = 'auto'
    width: int = 16
    proposals_per_state: int = 4
    sweeps: int = 2

    def __post_init__(self):
        if self.mode not in ('auto', 'exhaustive', 'bounded'):
            raise ValueError('search_policy must be auto, exhaustive or bounded')
        for name in ('width', 'proposals_per_state', 'sweeps'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'Search {name} must be a positive integer')

    @property
    def partition_limit_per_q(self):
        return self.width*self.proposals_per_state*self.sweeps

    def resolve(self, max_clusters):
        mode = self.mode
        if mode == 'auto':
            mode = 'exhaustive' if max_clusters <= EXHAUSTIVE_MAX_CLUSTERS else 'bounded'
        if mode == 'exhaustive' and max_clusters > EXHAUSTIVE_MAX_CLUSTERS:
            raise ValueError(f'Exhaustive search is restricted to max_clusters <= {EXHAUSTIVE_MAX_CLUSTERS}')
        return SearchPolicy(mode, self.width, self.proposals_per_state, self.sweeps)


def validate_budget(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number of GiB")
    return float(value)


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
    ram_budget_gib: float = 1.0
    vram_budget_gib: float = 0.0
    search_policy: str = 'auto'
    search_width: int = 16
    search_proposals: int = 4
    search_sweeps: int = 2
    qualify_device_weights: bool = False

    def __post_init__(self) -> None:
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda.")
        if not isinstance(self.verbose, bool):
            raise TypeError("verbose must be a boolean.")
        object.__setattr__(self, "max_major_cn", validate_max_major_cn(self.max_major_cn))
        if type(self.max_clusters) is not int or not 1 <= self.max_clusters <= MAX_CLUSTERS:
            raise ValueError(f"max_clusters must be an integer in 1..{MAX_CLUSTERS}.")
        for name in ('ram_budget_gib', 'vram_budget_gib'):
            object.__setattr__(self, name, validate_budget(getattr(self, name), name))
        self.resolved_search()
        if not isinstance(self.qualify_device_weights, bool):
            raise ValueError('qualify_device_weights must be a boolean')
        if self.qualify_device_weights and (self.device != 'cuda' or self.vram_budget_gib == 0):
            raise ValueError('Device weight qualification requires CUDA and a positive VRAM budget')

    def resolved_search(self):
        return SearchPolicy(self.search_policy, self.search_width,
                            self.search_proposals, self.search_sweeps).resolve(self.max_clusters)


def resolve_fit_config(
    *, device: str = DEFAULT_DEVICE, verbose: bool = False,
    max_major_cn: int = DEFAULT_MAX_MAJOR_CN,
    max_clusters: int = DEFAULT_MAX_CLUSTERS,
    ram_budget_gib: float = 1.0, vram_budget_gib: float = 0.0,
    search_policy: str = 'auto', search_width: int = 16,
    search_proposals: int = 4, search_sweeps: int = 2,
    qualify_device_weights: bool = False,
) -> FitConfig:
    """Construct the only public fit configuration; removed knobs raise TypeError."""
    return FitConfig(device=device, verbose=verbose, max_major_cn=max_major_cn,
                     max_clusters=max_clusters, ram_budget_gib=ram_budget_gib,
                     vram_budget_gib=vram_budget_gib, search_policy=search_policy,
                     search_width=search_width, search_proposals=search_proposals,
                     search_sweeps=search_sweeps, qualify_device_weights=qualify_device_weights)


__all__ = ["FitConfig", "resolve_fit_config"]
