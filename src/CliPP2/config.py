"""One immutable, fully resolved configuration for a CliPP2 fit."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Integral
from typing import Final

DEFAULT_MAX_MAJOR_CN: Final = 10
DEFAULT_MAX_CLUSTERS: Final = 20
DEFAULT_STAGE_ONE_MAX_CLUSTERS: Final = 10
# Preserved mixed-CN input bounds use the historical four-copy support cap;
# the tree model gives matched single-state CN its full 1..major support.
MAX_MULTIPLICITY: Final = 4
DEFAULT_DEVICE: Final = "cuda"
DEFAULT_AUDIT_SCOPE: Final = "selected"


def validate_audit_scope(value):
    if value not in ("selected", "all"):
        raise ValueError("audit_scope must be selected or all.")
    return value


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
    audit_scope: str = DEFAULT_AUDIT_SCOPE

    def __post_init__(self) -> None:
        validate_audit_scope(self.audit_scope)
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
    audit_scope: str = DEFAULT_AUDIT_SCOPE,
) -> FitConfig:
    """Construct the only public fit configuration; removed knobs raise TypeError."""
    return FitConfig(device=device, max_major_cn=max_major_cn,
                     max_clusters=max_clusters, cpu_threads=cpu_threads, audit_scope=audit_scope)


__all__ = ["FitConfig", "resolve_fit_config"]


@dataclass(frozen=True, slots=True)
class AssignmentThresholds:
    """Explicit scientific gates; equality passes probability/residual gates.

    A log-score gap at or below tie_atol remains ambiguous. Residual gates
    compare the mean profile deviance over informative regions of a target.
    No probability or residual cutoff is inferred from the data.
    """

    posterior_min: float
    mean_deviance_max: float
    tie_atol: float

    def __post_init__(self):
        for name in ("posterior_min", "mean_deviance_max", "tie_atol"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0 <= self.posterior_min <= 1 or self.mean_deviance_max < 0 or self.tie_atol < 0:
            raise ValueError("Invalid assignment probability, residual, or tie threshold")


@dataclass(frozen=True, slots=True)
class WorkflowConfig:
    """Three-stage policy with explicitly supplied confidence and fit gates.

    Group residuals are means within each requested region, and every region
    must pass both coverage and residual gates. Clonal distance is Euclidean
    on the complete regional vector, with a strict inequality.
    ``fit.max_clusters`` controls stage one; stage two has its own limit.
    """

    assignment: AssignmentThresholds
    contributor_probability_min: float
    contributor_deviance_max: float
    group_mean_deviance_max: float
    fit: FitConfig = field(default_factory=lambda: FitConfig(
        device="cpu", max_clusters=DEFAULT_STAGE_ONE_MAX_CLUSTERS))
    clonal_distance_max: float = 0.05
    min_informative_per_region: int = 1
    min_depth_per_region: int = 1
    stage_two_max_clusters: int = DEFAULT_MAX_CLUSTERS

    def __post_init__(self):
        if type(self.assignment) is not AssignmentThresholds or type(self.fit) is not FitConfig:
            raise TypeError("Workflow requires AssignmentThresholds and FitConfig records")
        for name in ("contributor_probability_min", "contributor_deviance_max",
                     "group_mean_deviance_max", "clonal_distance_max"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0 <= self.contributor_probability_min <= 1:
            raise ValueError("contributor_probability_min must be in [0,1]")
        if min(self.contributor_deviance_max, self.group_mean_deviance_max) < 0 or self.clonal_distance_max <= 0:
            raise ValueError("Residual thresholds must be nonnegative and clonal distance positive")
        for name in ("min_informative_per_region", "min_depth_per_region", "stage_two_max_clusters"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        values["assignment"] = AssignmentThresholds(**values["assignment"])
        fit_values = dict(device="cpu", max_clusters=DEFAULT_STAGE_ONE_MAX_CLUSTERS)
        fit_values.update(values.get("fit", {}))
        values["fit"] = FitConfig(**fit_values)
        return cls(**values)


__all__ += ["AssignmentThresholds", "WorkflowConfig"]
