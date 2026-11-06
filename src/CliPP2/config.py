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
class DescriptiveScreen:
    """Explicit profile-deviance cutoffs; equality passes every screen."""

    pair_deviance_max: float
    mean_deviance_max: float
    outlier_fraction_max: float

    def __post_init__(self):
        for name in ("pair_deviance_max", "mean_deviance_max", "outlier_fraction_max"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if min(self.pair_deviance_max, self.mean_deviance_max) < 0 or not 0 <= self.outlier_fraction_max <= 1:
            raise ValueError("Deviance cutoffs must be nonnegative and the outlier fraction in [0,1]")


@dataclass(frozen=True, slots=True)
class AssignmentThresholds:
    """Explicit probability, descriptive-fit, tie and sensitivity policies.

    A log-score gap at or below tie_atol remains ambiguous. Residual gates
    bound both mean profile deviance and the outlier fraction over a target's
    informative regions.
    No probability or residual cutoff is inferred from the data.
    """

    posterior_min: float
    screen: DescriptiveScreen
    tie_atol: float
    multiplicity_tie_atol: float
    probability_sensitivity_atol: float

    def __post_init__(self):
        if type(self.screen) is not DescriptiveScreen:
            raise TypeError("Assignment requires a DescriptiveScreen")
        for name in ("posterior_min", "tie_atol", "multiplicity_tie_atol", "probability_sensitivity_atol"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if (not 0 < self.posterior_min <= 1 or min(self.tie_atol, self.multiplicity_tie_atol) < 0
                or not 0 <= self.probability_sensitivity_atol <= 1):
            raise ValueError("Invalid assignment confidence, tie, or sensitivity threshold")

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        values["screen"] = DescriptiveScreen(**values["screen"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class WorkflowConfig:
    """Three-stage policy with explicitly supplied confidence and fit gates.

    Group screens summarize informative members within each requested region;
    every region must pass coverage, mean, and outlier gates. Clonal distance is Euclidean
    on the complete regional vector, with a strict inequality.
    ``fit.max_clusters`` controls stage one; stage two has its own limit.
    """

    assignment: AssignmentThresholds
    contributor_probability_min: float
    contributor_deviance_max: float
    group_screen: DescriptiveScreen
    contributor_tie_atol: float
    fit: FitConfig = field(default_factory=lambda: FitConfig(
        device="cpu", max_clusters=DEFAULT_STAGE_ONE_MAX_CLUSTERS))
    clonal_distance_max: float = 0.05
    min_informative_per_region: int = 1
    min_depth_per_region: int = 1
    stage_two_max_clusters: int = DEFAULT_MAX_CLUSTERS

    def __post_init__(self):
        if (type(self.assignment) is not AssignmentThresholds or type(self.fit) is not FitConfig
                or type(self.group_screen) is not DescriptiveScreen):
            raise TypeError("Workflow requires validated assignment, group-screen and fit records")
        for name in ("contributor_probability_min", "contributor_deviance_max",
                     "contributor_tie_atol", "clonal_distance_max"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0 < self.contributor_probability_min <= 1:
            raise ValueError("contributor_probability_min must be in (0,1]")
        if min(self.contributor_deviance_max, self.contributor_tie_atol) < 0 or self.clonal_distance_max <= 0:
            raise ValueError("Residual thresholds must be nonnegative and clonal distance positive")
        for name in ("min_informative_per_region", "min_depth_per_region", "stage_two_max_clusters"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        values["assignment"] = AssignmentThresholds.from_dict(values["assignment"])
        values["group_screen"] = DescriptiveScreen(**values["group_screen"])
        fit_values = dict(device="cpu", max_clusters=DEFAULT_STAGE_ONE_MAX_CLUSTERS)
        fit_values.update(values.get("fit", {}))
        values["fit"] = FitConfig(**fit_values)
        return cls(**values)


__all__ += ["DescriptiveScreen", "AssignmentThresholds", "WorkflowConfig"]
