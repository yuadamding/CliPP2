"""One immutable, fully resolved configuration for a CliPP2 fit."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Final

DEFAULT_MAX_MAJOR_CN: Final = None
DEFAULT_MAX_CLUSTERS: Final = 20
DEFAULT_STAGE_ONE_MAX_CLUSTERS: Final = 10
DEFAULT_DEVICE: Final = "cpu"
DEFAULT_MISSING_COUNTS: Final = "zero_alt"
PROTOCOL_VERSION: Final = "2.5"
STAGE_TWO_CLONAL_MERGE_DISTANCE: Final = 0.05
STAGE_THREE_REFERENCE_POLICY: Final = "single_fixed_clonal_stage_two_l2_lt_0p05_v2"
ASSIGNMENT_METHODS: Final = ("marginal", "profile")
SELECTION_CRITERION: Final = "marginal_mixture_bic_at_fitted_centers_v1"


def validate_max_multiplicity_difference(value):
    if value is not None and (type(value) is not int or value != 1):
        raise ValueError("max_multiplicity_difference must be 1 or None (independent)")
    return value


def fitting_identity(method="marginal", selection=SELECTION_CRITERION,
                     max_multiplicity_difference=1):
    if not isinstance(method, str) or method not in ("marginal", "profile"):
        raise ValueError("fit_multiplicity must be marginal or profile")
    if selection != SELECTION_CRITERION:
        raise ValueError(f"selection_criterion must be {SELECTION_CRITERION}")
    validate_max_multiplicity_difference(max_multiplicity_difference)
    if max_multiplicity_difference == 1:
        return {"fit_objective_id": ("joint_range_one_profile_v1" if method == "profile"
                                     else "joint_range_one_marginal_v1"),
                "selection_criterion_id": selection,
                "initialization_policy_id": "joint_pooled_posterior_pilot_v1",
                "scalar_refit_policy_id": ("joint_profile_coordinate_envelope_v1" if method == "profile"
                                           else "joint_marginal_em_stationary_v1"),
                "max_multiplicity_difference": 1,
                "multiplicity_policy_id": "joint_range_one_v1",
                "multiplicity_normalization_id": "uniform_joint_vectors_v1"}
    return {"fit_objective_id": ("integer_multiplicity_profile_v1" if method == "profile"
                                 else "uniform_multiplicity_marginal_v1"),
            "selection_criterion_id": selection,
            "initialization_policy_id": "existing_marginal_pooled_pilot_v1",
            "scalar_refit_policy_id": ("profile_branch_interval_search_v1" if method == "profile"
                                       else "marginal_grid_interval_search_v1"),
            "max_multiplicity_difference": None,
            "multiplicity_policy_id": "independent_regions_v1",
            "multiplicity_normalization_id": "uniform_interval_width_v1"}


def validate_assignment_method(value):
    if not isinstance(value, str) or value not in ASSIGNMENT_METHODS:
        raise ValueError("assignment_method must be marginal or profile")
    return value


def validate_missing_counts(value):
    if value not in ("zero_alt", "unobserved"):
        raise ValueError("missing_counts must be zero_alt or unobserved.")
    return value


def validate_max_major_cn(value: int | None) -> int | None:
    """None keeps all valid CN; a positive integer declares a population cutoff."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError("max_major_cn must be None (unrestricted) or a positive integer.")
    return int(value)


@dataclass(frozen=True, slots=True)
class FitConfig:
    """Public execution and CN eligibility options; numerical policy is fixed."""

    device: str = DEFAULT_DEVICE
    max_major_cn: int | None = DEFAULT_MAX_MAJOR_CN
    max_clusters: int = DEFAULT_MAX_CLUSTERS
    cpu_threads: int | None = None
    fit_multiplicity: str = "marginal"
    selection_criterion: str = SELECTION_CRITERION
    max_multiplicity_difference: int | None = 1

    def __post_init__(self) -> None:
        fitting_identity(self.fit_multiplicity, self.selection_criterion, self.max_multiplicity_difference)
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
    max_major_cn: int | None = DEFAULT_MAX_MAJOR_CN,
    max_clusters: int = DEFAULT_MAX_CLUSTERS,
    cpu_threads: int | None = None,
    fit_multiplicity: str = "marginal",
    selection_criterion: str = SELECTION_CRITERION,
    max_multiplicity_difference: int | None = 1,
) -> FitConfig:
    """Construct the only public fit configuration; removed knobs raise TypeError."""
    return FitConfig(device=device, max_major_cn=max_major_cn,
                     max_clusters=max_clusters, cpu_threads=cpu_threads,
                     fit_multiplicity=fit_multiplicity, selection_criterion=selection_criterion,
                     max_multiplicity_difference=max_multiplicity_difference)


__all__ = ["FitConfig", "resolve_fit_config"]


@dataclass(frozen=True, slots=True)
class WorkflowConfig:
    """Current estimation policy; numerical verification remains mandatory."""

    fit: FitConfig = field(default_factory=lambda: FitConfig(
        device="cpu", max_clusters=DEFAULT_STAGE_ONE_MAX_CLUSTERS))
    stage_two_max_clusters: int = DEFAULT_MAX_CLUSTERS
    normal_cn_policy: str | None = None
    missing_counts: str = DEFAULT_MISSING_COUNTS
    stage_three_assignment_method: str = "marginal"

    def __post_init__(self):
        validate_assignment_method(self.stage_three_assignment_method)
        validate_missing_counts(self.missing_counts)
        if type(self.fit) is not FitConfig:
            raise TypeError("Workflow requires a validated FitConfig")
        if self.normal_cn_policy not in (None, "diploid"):
            raise ValueError("normal_cn_policy must be None or the explicit 'diploid' assumption")
        if type(self.stage_two_max_clusters) is not int or self.stage_two_max_clusters < 1:
            raise ValueError("stage_two_max_clusters must be a positive integer")

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        fit_values = dict(device="cpu", max_clusters=DEFAULT_STAGE_ONE_MAX_CLUSTERS)
        fit_values.update(values.get("fit", {}))
        values["fit"] = FitConfig(**fit_values)
        return cls(**values)


__all__ += ["WorkflowConfig"]
