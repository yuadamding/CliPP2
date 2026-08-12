"""Explicit computational contracts for single-tumor model selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, TypeAlias, cast


ProfileName: TypeAlias = Literal["strict", "balanced", "fast"]


@dataclass(frozen=True, slots=True)
class ComputationProfile:
    """Resource policy and statistical contract for one tumor fit.

    ``strict`` is the reference implementation. ``balanced`` and ``fast``
    deliberately trade stronger refit certification for bounded single-tumor
    latency; their output provenance remains explicit.
    """

    name: ProfileName
    raw_dtype: Literal["float32", "float64"]
    scalar_mode: Literal["interval_certified", "grid_local"]
    scalar_grid_points: int
    scalar_local_steps: int
    lambda_budget: int
    lambda_refinement_budget: int
    outer_max_iter: int
    inner_max_iter: int
    solver_tolerance: float
    solver_retry_limit: int

    @property
    def is_strict(self) -> bool:
        return self.name == "strict"

    @property
    def objective_equivalent_to_strict(self) -> bool:
        # The strict graph weights are defined by the Ward/CEM guide.  The
        # approximate profiles retain complete topology but use the likelihood
        # pilot for weights, so they are not the identical fixed objective.
        return self.is_strict

    @property
    def refit_globally_certified(self) -> bool:
        return self.scalar_mode == "interval_certified"


STRICT_PROFILE: Final = ComputationProfile(
    name="strict",
    raw_dtype="float64",
    scalar_mode="interval_certified",
    scalar_grid_points=17,
    scalar_local_steps=0,
    lambda_budget=12,
    lambda_refinement_budget=12,
    outer_max_iter=8,
    inner_max_iter=30,
    solver_tolerance=5e-5,
    solver_retry_limit=4,
)

# For a single tumor, the complete graph is retained in balanced mode to avoid
# silently changing the estimator.  The dominant latency reductions come from
# approximate scalar refits and a shorter lambda path.
BALANCED_PROFILE: Final = ComputationProfile(
    name="balanced",
    raw_dtype="float32",
    scalar_mode="grid_local",
    scalar_grid_points=64,
    scalar_local_steps=3,
    lambda_budget=8,
    lambda_refinement_budget=2,
    outer_max_iter=6,
    inner_max_iter=25,
    solver_tolerance=2e-4,
    solver_retry_limit=1,
)

FAST_PROFILE: Final = ComputationProfile(
    name="fast",
    raw_dtype="float32",
    scalar_mode="grid_local",
    scalar_grid_points=32,
    scalar_local_steps=1,
    lambda_budget=6,
    lambda_refinement_budget=0,
    outer_max_iter=4,
    inner_max_iter=16,
    solver_tolerance=1e-3,
    solver_retry_limit=0,
)

COMPUTATION_PROFILES: Final = {
    profile.name: profile
    for profile in (STRICT_PROFILE, BALANCED_PROFILE, FAST_PROFILE)
}
COMPUTATION_PROFILE_NAMES: Final = tuple(COMPUTATION_PROFILES)
DEFAULT_COMPUTATION_PROFILE: Final[ProfileName] = "balanced"


def get_computation_profile(value: str) -> ComputationProfile:
    normalized = str(value).strip().lower().replace("-", "_")
    try:
        return COMPUTATION_PROFILES[cast(ProfileName, normalized)]
    except KeyError as error:
        allowed = ", ".join(COMPUTATION_PROFILE_NAMES)
        raise ValueError(
            f"computation_profile must be one of: {allowed}."
        ) from error


__all__ = [
    "BALANCED_PROFILE",
    "COMPUTATION_PROFILE_NAMES",
    "ComputationProfile",
    "DEFAULT_COMPUTATION_PROFILE",
    "FAST_PROFILE",
    "STRICT_PROFILE",
    "get_computation_profile",
]
