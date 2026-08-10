"""Shared compilation utilities for CliPP2 single-switch path likelihoods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .data import PathLikelihoodSpec, TumorData


PATH_LIKELIHOOD_MODEL_ID = "clipp2_single_switch_path_mixture_v1"
PATH_LIKELIHOOD_MODEL_VERSION = "1"
PATH_CANDIDATE_GENERATOR_VERSION = "phased_unphased_two_state_positive_dosages_v1"


@dataclass(frozen=True, slots=True)
class LocalCopyNumberState:
    """One phased or unphased allele-specific tumor copy-number state."""

    fraction: float
    allele_a_cn: int
    allele_b_cn: int


@dataclass(frozen=True, slots=True)
class CompiledPathSet:
    """Canonical numeric paths with normalized alias-weighted priors."""

    paths: tuple[tuple[float, float, float], ...]
    log_prior: tuple[float, ...]
    biological_duplicate_count: tuple[int, ...]


def path_prior_mode(dosage_prior_penalty: float) -> str:
    """Return the versioned prior identifier for a dosage penalty."""

    return (
        "endpoint_excess_dosage_penalty_biological_alias_mass_v1:"
        f"alpha={float(dosage_prior_penalty):.12g}"
    )


def dominant_copy_number_state(
    states: Sequence[LocalCopyNumberState],
) -> LocalCopyNumberState:
    """Select one deterministic display state without affecting likelihood."""

    if not states:
        raise ValueError("At least one copy-number state is required.")
    return max(
        states,
        key=lambda state: (
            state.fraction,
            state.allele_a_cn + state.allele_b_cn,
            state.allele_a_cn,
            state.allele_b_cn,
        ),
    )


def _canonical_path(
    first_copy: int,
    second_copy: int,
    switch_fraction: float,
) -> tuple[float, float, float]:
    first = float(first_copy)
    second = float(second_copy)
    switch = 1.0 if first == second else float(switch_fraction)
    return first, second, switch


def compile_single_switch_paths(
    states: Sequence[LocalCopyNumberState],
    *,
    allele_mode: str,
    dosage_prior_penalty: float,
) -> CompiledPathSet:
    """Compile one/two local states and retain biological alias prior mass.

    ``phased`` connects A-to-A and B-to-B. ``unphased`` additionally includes
    the swapped homolog assignment. Numerically identical paths are stored
    once, after summing the prior mass of every biological mapping that
    generated them.
    """

    mode = str(allele_mode)
    if mode not in {"phased", "unphased"}:
        raise ValueError("allele_mode must be 'phased' or 'unphased'.")
    penalty = float(dosage_prior_penalty)
    if not np.isfinite(penalty) or penalty < 0.0:
        raise ValueError("dosage_prior_penalty must be finite and nonnegative.")
    normalized_states = tuple(states)
    if len(normalized_states) not in {1, 2}:
        return CompiledPathSet((), (), ())
    for state in normalized_states:
        if (
            not np.isfinite(state.fraction)
            or state.fraction <= 0.0
            or int(state.allele_a_cn) != state.allele_a_cn
            or int(state.allele_b_cn) != state.allele_b_cn
            or state.allele_a_cn < 0
            or state.allele_b_cn < 0
        ):
            raise ValueError(
                "Copy-number states require positive finite fractions and "
                "nonnegative integer allele copy numbers."
            )
        if mode == "unphased" and state.allele_a_cn < state.allele_b_cn:
            raise ValueError("Unphased states require allele_a_cn >= allele_b_cn.")
    if not np.isclose(
        sum(state.fraction for state in normalized_states),
        1.0,
        rtol=0.0,
        atol=1e-8,
    ):
        raise ValueError("Local copy-number state fractions must sum to one.")

    candidates: list[tuple[float, float, float]] = []
    if len(normalized_states) == 1:
        state = normalized_states[0]
        for copy_number in (state.allele_a_cn, state.allele_b_cn):
            for dosage in range(1, copy_number + 1):
                candidates.append(_canonical_path(dosage, dosage, 1.0))
    else:
        state1, state2 = normalized_states
        matchings = [
            (
                (state1.allele_a_cn, state2.allele_a_cn),
                (state1.allele_b_cn, state2.allele_b_cn),
            )
        ]
        if mode == "unphased":
            matchings.append(
                (
                    (state1.allele_a_cn, state2.allele_b_cn),
                    (state1.allele_b_cn, state2.allele_a_cn),
                )
            )
        for matching in matchings:
            for first_state_copies, second_state_copies in matching:
                for dosage1 in range(1, first_state_copies + 1):
                    for dosage2 in range(1, second_state_copies + 1):
                        candidates.append(
                            _canonical_path(
                                dosage1,
                                dosage2,
                                state1.fraction,
                            )
                        )
                        candidates.append(
                            _canonical_path(
                                dosage2,
                                dosage1,
                                state2.fraction,
                            )
                        )

    log_prior_mass: dict[tuple[float, float, float], float] = {}
    duplicate_count: dict[tuple[float, float, float], int] = {}
    for path in candidates:
        first, second, switch = path
        endpoint_mass = first * switch + second * (1.0 - switch)
        log_weight = -penalty * max(endpoint_mass - 1.0, 0.0)
        if path in log_prior_mass:
            log_prior_mass[path] = float(
                np.logaddexp(log_prior_mass[path], log_weight)
            )
        else:
            log_prior_mass[path] = float(log_weight)
        duplicate_count[path] = duplicate_count.get(path, 0) + 1
    if not log_prior_mass:
        return CompiledPathSet((), (), ())
    paths = tuple(
        sorted(log_prior_mass, key=lambda item: (item[2], item[0], item[1]))
    )
    log_prior = np.asarray([log_prior_mass[path] for path in paths], dtype=np.float64)
    log_prior -= np.logaddexp.reduce(log_prior)
    return CompiledPathSet(
        paths=paths,
        log_prior=tuple(float(value) for value in log_prior),
        biological_duplicate_count=tuple(duplicate_count[path] for path in paths),
    )


def build_path_likelihood(
    compiled_units: Sequence[Sequence[CompiledPathSet]],
    *,
    model_id: str,
    model_version: str,
    candidate_generator_version: str = PATH_CANDIDATE_GENERATOR_VERSION,
    prior_mode: str,
) -> tuple[PathLikelihoodSpec, np.ndarray]:
    """Pad compiled unit paths into one immutable likelihood specification."""

    num_mutations = len(compiled_units)
    if num_mutations == 0:
        raise ValueError("compiled_units must contain at least one mutation.")
    num_samples = len(compiled_units[0])
    if num_samples == 0 or any(len(row) != num_samples for row in compiled_units):
        raise ValueError("compiled_units must be a nonempty rectangular matrix.")
    max_paths = max(len(compiled.paths) for row in compiled_units for compiled in row)
    if max_paths == 0:
        raise ValueError("Every compiled unit must contain at least one path.")
    shape = (num_mutations, num_samples, max_paths)
    first_copy = np.zeros(shape, dtype=np.float64)
    second_copy = np.zeros(shape, dtype=np.float64)
    switch_fraction = np.zeros(shape, dtype=np.float64)
    log_prior = np.full(shape, -np.inf, dtype=np.float64)
    valid = np.zeros(shape, dtype=bool)
    biological_duplicates = np.zeros(shape, dtype=np.int64)
    for mutation_index, row in enumerate(compiled_units):
        for sample_index, compiled in enumerate(row):
            if not compiled.paths:
                raise ValueError("Every compiled unit must contain at least one path.")
            if not (
                len(compiled.paths)
                == len(compiled.log_prior)
                == len(compiled.biological_duplicate_count)
            ):
                raise ValueError("Compiled path arrays have inconsistent lengths.")
            for path_index, path in enumerate(compiled.paths):
                first_copy[mutation_index, sample_index, path_index] = path[0]
                second_copy[mutation_index, sample_index, path_index] = path[1]
                switch_fraction[mutation_index, sample_index, path_index] = path[2]
                log_prior[mutation_index, sample_index, path_index] = (
                    compiled.log_prior[path_index]
                )
                biological_duplicates[mutation_index, sample_index, path_index] = (
                    compiled.biological_duplicate_count[path_index]
                )
                valid[mutation_index, sample_index, path_index] = True
    return (
        PathLikelihoodSpec(
            model_id=model_id,
            model_version=model_version,
            candidate_generator_version=candidate_generator_version,
            prior_mode=prior_mode,
            first_copy=first_copy,
            second_copy=second_copy,
            switch_fraction=switch_fraction,
            log_prior=log_prior,
            valid=valid,
        ),
        biological_duplicates,
    )


def initialize_path_marginal_phi(
    data: TumorData,
    *,
    eps: float,
) -> np.ndarray:
    """Return the canonical exact scalar minimizer of each path-marginal unit."""

    if data.path_likelihood is None:
        raise ValueError("TumorData must contain a path likelihood.")
    epsilon = float(eps)
    if not np.isfinite(epsilon) or not 0.0 < epsilon < 0.5:
        raise ValueError("eps must be finite and lie strictly in (0, 0.5).")
    data.phi_init = np.full_like(data.alt_counts, 0.5, dtype=np.float64)

    # Delayed import avoids an io/core import cycle during package startup.
    from ..core.fusion.starts import compute_scalar_mutation_region_wells

    primary, _secondary, _valid_secondary = compute_scalar_mutation_region_wells(
        data,
        major_prior=0.5,
        eps=epsilon,
        tol=1e-10,
        max_iter=256,
    )
    return np.clip(np.asarray(primary, dtype=np.float64), epsilon, 1.0)


__all__ = [
    "CompiledPathSet",
    "LocalCopyNumberState",
    "PATH_CANDIDATE_GENERATOR_VERSION",
    "PATH_LIKELIHOOD_MODEL_ID",
    "PATH_LIKELIHOOD_MODEL_VERSION",
    "build_path_likelihood",
    "compile_single_switch_paths",
    "dominant_copy_number_state",
    "initialize_path_marginal_phi",
    "path_prior_mode",
]
