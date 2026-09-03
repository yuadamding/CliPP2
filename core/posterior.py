"""Posterior summaries derived from the canonical observed-emission model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .objective import ObservedModel, ObservedTerms, observed_terms_numpy


DEFAULT_PATH_BOUNDARY_TOL = 1e-8
DEFAULT_AMPLIFIED_MUTANT_COPY_TOL = 1e-8


def _readonly(values: np.ndarray, *, dtype: np.dtype | None = None) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class PosteriorSummary:
    """One typed posterior view for every compiled copy-number scenario."""

    path_probability: np.ndarray
    path_mutant_copy_mass: np.ndarray
    map_path: np.ndarray
    reportable: np.ndarray
    pre_switch_probability: np.ndarray
    post_switch_probability: np.ndarray
    switch_boundary_probability: np.ndarray
    expected_mutant_copy_mass: np.ndarray
    expected_multiplicity: np.ndarray
    map_mutant_copy_mass: np.ndarray
    map_multiplicity: np.ndarray
    amplified_mutant_copy_probability: np.ndarray
    amplified_mutant_copy_call: np.ndarray
    entropy: np.ndarray
    major_probability: np.ndarray | None = None
    major_call: np.ndarray | None = None
    multiplicity_estimated: np.ndarray | None = None


def summarize_posterior_numpy(
    model: ObservedModel,
    phi: np.ndarray,
    *,
    eps: float,
    terms: ObservedTerms | None = None,
    reportable: np.ndarray | None = None,
    boundary_tol: float = DEFAULT_PATH_BOUNDARY_TOL,
    amplification_tol: float = DEFAULT_AMPLIFIED_MUTANT_COPY_TOL,
) -> PosteriorSummary:
    """Evaluate and summarize path posterior mass at a fitted CCF profile.

    ``ObservedModel`` is the sole emission authority.  Its numerical kernels
    use scaled copy mass, while its reporting metadata retains the exact
    unscaled paths needed for multiplicity and occupancy summaries.
    """

    phi_array = np.asarray(phi, dtype=np.float64)
    if phi_array.shape != model.shape or not np.all(np.isfinite(phi_array)):
        raise ValueError(f"phi must be a finite array with shape {model.shape}.")
    if reportable is None:
        reportable_array = np.ones(model.shape, dtype=bool)
    else:
        reportable_array = np.asarray(reportable, dtype=bool)
        if reportable_array.shape != model.shape:
            raise ValueError(f"reportable must have shape {model.shape}.")

    boundary_tolerance = float(boundary_tol)
    amplification_tolerance = float(amplification_tol)
    if not np.isfinite(boundary_tolerance) or boundary_tolerance < 0.0:
        raise ValueError("boundary_tol must be finite and nonnegative.")
    if not np.isfinite(amplification_tolerance) or amplification_tolerance < 0.0:
        raise ValueError("amplification_tol must be finite and nonnegative.")

    evaluated = (
        observed_terms_numpy(model, phi_array, eps=float(eps))
        if terms is None
        else terms
    )
    probability = np.where(
        model.valid,
        np.asarray(evaluated.posterior, dtype=np.float64),
        0.0,
    )
    # Preserve the historical reporting helper's defensive renormalization,
    # including its exact floating-point output values.
    probability = probability / np.sum(probability, axis=-1, keepdims=True)
    valid = np.asarray(model.valid, dtype=bool)
    major_probability_values = (
        None
        if model.major_indicator is None
        else np.sum(probability * model.major_indicator, axis=-1)
    )
    expanded_phi = phi_array[..., None]
    switch = np.asarray(model.switch, dtype=np.float64)
    first_copy = np.asarray(model.first_copy, dtype=np.float64)
    second_copy = np.asarray(model.second_copy, dtype=np.float64)
    mass = first_copy * np.minimum(
        expanded_phi,
        switch,
    )
    mass += second_copy * np.maximum(
        expanded_phi - switch,
        0.0,
    )

    pre_switch = expanded_phi < switch - boundary_tolerance
    post_switch = expanded_phi > switch + boundary_tolerance
    at_boundary = ~(pre_switch | post_switch)
    map_path = np.argmax(np.where(valid, probability, -np.inf), axis=-1)
    map_mass = np.take_along_axis(mass, map_path[..., None], axis=-1)[..., 0]
    expected_mass = np.sum(probability * mass, axis=-1)
    expected_multiplicity = np.divide(
        expected_mass,
        phi_array,
        out=np.full_like(expected_mass, np.nan),
        where=phi_array > 0.0,
    )
    map_multiplicity = np.divide(
        map_mass,
        phi_array,
        out=np.full_like(map_mass, np.nan),
        where=phi_array > 0.0,
    )
    if (
        model.uses_binary_linear_mixture_fast_path
        and model.major_indicator is not None
    ):
        # The historical linear mixture reports its exact discrete copy state,
        # including its established major-state tie break at probability 0.5.
        # That differs from NumPy's first-index MAP tie break, so derive the
        # compatibility value from the same posterior-major call explicitly.
        assert major_probability_values is not None
        major_probability_for_call = major_probability_values
        major_call_for_call = major_probability_for_call >= 0.5
        estimated = np.sum(valid, axis=-1) > 1
        low_copy = first_copy[..., 0]
        major_copy = first_copy[..., 1]
        map_multiplicity = np.where(
            estimated,
            np.where(major_call_for_call, major_copy, low_copy),
            low_copy,
        )
    amplified_path = mass > expanded_phi + amplification_tolerance
    amplified_probability = np.sum(probability * amplified_path, axis=-1)
    entropy = -np.sum(
        np.where(
            probability > 0.0,
            probability * np.log(np.clip(probability, np.finfo(np.float64).tiny, None)),
            0.0,
        ),
        axis=-1,
    )

    def masked(values: np.ndarray) -> np.ndarray:
        return np.where(reportable_array, values, np.nan)

    major_probability: np.ndarray | None = None
    major_call: np.ndarray | None = None
    multiplicity_estimated: np.ndarray | None = None
    if major_probability_values is not None:
        major_probability = masked(major_probability_values)
        major_call = np.asarray(major_probability_values >= 0.5, dtype=bool)
        multiplicity_estimated = np.sum(valid, axis=-1) > 1

    return PosteriorSummary(
        path_probability=_readonly(
            np.where(reportable_array[..., None], probability, np.nan)
        ),
        path_mutant_copy_mass=_readonly(
            np.where(reportable_array[..., None], mass, np.nan)
        ),
        map_path=_readonly(np.where(reportable_array, map_path, -1), dtype=np.int64),
        reportable=_readonly(reportable_array, dtype=bool),
        pre_switch_probability=_readonly(
            masked(np.sum(probability * pre_switch, axis=-1))
        ),
        post_switch_probability=_readonly(
            masked(np.sum(probability * post_switch, axis=-1))
        ),
        switch_boundary_probability=_readonly(
            masked(np.sum(probability * at_boundary, axis=-1))
        ),
        expected_mutant_copy_mass=_readonly(masked(expected_mass)),
        expected_multiplicity=_readonly(masked(expected_multiplicity)),
        map_mutant_copy_mass=_readonly(masked(map_mass)),
        map_multiplicity=_readonly(masked(map_multiplicity)),
        amplified_mutant_copy_probability=_readonly(masked(amplified_probability)),
        amplified_mutant_copy_call=_readonly(
            np.where(
                reportable_array,
                (amplified_probability >= 0.5).astype(np.float64),
                np.nan,
            )
        ),
        entropy=_readonly(masked(entropy)),
        major_probability=(
            None if major_probability is None else _readonly(major_probability)
        ),
        major_call=None if major_call is None else _readonly(major_call, dtype=bool),
        multiplicity_estimated=(
            None
            if multiplicity_estimated is None
            else _readonly(multiplicity_estimated, dtype=bool)
        ),
    )


__all__ = [
    "DEFAULT_AMPLIFIED_MUTANT_COPY_TOL",
    "DEFAULT_PATH_BOUNDARY_TOL",
    "PosteriorSummary",
    "summarize_posterior_numpy",
]
