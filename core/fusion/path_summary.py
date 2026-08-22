from __future__ import annotations

import numpy as np

from ...io.data import PathLikelihoodSpec, TumorData
from ..objective import compile_observed_model, observed_terms_numpy


DEFAULT_PATH_BOUNDARY_TOL = 1e-8
DEFAULT_AMPLIFIED_MUTANT_COPY_TOL = 1e-8


def path_posterior_at_phi_numpy(
    data: TumorData,
    phi: np.ndarray,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """Evaluate the fixed path posterior at an arbitrary CCF array.

    This is a reporting/evaluation helper. It uses the same observed likelihood
    kernel and fixed path prior as fitting, but does not alter a fitted objective.
    """

    if data.path_likelihood is None:
        raise ValueError("TumorData does not contain a path likelihood.")
    model = compile_observed_model(data, major_prior=0.5, eps=float(eps))
    terms = observed_terms_numpy(
        model,
        np.asarray(phi, dtype=np.float64),
        eps=float(eps),
    )
    return terms.posterior


def summarize_path_posterior_numpy(
    spec: PathLikelihoodSpec,
    *,
    phi: np.ndarray,
    posterior: np.ndarray,
    supported: np.ndarray | None = None,
    boundary_tol: float = DEFAULT_PATH_BOUNDARY_TOL,
    amplification_tol: float = DEFAULT_AMPLIFIED_MUTANT_COPY_TOL,
) -> dict[str, np.ndarray]:
    """Summarize path posterior mass at the CCF where it was evaluated.

    ``phi`` has shape ``(mutation, region)`` and ``posterior`` has the path
    specification's ``(mutation, region, path)`` shape. Mutant-copy mass is in
    unscaled copy units; effective multiplicity divides that mass by CCF.
    """

    observation_shape = spec.shape[:2]
    phi_array = np.asarray(phi, dtype=np.float64)
    if tuple(phi_array.shape) != tuple(observation_shape):
        raise ValueError(
            f"phi must have shape {observation_shape}, not {phi_array.shape}."
        )
    if not np.all(np.isfinite(phi_array)):
        raise ValueError("phi must contain only finite values.")

    posterior_array = np.asarray(posterior, dtype=np.float64)
    if tuple(posterior_array.shape) != tuple(spec.shape):
        raise ValueError(
            "posterior must have the PathLikelihoodSpec shape "
            f"{spec.shape}, not {posterior_array.shape}."
        )
    valid = np.asarray(spec.valid, dtype=bool)
    valid_posterior = posterior_array[valid]
    if np.any(~np.isfinite(valid_posterior)) or np.any(valid_posterior < 0.0):
        raise ValueError(
            "posterior must be finite and nonnegative on every valid path."
        )
    normalized = np.where(valid, posterior_array, 0.0)
    normalizer = np.sum(normalized, axis=-1, keepdims=True)
    if np.any(~np.isfinite(normalizer)) or np.any(normalizer <= 0.0):
        raise ValueError(
            "posterior must have positive finite mass in every mutation-region."
        )
    normalized = normalized / normalizer

    if supported is None:
        supported_array = np.ones(observation_shape, dtype=bool)
    else:
        supported_array = np.asarray(supported, dtype=bool)
        if tuple(supported_array.shape) != tuple(observation_shape):
            raise ValueError(
                "supported must have shape "
                f"{observation_shape}, not {supported_array.shape}."
            )

    boundary_tolerance = float(boundary_tol)
    amplification_tolerance = float(amplification_tol)
    if not np.isfinite(boundary_tolerance) or boundary_tolerance < 0.0:
        raise ValueError("boundary_tol must be finite and nonnegative.")
    if not np.isfinite(amplification_tolerance) or amplification_tolerance < 0.0:
        raise ValueError("amplification_tol must be finite and nonnegative.")

    expanded_phi = phi_array[..., None]
    switch = np.asarray(spec.switch_fraction, dtype=np.float64)
    first_copy = np.asarray(spec.first_copy, dtype=np.float64)
    second_copy = np.asarray(spec.second_copy, dtype=np.float64)
    mass = first_copy * np.minimum(expanded_phi, switch)
    mass += second_copy * np.maximum(expanded_phi - switch, 0.0)

    single = expanded_phi < switch - boundary_tolerance
    multi = expanded_phi > switch + boundary_tolerance
    boundary = ~(single | multi)
    map_index = np.argmax(np.where(valid, normalized, -np.inf), axis=-1)
    safe_map_index = np.maximum(map_index, 0)
    map_mass = np.take_along_axis(
        mass,
        safe_map_index[..., None],
        axis=-1,
    )[..., 0]
    expected_mass = np.sum(normalized * mass, axis=-1)
    effective_multiplicity = np.divide(
        expected_mass,
        phi_array,
        out=np.full_like(expected_mass, np.nan),
        where=phi_array > 0.0,
    )
    map_effective_multiplicity = np.divide(
        map_mass,
        phi_array,
        out=np.full_like(map_mass, np.nan),
        where=phi_array > 0.0,
    )
    amplified_path = mass > expanded_phi + amplification_tolerance
    amplified_probability = np.sum(normalized * amplified_path, axis=-1)
    entropy = -np.sum(
        np.where(
            normalized > 0.0,
            normalized * np.log(np.clip(normalized, np.finfo(float).tiny, None)),
            0.0,
        ),
        axis=-1,
    )

    def supported_or_nan(values: np.ndarray) -> np.ndarray:
        return np.where(supported_array, values, np.nan)

    return {
        "posterior": np.where(
            supported_array[..., None],
            normalized,
            np.nan,
        ),
        "mass": np.where(supported_array[..., None], mass, np.nan),
        "map_index": np.where(supported_array, map_index, -1),
        "supported": supported_array,
        "single_probability": supported_or_nan(np.sum(normalized * single, axis=-1)),
        "multi_probability": supported_or_nan(np.sum(normalized * multi, axis=-1)),
        "boundary_probability": supported_or_nan(
            np.sum(normalized * boundary, axis=-1)
        ),
        "posterior_mutant_copy_mass": supported_or_nan(expected_mass),
        "posterior_effective_multiplicity": supported_or_nan(effective_multiplicity),
        "map_mutant_copy_mass": supported_or_nan(map_mass),
        "map_effective_multiplicity": supported_or_nan(map_effective_multiplicity),
        "amplified_mutant_copy_probability": supported_or_nan(amplified_probability),
        "amplified_mutant_copy_call": np.where(
            supported_array,
            (amplified_probability >= 0.5).astype(np.float64),
            np.nan,
        ),
        "path_entropy": supported_or_nan(entropy),
    }


__all__ = [
    "DEFAULT_AMPLIFIED_MUTANT_COPY_TOL",
    "DEFAULT_PATH_BOUNDARY_TOL",
    "path_posterior_at_phi_numpy",
    "summarize_path_posterior_numpy",
]
