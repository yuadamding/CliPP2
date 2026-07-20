from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...io.data import TumorData


@dataclass(frozen=True)
class MultiplicityPosterior:
    """Posterior multiplicity state at a fitted mutation-region CCF matrix."""

    gamma_major: np.ndarray
    major_call: np.ndarray
    multiplicity_call: np.ndarray
    estimation_mask: np.ndarray


def infer_multiplicity_posterior_numpy(
    data: TumorData,
    phi: np.ndarray,
    *,
    major_prior: float,
    eps: float,
) -> MultiplicityPosterior:
    """Infer the major/minor-copy posterior under the observed-count model.

    The calculation is the NumPy counterpart of
    ``mutation_region_terms_torch``. Entries outside the model's multiplicity
    estimation mask remain fixed to the available major-copy multiplicity.
    Entries whose counts are marked unobserved retain the configured prior.
    """

    phi_array = np.asarray(phi, dtype=np.float64)
    expected_shape = np.asarray(data.alt_counts).shape
    if phi_array.shape != expected_shape:
        raise ValueError(
            "phi must have the same mutation-region shape as the tumor data; "
            f"got {phi_array.shape}, expected {expected_shape}."
        )
    if not np.all(np.isfinite(phi_array)):
        raise ValueError("phi must contain only finite values.")

    prior = float(major_prior)
    if not np.isfinite(prior) or not (0.0 < prior < 1.0):
        raise ValueError("major_prior must lie strictly in (0, 1).")
    probability_eps = float(eps)
    if not np.isfinite(probability_eps) or not (0.0 < probability_eps < 0.5):
        raise ValueError("eps must lie strictly in (0, 0.5).")

    alt = np.asarray(data.alt_counts, dtype=np.float64)
    total = np.asarray(data.total_counts, dtype=np.float64)
    nonalt = total - alt
    scaling = np.asarray(data.scaling, dtype=np.float64)

    prob_minor = np.clip(
        scaling * np.asarray(data.minor_cn, dtype=np.float64) * phi_array,
        probability_eps,
        1.0 - probability_eps,
    )
    prob_major = np.clip(
        scaling * np.asarray(data.major_cn, dtype=np.float64) * phi_array,
        probability_eps,
        1.0 - probability_eps,
    )
    log_minor = (
        alt * np.log(prob_minor) + nonalt * np.log1p(-prob_minor) + np.log1p(-prior)
    )
    log_major = (
        alt * np.log(prob_major) + nonalt * np.log1p(-prob_major) + np.log(prior)
    )

    # exp(-logaddexp(0, -delta)) is a stable sigmoid(delta).
    posterior_major = np.exp(-np.logaddexp(0.0, log_minor - log_major))
    estimation_mask = np.asarray(data.multiplicity_estimation_mask, dtype=bool)
    count_observed = getattr(data, "count_observed", None)
    observed_mask = (
        np.ones_like(estimation_mask, dtype=bool)
        if count_observed is None
        else np.asarray(count_observed, dtype=bool)
    )
    if observed_mask.shape != estimation_mask.shape:
        raise ValueError(
            "count_observed must have the same mutation-region shape as the tumor data."
        )

    major_probability = np.ones_like(phi_array, dtype=np.float64)
    informed_mask = estimation_mask & observed_mask
    major_probability[estimation_mask] = prior
    major_probability[informed_mask] = posterior_major[informed_mask]

    major_call = major_probability >= 0.5
    multiplicity_call = np.where(
        estimation_mask,
        np.where(major_call, data.major_cn, data.minor_cn),
        data.fixed_multiplicity,
    ).astype(np.float64, copy=False)
    return MultiplicityPosterior(
        gamma_major=major_probability,
        major_call=major_call.astype(bool, copy=False),
        multiplicity_call=multiplicity_call,
        estimation_mask=estimation_mask,
    )


__all__ = ["MultiplicityPosterior", "infer_multiplicity_posterior_numpy"]
