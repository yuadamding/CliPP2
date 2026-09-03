"""Immutable, input-only tumor representation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence

import numpy as np


class ExclusionCode(IntEnum):
    """Compact provenance for a mutation-region row excluded from inference."""

    INCLUDED = 0
    COUNT_UNAVAILABLE = 1
    LIKELIHOOD_UNSUPPORTED = 2
    MORE_THAN_TWO_LOCAL_CN_STATES = 3
    NO_POSITIVE_MUTANT_COPY_PATH = 4
    LIKELIHOOD_EXCLUDED_BY_POLICY = 5


_UNSUPPORTED_CODES = frozenset(
    {
        ExclusionCode.LIKELIHOOD_UNSUPPORTED,
        ExclusionCode.MORE_THAN_TWO_LOCAL_CN_STATES,
        ExclusionCode.NO_POSITIVE_MUTANT_COPY_PATH,
    }
)


def _readonly(value: object, *, dtype: np.dtype) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True, order="C")
    array.setflags(write=False)
    return array


def _arithmetic_epsilon(*values: object) -> float:
    epsilon = float(np.finfo(np.float64).eps)
    for value in values:
        dtype = np.asarray(value).dtype
        if np.issubdtype(dtype, np.floating):
            epsilon = max(epsilon, float(np.finfo(dtype).eps))
    return epsilon


def _mean_tumor_cn_for_scale(
    purity: np.ndarray,
    normal_cn: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """Recover a mean tumor CN whose float64 compilation retains ``scale``."""

    total = (purity / scale - (1.0 - purity) * normal_cn) / purity
    if np.any(~np.isfinite(total)) or np.any(total < 0.0):
        raise ValueError("Historical scaling is incompatible with purity and normal_cn.")
    # The algebraic inverse can round one ULP away when the forward expression
    # is evaluated again. Choose the adjacent representable mean CN that keeps
    # the v0.3 likelihood scale exact; this does not change any supplied fact.
    for _ in range(16):
        denominator = (1.0 - purity) * normal_cn + purity * total
        recovered = purity / denominator
        unresolved = recovered != scale
        if not np.any(unresolved):
            break
        direction = np.where(recovered > scale, np.inf, -np.inf)
        candidate = np.nextafter(total, direction)
        total = np.where(unresolved & (candidate >= 0.0), candidate, total)
    return total


@dataclass(frozen=True, slots=True)
class EmissionPaths:
    """One normalized piecewise-affine mutant-copy input representation.

    Arrays have shape ``(mutation, region, path)``. Before sample-specific
    probability scaling, a path's mutant-copy mass is
    ``first_copy * min(phi, switch_fraction)`` plus
    ``second_copy * max(phi - switch_fraction, 0)``.
    """

    model_id: str
    first_copy: np.ndarray
    second_copy: np.ndarray
    switch_fraction: np.ndarray
    log_prior: np.ndarray
    valid: np.ndarray
    model_version: str = "1"
    candidate_generator_version: str = "unspecified"
    prior_mode: str = "fixed_canonical_path_prior"
    major_indicator: np.ndarray | None = None
    major_prior_weighted: bool = False
    constrain_probability_box: bool = False

    def __post_init__(self) -> None:
        identifiers = {
            "model_id": str(self.model_id).strip(),
            "model_version": str(self.model_version).strip(),
            "candidate_generator_version": str(
                self.candidate_generator_version
            ).strip(),
            "prior_mode": str(self.prior_mode).strip(),
        }
        for name, value in identifiers.items():
            if not value:
                raise ValueError(f"EmissionPaths.{name} must be nonempty.")

        numeric = {
            name: np.array(getattr(self, name), dtype=np.float64, copy=True, order="C")
            for name in ("first_copy", "second_copy", "switch_fraction", "log_prior")
        }
        shape = numeric["first_copy"].shape
        if len(shape) != 3 or shape[2] <= 0:
            raise ValueError("EmissionPaths arrays must have shape (M, S, P), P > 0.")
        for name, value in numeric.items():
            if value.shape != shape:
                raise ValueError(f"EmissionPaths.{name} must have shape {shape}.")

        valid = np.array(self.valid, dtype=bool, copy=True, order="C")
        if valid.shape != shape:
            raise ValueError(f"EmissionPaths.valid must have shape {shape}.")
        if not np.all(np.any(valid, axis=-1)):
            raise ValueError("Every mutation-region entry needs a valid emission path.")
        for name in ("first_copy", "second_copy"):
            values = numeric[name][valid]
            if np.any(~np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError(f"Valid {name} values must be finite and nonnegative.")
        switches = numeric["switch_fraction"][valid]
        if np.any(~np.isfinite(switches)) or np.any(
            (switches < 0.0) | (switches > 1.0)
        ):
            raise ValueError("Valid switch fractions must lie in [0, 1].")
        if np.any(~np.isfinite(numeric["log_prior"][valid])):
            raise ValueError("Valid log priors must be finite.")

        for name in ("first_copy", "second_copy", "switch_fraction"):
            numeric[name] = np.where(valid, numeric[name], 0.0)
        numeric["log_prior"] = np.where(valid, numeric["log_prior"], -np.inf)
        maximum = np.max(numeric["log_prior"], axis=-1, keepdims=True)
        normalizer = np.squeeze(
            maximum
            + np.log(
                np.sum(
                    np.where(
                        valid,
                        np.exp(numeric["log_prior"] - maximum),
                        0.0,
                    ),
                    axis=-1,
                    keepdims=True,
                )
            ),
            axis=-1,
        )
        if not np.allclose(normalizer, 0.0, rtol=0.0, atol=1e-10):
            raise ValueError("EmissionPaths.log_prior must normalize over valid paths.")

        indicator = self.major_indicator
        if indicator is not None:
            indicator = np.array(indicator, dtype=bool, copy=True, order="C")
            if indicator.shape != shape:
                raise ValueError(f"EmissionPaths.major_indicator must have shape {shape}.")
            indicator &= valid
            indicator.setflags(write=False)
        if self.major_prior_weighted:
            if indicator is None:
                raise ValueError("major_prior_weighted paths require major_indicator.")
            if np.any(np.sum(indicator, axis=-1) != 1) or np.any(
                np.sum(valid, axis=-1) > 2
            ):
                raise ValueError(
                    "Major-prior-weighted units require one major path and at most "
                    "one alternative."
                )

        for name, value in numeric.items():
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        valid.setflags(write=False)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "major_indicator", indicator)
        for name, value in identifiers.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "major_prior_weighted", bool(self.major_prior_weighted))
        object.__setattr__(
            self, "constrain_probability_box", bool(self.constrain_probability_box)
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.first_copy.shape)

    def validate_observation_shape(self, shape: tuple[int, int]) -> None:
        if tuple(self.first_copy.shape[:2]) != tuple(shape):
            raise ValueError(
                "EmissionPaths mutation-region shape "
                f"{self.first_copy.shape[:2]} does not match observations {shape}."
            )

    def log_prior_for(self, major_prior: float) -> np.ndarray:
        """Return priors after applying an optional major-path parameter."""

        if not self.major_prior_weighted:
            return self.log_prior
        prior = float(major_prior)
        if not np.isfinite(prior) or not 0.0 < prior < 1.0:
            raise ValueError("major_prior must lie strictly in (0, 1).")
        assert self.major_indicator is not None
        alternatives = self.valid & ~self.major_indicator
        values = np.where(
            self.major_indicator,
            np.log(prior),
            np.where(alternatives, np.log1p(-prior), -np.inf),
        )
        return np.where(
            np.sum(self.valid, axis=-1, keepdims=True) == 1, 0.0, values
        ).astype(np.float64, copy=False)


@dataclass(frozen=True, slots=True)
class TumorData:
    """Canonical immutable tumor input, with no solver-derived state."""

    tumor_id: str
    mutation_ids: Sequence[str]
    region_ids: Sequence[str]
    alt_counts: np.ndarray
    total_counts: np.ndarray
    purity: np.ndarray
    major_cn: np.ndarray
    minor_cn: np.ndarray
    normal_cn: np.ndarray
    tumor_total_cn: np.ndarray
    count_available: np.ndarray
    likelihood_supported: np.ndarray
    policy_included: np.ndarray
    emission_paths: EmissionPaths
    exclusion_code: np.ndarray | None = None

    def __post_init__(self) -> None:
        tumor_id = str(self.tumor_id).strip()
        mutations = tuple(str(value) for value in self.mutation_ids)
        regions = tuple(str(value) for value in self.region_ids)
        if not tumor_id or not mutations or not regions:
            raise ValueError("Tumor, mutation, and region identifiers must be nonempty.")
        if any(not value for value in mutations + regions):
            raise ValueError("Mutation and region identifiers must be nonempty strings.")
        if len(set(mutations)) != len(mutations) or len(set(regions)) != len(regions):
            raise ValueError("Mutation and region identifiers must be unique.")

        numeric = {
            name: _readonly(getattr(self, name), dtype=np.float64)
            for name in (
                "alt_counts", "total_counts", "purity", "major_cn", "minor_cn",
                "normal_cn", "tumor_total_cn",
            )
        }
        shape = numeric["alt_counts"].shape
        if shape != (len(mutations), len(regions)):
            raise ValueError(
                "Tumor arrays must have shape "
                f"({len(mutations)}, {len(regions)}), not {shape}."
            )
        for name, value in numeric.items():
            if value.shape != shape or np.any(~np.isfinite(value)):
                raise ValueError(f"TumorData.{name} must be finite with shape {shape}.")
        if np.any(numeric["alt_counts"] < 0.0) or np.any(
            numeric["total_counts"] < numeric["alt_counts"]
        ):
            raise ValueError("Counts must satisfy 0 <= alt_counts <= total_counts.")
        if np.any((numeric["purity"] <= 0.0) | (numeric["purity"] > 1.0)):
            raise ValueError("Purity must lie in (0, 1].")
        for name in ("major_cn", "minor_cn", "normal_cn", "tumor_total_cn"):
            if np.any(numeric[name] < 0.0):
                raise ValueError(f"TumorData.{name} must be nonnegative.")
        if np.any(numeric["minor_cn"] > numeric["major_cn"]):
            raise ValueError("minor_cn cannot exceed major_cn.")

        masks = {
            name: _readonly(getattr(self, name), dtype=bool)
            for name in ("count_available", "likelihood_supported", "policy_included")
        }
        for name, value in masks.items():
            if value.shape != shape:
                raise ValueError(f"TumorData.{name} must have shape {shape}.")
        self.emission_paths.validate_observation_shape(shape)
        included = (
            masks["count_available"]
            & masks["likelihood_supported"]
            & masks["policy_included"]
        )

        if self.exclusion_code is None:
            codes = np.full(shape, int(ExclusionCode.INCLUDED), dtype=np.uint8)
            codes[~masks["policy_included"]] = int(
                ExclusionCode.LIKELIHOOD_EXCLUDED_BY_POLICY
            )
            codes[~masks["count_available"]] = int(ExclusionCode.COUNT_UNAVAILABLE)
            codes[~masks["likelihood_supported"]] = int(
                ExclusionCode.LIKELIHOOD_UNSUPPORTED
            )
        else:
            raw_codes = np.asarray(self.exclusion_code)
            if raw_codes.shape != shape:
                raise ValueError(f"TumorData.exclusion_code must have shape {shape}.")
            try:
                codes = np.array(raw_codes, dtype=np.uint8, copy=True, order="C")
                if np.any(~np.isin(codes, [int(value) for value in ExclusionCode])):
                    raise ValueError
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("TumorData.exclusion_code contains an unknown code.") from exc

        if np.any(codes[included] != int(ExclusionCode.INCLUDED)):
            raise ValueError("Included coordinates must use ExclusionCode.INCLUDED.")
        unsupported = ~masks["likelihood_supported"]
        if np.any(
            ~np.isin(codes[unsupported], [int(code) for code in _UNSUPPORTED_CODES])
        ):
            raise ValueError("Unsupported coordinates require a support exclusion code.")
        unavailable = masks["likelihood_supported"] & ~masks["count_available"]
        if np.any(codes[unavailable] != int(ExclusionCode.COUNT_UNAVAILABLE)):
            raise ValueError("Unavailable coordinates require COUNT_UNAVAILABLE.")
        policy_excluded = (
            masks["likelihood_supported"]
            & masks["count_available"]
            & ~masks["policy_included"]
        )
        if np.any(
            codes[policy_excluded]
            != int(ExclusionCode.LIKELIHOOD_EXCLUDED_BY_POLICY)
        ):
            raise ValueError("Policy-excluded coordinates require their policy code.")

        codes.setflags(write=False)
        object.__setattr__(self, "tumor_id", tumor_id)
        object.__setattr__(self, "mutation_ids", mutations)
        object.__setattr__(self, "region_ids", regions)
        for name, value in numeric.items():
            object.__setattr__(self, name, value)
        for name, value in masks.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "exclusion_code", codes)

    def objective_inclusion_mask(self) -> np.ndarray:
        included = self.count_available & self.likelihood_supported & self.policy_included
        included.setflags(write=False)
        return included

    @property
    def num_mutations(self) -> int:
        return int(self.alt_counts.shape[0])

    @property
    def num_regions(self) -> int:
        return int(self.alt_counts.shape[1])

def _hash_text(digest: object, value: str) -> None:
    encoded = str(value).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)


def _hash_array(digest: object, name: str, values: np.ndarray) -> None:
    _hash_text(digest, name)
    array = np.ascontiguousarray(np.asarray(values))
    _hash_text(digest, str(array.dtype))
    digest.update(len(array.shape).to_bytes(8, "little"))
    for dimension in array.shape:
        digest.update(int(dimension).to_bytes(8, "little", signed=True))
    digest.update(array.tobytes())


def tumor_data_fingerprint(data: TumorData) -> str:
    """Return deterministic identity for the normalized input-only payload."""

    digest = hashlib.sha256()
    _hash_text(digest, "clipp2.tumor-input.v2")
    _hash_text(digest, data.tumor_id)
    for values in (data.mutation_ids, data.region_ids):
        digest.update(len(values).to_bytes(8, "little"))
        for value in values:
            _hash_text(digest, value)
    for name in (
        "alt_counts", "total_counts", "purity", "major_cn", "minor_cn",
        "normal_cn", "tumor_total_cn", "count_available",
        "likelihood_supported", "policy_included", "exclusion_code",
    ):
        _hash_array(digest, name, getattr(data, name))
    paths = data.emission_paths
    for name in (
        "model_id", "model_version", "candidate_generator_version", "prior_mode",
    ):
        _hash_text(digest, str(getattr(paths, name)))
    _hash_text(digest, f"major_prior_weighted:{int(paths.major_prior_weighted)}")
    _hash_text(
        digest, f"constrain_probability_box:{int(paths.constrain_probability_box)}"
    )
    for name in ("first_copy", "second_copy", "switch_fraction", "log_prior", "valid"):
        _hash_array(digest, f"emission_paths.{name}", getattr(paths, name))
    _hash_text(
        digest,
        "major_indicator:" + ("none" if paths.major_indicator is None else "present"),
    )
    if paths.major_indicator is not None:
        _hash_array(digest, "emission_paths.major_indicator", paths.major_indicator)
    return digest.hexdigest()


def exclusion_codes_from_strings(values: object) -> np.ndarray:
    """Translate a historical reason array at the IO migration boundary."""

    mapping = {
        None: ExclusionCode.INCLUDED,
        "COUNT_UNAVAILABLE": ExclusionCode.COUNT_UNAVAILABLE,
        "LIKELIHOOD_UNSUPPORTED": ExclusionCode.LIKELIHOOD_UNSUPPORTED,
        "MORE_THAN_TWO_LOCAL_CN_STATES": ExclusionCode.MORE_THAN_TWO_LOCAL_CN_STATES,
        "NO_POSITIVE_MUTANT_COPY_PATH": ExclusionCode.NO_POSITIVE_MUTANT_COPY_PATH,
        "LIKELIHOOD_EXCLUDED_BY_POLICY": ExclusionCode.LIKELIHOOD_EXCLUDED_BY_POLICY,
    }
    array = np.asarray(values, dtype=object)
    try:
        result = np.asarray([int(mapping[value]) for value in array.flat], dtype=np.uint8)
    except KeyError as exc:
        raise ValueError(f"Unknown historical exclusion reason: {exc.args[0]!r}.") from exc
    return result.reshape(array.shape)


def migrate_v03_tumor_data(
    *,
    tumor_id: str,
    mutation_ids: Sequence[str],
    region_ids: Sequence[str],
    alt_counts: np.ndarray,
    total_counts: np.ndarray,
    purity: np.ndarray,
    major_cn: np.ndarray,
    minor_cn: np.ndarray,
    normal_cn: np.ndarray,
    scaling: np.ndarray,
    count_available: np.ndarray | None = None,
    likelihood_supported: np.ndarray | None = None,
    policy_included: np.ndarray | None = None,
    emission_paths: EmissionPaths | None = None,
    exclusion_code: np.ndarray | None = None,
    # Historical names are intentionally accepted only by this IO adapter.
    count_observed: np.ndarray | None = None,
    path_likelihood: EmissionPaths | None = None,
    path_unsupported_reason: np.ndarray | None = None,
    likelihood_included: np.ndarray | None = None,
    likelihood_exclusion_reason: np.ndarray | None = None,
    has_cna: np.ndarray | None = None,
    phi_upper: np.ndarray | None = None,
    phi_init: np.ndarray | None = None,
    init_major_mask: np.ndarray | None = None,
    eps: float = 1e-6,
) -> TumorData:
    """Normalize a v0.3 programmatic payload into the v0.4 input schema.

    This adapter is intentionally outside the numerical core. Solver-derived
    historical fields are accepted only so callers can migrate atomically;
    they are never retained by :class:`TumorData`.
    """

    del has_cna, phi_init, init_major_mask
    alt = np.asarray(alt_counts, dtype=np.float64)
    shape = alt.shape
    available = count_available if count_available is not None else count_observed
    if available is None:
        available = np.ones(shape, dtype=bool)
    elif count_available is not None and count_observed is not None and not np.array_equal(
        np.asarray(count_available, dtype=bool), np.asarray(count_observed, dtype=bool)
    ):
        raise ValueError("count_observed and count_available must agree.")
    available = np.asarray(available, dtype=bool)
    if available.shape != shape:
        raise ValueError("Historical count availability must match counts.")
    path_reasons = (
        None
        if path_unsupported_reason is None
        else np.asarray(path_unsupported_reason, dtype=object)
    )
    exclusion_reasons = (
        None
        if likelihood_exclusion_reason is None
        else np.asarray(likelihood_exclusion_reason, dtype=object)
    )
    for name, values in (
        ("path_unsupported_reason", path_reasons),
        ("likelihood_exclusion_reason", exclusion_reasons),
    ):
        if values is not None and values.shape != shape:
            raise ValueError(f"Historical {name} must match counts.")
    unsupported_codes = np.asarray(
        [int(value) for value in _UNSUPPORTED_CODES], dtype=np.uint8
    )
    path_reason_codes = (
        None if path_reasons is None else exclusion_codes_from_strings(path_reasons)
    )
    exclusion_reason_codes = (
        None
        if exclusion_reasons is None
        else exclusion_codes_from_strings(exclusion_reasons)
    )
    if path_reason_codes is not None:
        path_present = ~np.equal(path_reasons, None)  # noqa: E711
        if np.any(path_present & ~np.isin(path_reason_codes, unsupported_codes)):
            raise ValueError(
                "path_unsupported_reason may name only likelihood-support failures."
            )
    if path_reasons is not None and exclusion_reasons is not None:
        assert exclusion_reason_codes is not None
        general_support_failure = np.isin(
            exclusion_reason_codes, unsupported_codes
        )
        expected_path_reasons = np.where(
            general_support_failure, exclusion_reasons, None
        )
        if not np.array_equal(path_reasons, expected_path_reasons):
            raise ValueError(
                "path_unsupported_reason and likelihood_exclusion_reason must "
                "agree on every likelihood-support failure."
            )
    inferred_unsupported = np.zeros(shape, dtype=bool)
    if path_reason_codes is not None:
        inferred_unsupported |= np.isin(path_reason_codes, unsupported_codes)
    if exclusion_reason_codes is not None:
        inferred_unsupported |= np.isin(exclusion_reason_codes, unsupported_codes)
    supported = (
        ~inferred_unsupported
        if likelihood_supported is None
        else np.asarray(likelihood_supported, dtype=bool)
    )
    if supported.shape != shape:
        raise ValueError("Historical likelihood_supported must match counts.")
    if path_reasons is not None:
        path_present = ~np.equal(path_reasons, None)  # noqa: E711
        if np.any(path_present & supported):
            raise ValueError(
                "path_unsupported_reason disagrees with likelihood_supported."
            )
    if exclusion_reasons is not None:
        assert exclusion_reason_codes is not None
        reason_present = ~np.equal(exclusion_reasons, None)  # noqa: E711
        reason_supported = ~np.isin(exclusion_reason_codes, unsupported_codes)
        if np.any(reason_present & (supported != reason_supported)):
            raise ValueError(
                "likelihood_exclusion_reason disagrees with likelihood_supported."
            )
    legacy_included = (
        None
        if likelihood_included is None
        else np.asarray(likelihood_included, dtype=bool)
    )
    if legacy_included is not None:
        if legacy_included.shape != shape:
            raise ValueError("Historical likelihood_included must match counts.")
        eligible = available & supported
        if np.any(legacy_included & ~eligible):
            raise ValueError(
                "likelihood_included may be true only where counts are available "
                "and the likelihood is supported."
            )
    if policy_included is None:
        included_by_policy = (
            np.ones(shape, dtype=bool)
            if legacy_included is None
            else legacy_included | ~(available & supported)
        )
    else:
        included_by_policy = np.asarray(policy_included, dtype=bool)
        if included_by_policy.shape != shape:
            raise ValueError("Historical policy_included must match counts.")
        if legacy_included is not None and not np.array_equal(
            available & supported & included_by_policy,
            legacy_included,
        ):
            raise ValueError(
                "likelihood_included and policy_included disagree on the final "
                "likelihood mask."
            )
    supplied_paths = emission_paths is not None or path_likelihood is not None
    paths = emission_paths if emission_paths is not None else path_likelihood
    if emission_paths is not None and path_likelihood is not None:
        raise ValueError("Supply emission_paths, not both modern and historical names.")
    if paths is None:
        from .path_compiler import build_major_low_emission_paths

        paths = build_major_low_emission_paths(major_cn, minor_cn)

    purity_array = np.asarray(purity, dtype=np.float64)
    normal_array = np.asarray(normal_cn, dtype=np.float64)
    scale = np.asarray(scaling, dtype=np.float64)
    if scale.shape != shape or np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("Historical scaling must be positive and match counts.")
    if purity_array.shape != shape or np.any(~np.isfinite(purity_array)) or np.any(
        (purity_array <= 0.0) | (purity_array > 1.0)
    ):
        raise ValueError("Historical purity must lie in (0, 1] and match counts.")
    if normal_array.shape != shape or np.any(~np.isfinite(normal_array)) or np.any(
        normal_array < 0.0
    ):
        raise ValueError("Historical normal_cn must be nonnegative and match counts.")
    major_array = np.asarray(major_cn, dtype=np.float64)
    minor_array = np.asarray(minor_cn, dtype=np.float64)
    mixed_cn_paths = supplied_paths and not paths.major_prior_weighted
    if not mixed_cn_paths:
        if major_array.shape != shape or minor_array.shape != shape:
            raise ValueError("Historical copy-number arrays must match counts.")
        if (
            np.any(~np.isfinite(major_array))
            or np.any(~np.isfinite(minor_array))
            or np.any(major_array < 0.0)
            or np.any(minor_array < 0.0)
            or np.any(minor_array > major_array)
        ):
            raise ValueError("Historical major/minor copy numbers are invalid.")
        tumor_total = major_array + minor_array
        denominator = (
            (1.0 - purity_array) * normal_array + purity_array * tumor_total
        )
        if np.any(~np.isfinite(denominator)) or np.any(denominator <= 0.0):
            raise ValueError("Historical copy-number denominator must be positive.")
        expected_scale = purity_array / denominator
        tolerance = 8.0 * _arithmetic_epsilon(
            scaling, purity, normal_cn, major_cn, minor_cn
        )
        if not np.allclose(
            scale,
            expected_scale,
            rtol=tolerance,
            atol=tolerance,
        ):
            raise ValueError(
                "Historical scaling is inconsistent with purity and one-state "
                "major/minor copy number. Supply fixed-prior emission_paths when "
                "major/minor are display states for a mixed-CN model."
            )
    else:
        # With explicit mixed-CN paths, major/minor may be dominant display
        # states. Recover the honest mean tumor total that generated the
        # historical likelihood scale without changing purity or normal CN.
        tumor_total = _mean_tumor_cn_for_scale(
            purity_array,
            normal_array,
            scale,
        )

    epsilon = float(eps)
    if not np.isfinite(epsilon) or not 0.0 < epsilon < 0.5:
        raise ValueError("eps must be finite and lie strictly in (0, 0.5).")
    if phi_upper is not None:
        historical_upper = np.asarray(phi_upper, dtype=np.float64)
        if historical_upper.shape != shape or np.any(~np.isfinite(historical_upper)):
            raise ValueError("Historical phi_upper must be finite and match counts.")
        if paths.constrain_probability_box:
            from ..core.objective import piecewise_affine_probability_upper

            expected_upper = piecewise_affine_probability_upper(
                first_scale=scale[..., None] * np.asarray(paths.first_copy),
                second_scale=scale[..., None] * np.asarray(paths.second_copy),
                switch=np.asarray(paths.switch_fraction),
                valid=np.asarray(paths.valid),
                eps=epsilon,
            )
        else:
            expected_upper = np.ones(shape, dtype=np.float64)
        tolerance = 8.0 * _arithmetic_epsilon(
            phi_upper, scaling, purity, normal_cn, major_cn, minor_cn
        )
        if not np.allclose(
            historical_upper,
            expected_upper,
            rtol=tolerance,
            atol=tolerance,
        ):
            raise ValueError(
                "Historical phi_upper is inconsistent with the compiled "
                "emission-path probability box."
            )

    reasons = exclusion_reasons if exclusion_reasons is not None else path_reasons
    reason_codes: np.ndarray | None = None
    if reasons is not None:
        reason_codes = exclusion_codes_from_strings(reasons)
        # Historical reason arrays often omitted count-unavailable provenance.
        missing = np.equal(reasons, None)  # noqa: E711
        reason_codes = np.asarray(reason_codes, dtype=np.uint8)
        reason_codes[~supported & missing] = int(ExclusionCode.LIKELIHOOD_UNSUPPORTED)
        reason_codes[
            supported & ~np.asarray(available, dtype=bool) & missing
        ] = int(
            ExclusionCode.COUNT_UNAVAILABLE
        )
        reason_codes[
            supported
            & np.asarray(available, dtype=bool)
            & ~included_by_policy
            & missing
        ] = int(ExclusionCode.LIKELIHOOD_EXCLUDED_BY_POLICY)
    codes = exclusion_code
    if codes is not None and reason_codes is not None:
        supplied_codes = np.asarray(codes)
        if supplied_codes.shape != shape:
            raise ValueError("Historical exclusion_code must match counts.")
        explicit_reason = ~np.equal(reasons, None)  # noqa: E711
        if np.any(
            np.asarray(supplied_codes, dtype=np.uint8)[explicit_reason]
            != reason_codes[explicit_reason]
        ):
            raise ValueError(
                "Historical exclusion reasons disagree with exclusion_code."
            )
    elif codes is None:
        codes = reason_codes

    return TumorData(
        tumor_id=tumor_id,
        mutation_ids=mutation_ids,
        region_ids=region_ids,
        alt_counts=alt_counts,
        total_counts=total_counts,
        purity=purity_array,
        major_cn=major_array,
        minor_cn=minor_array,
        normal_cn=normal_array,
        tumor_total_cn=tumor_total,
        count_available=np.asarray(available, dtype=bool),
        likelihood_supported=supported,
        policy_included=included_by_policy,
        emission_paths=paths,
        exclusion_code=codes,
    )


__all__ = [
    "EmissionPaths",
    "ExclusionCode",
    "TumorData",
    "exclusion_codes_from_strings",
    "migrate_v03_tumor_data",
    "tumor_data_fingerprint",
]
