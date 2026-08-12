from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PathLikelihoodSpec:
    """Immutable region-local categorical CCF path likelihood.

    Arrays have shape ``(mutation, region, path)``.  A path's unscaled
    mutant-copy mass is

    ``first_copy * min(phi, switch_fraction)
       + second_copy * max(phi - switch_fraction, 0)``.

    Every path has full CCF support on ``[0, 1]``. ``log_prior`` is fixed over
    CCF and normalized over ``valid`` paths for every mutation-region entry.
    ``legacy_major_indicator`` is optional
    compatibility metadata; it lets the generic likelihood reproduce the
    historical scalar major-copy posterior without assigning that meaning to
    new path families.  Model, generator, and prior identifiers are included
    in the parent ``TumorData`` fingerprint together with every numeric array.
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
    legacy_major_indicator: np.ndarray | None = None

    def __post_init__(self) -> None:
        model_id = str(self.model_id).strip()
        model_version = str(self.model_version).strip()
        candidate_generator_version = str(self.candidate_generator_version).strip()
        prior_mode = str(self.prior_mode).strip()
        if not model_id:
            raise ValueError("PathLikelihoodSpec.model_id must be nonempty.")
        if not model_version:
            raise ValueError("PathLikelihoodSpec.model_version must be nonempty.")
        if not candidate_generator_version:
            raise ValueError(
                "PathLikelihoodSpec.candidate_generator_version must be nonempty."
            )
        if not prior_mode:
            raise ValueError("PathLikelihoodSpec.prior_mode must be nonempty.")

        numeric: dict[str, np.ndarray] = {}
        for name in (
            "first_copy",
            "second_copy",
            "switch_fraction",
            "log_prior",
        ):
            value = np.array(
                getattr(self, name), dtype=np.float64, copy=True, order="C"
            )
            if value.ndim != 3:
                raise ValueError(
                    f"PathLikelihoodSpec.{name} must have shape (M, S, K)."
                )
            numeric[name] = value
        shape = numeric["first_copy"].shape
        if shape[2] <= 0:
            raise ValueError("PathLikelihoodSpec must contain at least one path.")
        for name, value in numeric.items():
            if value.shape != shape:
                raise ValueError(
                    f"PathLikelihoodSpec.{name} must have shape {shape}, "
                    f"not {value.shape}."
                )

        valid = np.array(self.valid, dtype=bool, copy=True, order="C")
        if valid.shape != shape:
            raise ValueError(
                f"PathLikelihoodSpec.valid must have shape {shape}, not {valid.shape}."
            )
        if not np.all(np.any(valid, axis=-1)):
            raise ValueError(
                "Every mutation-region entry must have at least one valid path."
            )
        if np.any(~np.isfinite(numeric["first_copy"][valid])) or np.any(
            numeric["first_copy"][valid] < 0.0
        ):
            raise ValueError("Valid first_copy values must be finite and nonnegative.")
        if np.any(~np.isfinite(numeric["second_copy"][valid])) or np.any(
            numeric["second_copy"][valid] < 0.0
        ):
            raise ValueError("Valid second_copy values must be finite and nonnegative.")
        switches = numeric["switch_fraction"][valid]
        if np.any(~np.isfinite(switches)) or np.any(
            (switches < 0.0) | (switches > 1.0)
        ):
            raise ValueError(
                "Valid switch_fraction values must be finite and lie in [0, 1]."
            )
        valid_log_prior = numeric["log_prior"][valid]
        if np.any(~np.isfinite(valid_log_prior)):
            raise ValueError("Valid log_prior values must be finite.")

        for name in ("first_copy", "second_copy", "switch_fraction"):
            numeric[name] = np.where(valid, numeric[name], 0.0)
        canonical_log_prior = np.where(valid, numeric["log_prior"], -np.inf)
        max_log_prior = np.max(canonical_log_prior, axis=-1, keepdims=True)
        log_normalizer = np.squeeze(
            max_log_prior
            + np.log(
                np.sum(
                    np.exp(canonical_log_prior - max_log_prior),
                    axis=-1,
                    keepdims=True,
                )
            ),
            axis=-1,
        )
        if not np.allclose(log_normalizer, 0.0, rtol=0.0, atol=1e-10):
            raise ValueError(
                "PathLikelihoodSpec.log_prior must be normalized over valid paths."
            )
        numeric["log_prior"] = canonical_log_prior

        indicator = self.legacy_major_indicator
        if indicator is not None:
            indicator = np.array(indicator, dtype=bool, copy=True, order="C")
            if indicator.shape != shape:
                raise ValueError(
                    "PathLikelihoodSpec.legacy_major_indicator must have shape "
                    f"{shape}, not {indicator.shape}."
                )
            indicator &= valid
            indicator.setflags(write=False)

        for name, value in numeric.items():
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        valid.setflags(write=False)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "model_version", model_version)
        object.__setattr__(
            self,
            "candidate_generator_version",
            candidate_generator_version,
        )
        object.__setattr__(self, "prior_mode", prior_mode)
        object.__setattr__(self, "legacy_major_indicator", indicator)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.first_copy.shape)

    @property
    def has_fixed_linear_emission(self) -> bool:
        """Whether every valid candidate in a unit has one shared linear slope.

        Such a categorical specification is only redundant bookkeeping: after
        marginalizing its normalized prior, each unit has the same fixed-copy
        likelihood as the existing scalar CliPP2 model.
        """

        first_valid_index = np.argmax(self.valid, axis=-1, keepdims=True)
        reference = np.take_along_axis(
            self.first_copy,
            first_valid_index,
            axis=-1,
        )
        linear = (~self.valid) | (self.first_copy == self.second_copy)
        shared = (~self.valid) | (self.first_copy == reference)
        return bool(np.all(linear & shared))

    def validate_observation_shape(self, shape: tuple[int, int]) -> None:
        if tuple(self.first_copy.shape[:2]) != tuple(shape):
            raise ValueError(
                "PathLikelihoodSpec mutation-region shape "
                f"{self.first_copy.shape[:2]} does not match observations {shape}."
            )


@dataclass
class TumorData:
    tumor_id: str
    mutation_ids: list[str]
    region_ids: list[str]
    alt_counts: np.ndarray  # float64 (M, S)
    total_counts: np.ndarray  # float64 (M, S)
    purity: np.ndarray  # float64 (M, S)
    major_cn: np.ndarray  # float64 (M, S)
    minor_cn: np.ndarray  # float64 (M, S)
    normal_cn: np.ndarray  # float64 (M, S)
    has_cna: np.ndarray  # bool (M, S)
    scaling: np.ndarray  # float64 (M, S)
    phi_upper: np.ndarray  # float64 (M, S)
    phi_init: np.ndarray  # float64 (M, S)
    init_major_mask: np.ndarray  # bool (M, S)
    count_observed: np.ndarray | None = (
        None  # bool (M, S) — True if counts observed; None means all observed
    )
    path_likelihood: PathLikelihoodSpec | None = None
    # Reporting-only metadata aligned to ``path_likelihood``.  These fields do
    # not alter the observed objective and are therefore intentionally excluded
    # from ``tumor_data_fingerprint``.
    path_reporting_fingerprint: str | None = None
    path_unsupported_reason: np.ndarray | None = None
    mean_tumor_total_cn: np.ndarray | None = None

    @property
    def num_mutations(self) -> int:
        return int(self.alt_counts.shape[0])

    @property
    def num_regions(self) -> int:
        return int(self.alt_counts.shape[1])

    @property
    def depth_scale(self) -> float:
        positive_depth = self.total_counts[self.total_counts > 0]
        if positive_depth.size == 0:
            return 1.0
        return float(np.median(positive_depth))

    @property
    def multiplicity_estimation_mask(self) -> np.ndarray:
        distinct_candidates = ~np.isclose(self.major_cn, self.minor_cn)
        positive_candidates = (self.major_cn > 0.0) & (self.minor_cn > 0.0)
        non_diploid = (self.major_cn != 1.0) | (self.minor_cn != 1.0)
        return self.has_cna & non_diploid & distinct_candidates & positive_candidates

    @property
    def fixed_multiplicity(self) -> np.ndarray:
        # Outside CNA-ambiguous entries, keep multiplicity fixed at the available major-copy value.
        return self.major_cn.astype(np.float64, copy=True)


def tumor_data_fingerprint(data: TumorData) -> str:
    """Return a deterministic identity for every observed-objective input."""

    digest = hashlib.sha256()

    def update_text(value: str) -> None:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)

    def update_text_sequence(values: list[str]) -> None:
        digest.update(len(values).to_bytes(8, "little"))
        for value in values:
            update_text(value)

    def update_array(name: str, values: np.ndarray) -> None:
        update_text(name)
        array = np.ascontiguousarray(np.asarray(values))
        update_text(str(array.dtype))
        digest.update(len(array.shape).to_bytes(8, "little"))
        for dimension in array.shape:
            digest.update(int(dimension).to_bytes(8, "little", signed=True))
        digest.update(array.tobytes())

    update_text(data.tumor_id)
    update_text_sequence(list(data.mutation_ids))
    update_text_sequence(list(data.region_ids))
    for name in (
        "alt_counts",
        "total_counts",
        "purity",
        "major_cn",
        "minor_cn",
        "normal_cn",
        "has_cna",
        "scaling",
        "phi_upper",
        "phi_init",
        "init_major_mask",
    ):
        update_array(name, getattr(data, name))
    count_observed = getattr(data, "count_observed", None)
    update_array(
        "count_observed",
        np.ones_like(np.asarray(data.alt_counts), dtype=bool)
        if count_observed is None
        else np.asarray(count_observed, dtype=bool),
    )
    path_likelihood = getattr(data, "path_likelihood", None)
    if path_likelihood is not None:
        path_likelihood.validate_observation_shape(
            (int(data.num_mutations), int(data.num_regions))
        )
        update_text("path_likelihood:present")
        update_text(path_likelihood.model_id)
        update_text(path_likelihood.model_version)
        update_text(path_likelihood.candidate_generator_version)
        update_text(path_likelihood.prior_mode)
        for name in (
            "first_copy",
            "second_copy",
            "switch_fraction",
            "log_prior",
            "valid",
        ):
            update_array(f"path_likelihood.{name}", getattr(path_likelihood, name))
        indicator = path_likelihood.legacy_major_indicator
        update_text(
            "path_likelihood.legacy_major_indicator:"
            + ("none" if indicator is None else "present")
        )
        if indicator is not None:
            update_array("path_likelihood.legacy_major_indicator", indicator)
    return digest.hexdigest()


def tumor_objective_fingerprint(data: TumorData) -> str:
    """Return a representation-neutral identity for the numeric likelihood.

    Initialization, reporting fields, display identifiers, and compiler
    version labels are deliberately excluded. The solver combines this digest
    with its graph arrays, epsilon, and effective prior to identify the full
    optimization objective.
    """

    digest = hashlib.sha256()

    def update_array(name: str, values: np.ndarray) -> None:
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "little"))
        digest.update(encoded_name)
        array = np.ascontiguousarray(np.asarray(values))
        encoded_dtype = str(array.dtype).encode("utf-8")
        digest.update(len(encoded_dtype).to_bytes(8, "little"))
        digest.update(encoded_dtype)
        digest.update(len(array.shape).to_bytes(8, "little"))
        for dimension in array.shape:
            digest.update(int(dimension).to_bytes(8, "little", signed=True))
        digest.update(array.tobytes())

    for name in ("alt_counts", "total_counts", "scaling", "phi_upper"):
        update_array(name, getattr(data, name))
    count_observed = getattr(data, "count_observed", None)
    update_array(
        "count_observed",
        np.ones_like(np.asarray(data.alt_counts), dtype=bool)
        if count_observed is None
        else np.asarray(count_observed, dtype=bool),
    )

    path_likelihood = getattr(data, "path_likelihood", None)
    if path_likelihood is None:
        for name in ("major_cn", "minor_cn", "has_cna"):
            update_array(name, getattr(data, name))
    else:
        path_likelihood.validate_observation_shape(
            (int(data.num_mutations), int(data.num_regions))
        )
        for name in (
            "first_copy",
            "second_copy",
            "switch_fraction",
            "log_prior",
            "valid",
        ):
            update_array(f"path_likelihood.{name}", getattr(path_likelihood, name))
    return digest.hexdigest()


def _safe_probability(
    scale: np.ndarray, multiplicity: np.ndarray, phi: np.ndarray, eps: float
) -> np.ndarray:
    return np.clip(scale * multiplicity * phi, eps, 1.0 - eps)


def compute_phi_init_from_counts(
    *,
    alt_counts: np.ndarray,
    total_counts: np.ndarray,
    scaling: np.ndarray,
    major_cn: np.ndarray,
    minor_cn: np.ndarray,
    phi_upper: np.ndarray,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    alt_counts = np.asarray(alt_counts, dtype=np.float64)
    total_counts = np.asarray(total_counts, dtype=np.float64)
    scaling = np.asarray(scaling, dtype=np.float64)
    major_cn = np.asarray(major_cn, dtype=np.float64)
    minor_cn = np.asarray(minor_cn, dtype=np.float64)
    phi_upper = np.asarray(phi_upper, dtype=np.float64)

    smoothed_vaf = (alt_counts + 0.5) / (total_counts + 1.0)

    phi_major = np.divide(
        smoothed_vaf,
        np.clip(scaling * major_cn, eps, None),
        out=np.zeros_like(smoothed_vaf),
        where=major_cn > 0,
    )
    phi_major = np.clip(phi_major, 0.0, phi_upper)

    phi_minor = np.divide(
        smoothed_vaf,
        np.clip(scaling * minor_cn, eps, None),
        out=np.zeros_like(smoothed_vaf),
        where=minor_cn > 0,
    )
    phi_minor = np.clip(phi_minor, 0.0, phi_upper)

    p_major = _safe_probability(scaling, major_cn, phi_major, eps)
    p_minor = _safe_probability(scaling, minor_cn, phi_minor, eps)

    loglik_major = alt_counts * np.log(p_major) + (
        total_counts - alt_counts
    ) * np.log1p(-p_major)
    loglik_minor = alt_counts * np.log(p_minor) + (
        total_counts - alt_counts
    ) * np.log1p(-p_minor)

    init_major_mask = loglik_major >= loglik_minor
    phi_init = np.where(init_major_mask, phi_major, phi_minor)
    phi_init = np.clip(phi_init, eps, phi_upper)
    return phi_init.astype(np.float64), init_major_mask.astype(bool)


__all__ = [
    "PathLikelihoodSpec",
    "TumorData",
    "compute_phi_init_from_counts",
    "tumor_data_fingerprint",
    "tumor_objective_fingerprint",
]
