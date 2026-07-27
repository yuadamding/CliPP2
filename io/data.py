from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def canonicalize_single_switch_path_values(
    candidates: Iterable[tuple[float, float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Canonicalize numeric single-switch paths and assign a uniform prior.

    Each candidate is ``(first_copy, second_copy, switch_fraction)``. Equal
    slopes describe the same linear function for every switch, so they are
    represented with ``switch_fraction=1`` before exact deduplication.
    """

    canonical: set[tuple[float, float, float]] = set()
    for index, candidate in enumerate(candidates):
        values = np.asarray(candidate, dtype=np.float64)
        if values.shape != (3,):
            raise ValueError(
                "Every path candidate must be a three-value "
                "(first_copy, second_copy, switch_fraction) tuple; "
                f"candidate {index} has shape {values.shape}."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Path candidate {index} contains a non-finite value.")
        first_copy, second_copy, switch_fraction = (float(value) for value in values)
        if first_copy < 0.0 or second_copy < 0.0:
            raise ValueError(
                f"Path candidate {index} has a negative copy-number slope."
            )
        if not 0.0 <= switch_fraction <= 1.0:
            raise ValueError(
                f"Path candidate {index} has switch_fraction outside [0, 1]."
            )
        if first_copy == second_copy:
            switch_fraction = 1.0
        canonical.add((first_copy, second_copy, switch_fraction))

    if not canonical:
        raise ValueError("At least one path candidate is required.")
    ordered = sorted(canonical, key=lambda item: (item[2], item[0], item[1]))
    paths = np.asarray(ordered, dtype=np.float64)
    log_prior = np.full(len(ordered), -np.log(float(len(ordered))), dtype=np.float64)
    paths.setflags(write=False)
    log_prior.setflags(write=False)
    return paths, log_prior


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
    def num_paths(self) -> int:
        return int(self.first_copy.shape[-1])

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
    path_annotations: object | None = None
    path_reporting_semantics: str | None = None
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


def legacy_path_likelihood_spec(
    data: TumorData,
    *,
    major_prior: float,
) -> PathLikelihoodSpec:
    """Represent the historical fixed/major-minor likelihood categorically.

    This adapter is intentionally explicit rather than installed by default:
    leaving ``TumorData.path_likelihood`` unset keeps every legacy solver path
    and result bit-for-bit unchanged while allowing parity tests and gradual
    generic-likelihood integration.
    """

    prior = float(major_prior)
    if not np.isfinite(prior) or not (0.0 < prior < 1.0):
        raise ValueError("major_prior must lie strictly in (0, 1).")
    shape = (int(data.num_mutations), int(data.num_regions), 2)
    ambiguous = np.asarray(data.multiplicity_estimation_mask, dtype=bool)
    first_copy = np.stack(
        (
            np.asarray(data.minor_cn, dtype=np.float64),
            np.asarray(data.major_cn, dtype=np.float64),
        ),
        axis=-1,
    )
    second_copy = first_copy.copy()
    switch_fraction = np.ones(shape, dtype=np.float64)
    valid = np.stack((ambiguous, np.ones_like(ambiguous)), axis=-1)
    log_prior = np.full(shape, -np.inf, dtype=np.float64)
    log_prior[..., 0] = np.where(ambiguous, np.log1p(-prior), -np.inf)
    log_prior[..., 1] = np.where(ambiguous, np.log(prior), 0.0)
    legacy_major_indicator = np.zeros(shape, dtype=bool)
    legacy_major_indicator[..., 1] = True
    return PathLikelihoodSpec(
        model_id="clipp2_legacy_major_minor_v1",
        model_version="1",
        candidate_generator_version="legacy_major_minor_adapter_v1",
        prior_mode="configured_major_prior_v1",
        first_copy=first_copy,
        second_copy=second_copy,
        switch_fraction=switch_fraction,
        log_prior=log_prior,
        valid=valid,
        legacy_major_indicator=legacy_major_indicator,
    )


def _first_seen(values: pd.Series) -> list[str]:
    return list(pd.Index(values.astype(str)).drop_duplicates())


def _parse_bool_like(value: object, *, column_name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise ValueError(f"Missing boolean value in column '{column_name}'.")
    if isinstance(value, (int, np.integer)):
        if int(value) in {0, 1}:
            return bool(int(value))
        raise ValueError(
            f"Invalid integer boolean value {value!r} in column '{column_name}'."
        )
    if isinstance(value, (float, np.floating)):
        if float(value) in {0.0, 1.0}:
            return bool(int(value))
        raise ValueError(
            f"Invalid float boolean value {value!r} in column '{column_name}'."
        )
    normalized = str(value).strip().lower()
    if normalized in {"true", "t", "yes", "y", "1"}:
        return True
    if normalized in {"false", "f", "no", "n", "0"}:
        return False
    raise ValueError(f"Invalid boolean value {value!r} in column '{column_name}'.")


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


def _validate_inputs_strict(
    *,
    file_path: Path,
    alt_counts: np.ndarray,
    total_counts: np.ndarray,
    purity: np.ndarray,
    major_cn: np.ndarray,
    minor_cn: np.ndarray,
    normal_cn: np.ndarray,
) -> None:
    errors: list[str] = []
    for name, matrix in [
        ("alt_counts", alt_counts),
        ("total_counts", total_counts),
        ("purity", purity),
        ("major_cn", major_cn),
        ("minor_cn", minor_cn),
        ("normal_cn", normal_cn),
    ]:
        if not np.all(np.isfinite(matrix)):
            errors.append(f"Non-finite values in '{name}'.")
    if errors:
        raise ValueError(f"Invalid input data in {file_path}: {'; '.join(errors)}")

    if np.any(alt_counts < 0.0):
        errors.append("Negative alt_counts found.")
    if np.any(total_counts < 0.0):
        errors.append("Negative total_counts found.")
    if np.any(alt_counts > total_counts + 0.5):
        errors.append("alt_counts > total_counts found.")
    if np.any(np.abs(alt_counts - np.round(alt_counts)) > 1e-6):
        errors.append("Non-integer alt_counts found.")
    if np.any(np.abs(total_counts - np.round(total_counts)) > 1e-6):
        errors.append("Non-integer total_counts found.")
    if np.any(purity <= 0.0):
        errors.append("Purity must be strictly positive (purity <= 0 found).")
    if np.any(purity > 1.0 + 1e-9):
        errors.append("Purity > 1 found.")
    if np.any(major_cn < 0.0):
        errors.append("Negative major_cn found.")
    if np.any(minor_cn < 0.0):
        errors.append("Negative minor_cn found.")
    if np.any(major_cn < minor_cn - 1e-9):
        errors.append("major_cn < minor_cn found (major must be >= minor).")
    if np.any(normal_cn <= 0.0):
        errors.append("Nonpositive normal_cn found.")
    if errors:
        raise ValueError(f"Invalid input data in {file_path}: {'; '.join(errors)}")


def _load_observation_tsv(
    file_path: str | Path,
    eps: float = 1e-6,
    *,
    missing_cna_policy: str = "error",
    validation_mode: str = "strict",
) -> TumorData:
    file_path = Path(file_path)
    df = pd.read_csv(
        file_path,
        sep="\t",
        dtype={
            "mutation_id": str,
            "sample_id": str,
            "region_id": str,
        },
        keep_default_na=False,
        na_values=[""],
    ).copy()

    required = {
        "mutation_id",
        "ref_counts",
        "alt_counts",
        "major_cn",
        "minor_cn",
    }
    if "sample_id" not in df.columns and "region_id" in df.columns:
        df["sample_id"] = df["region_id"]
    elif "sample_id" in df.columns and "region_id" not in df.columns:
        df["region_id"] = df["sample_id"]

    required.add("sample_id")
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {file_path}: {sorted(missing)}. "
            "Expected region identifiers in 'region_id' or 'sample_id'."
        )

    if "purity" in df.columns:
        purity_col = "purity"
    elif "tumour_content" in df.columns:
        purity_col = "tumour_content"
    else:
        raise ValueError(
            f"Missing purity column in {file_path}; expected 'purity' or 'tumour_content'."
        )

    if "normal_cn" not in df.columns:
        df["normal_cn"] = 2.0

    normalized_validation_mode = str(validation_mode).strip().lower()
    if normalized_validation_mode not in {"strict", "lenient"}:
        raise ValueError("validation_mode must be 'strict' or 'lenient'.")

    df["mutation_id"] = df["mutation_id"].astype(str)
    df["sample_id"] = df["sample_id"].astype(str)
    pair_df = df.loc[:, ["mutation_id", "sample_id"]].copy()
    duplicate_mask = pair_df.duplicated(keep=False)
    if bool(duplicate_mask.any()):
        duplicate_pairs = pair_df.loc[duplicate_mask].drop_duplicates().head(5)
        duplicate_examples = ", ".join(
            f"({row.mutation_id}, {row.sample_id})"
            for row in duplicate_pairs.itertuples(index=False)
        )
        raise ValueError(
            f"Duplicate mutation-region rows found in {file_path}. "
            f"Examples: {duplicate_examples}"
        )

    mutation_ids = _first_seen(df["mutation_id"])
    region_ids = _first_seen(df["sample_id"])

    # Completeness check: number of rows must equal M * S after dedup
    num_mutations = len(mutation_ids)
    num_regions = len(region_ids)
    if len(df) != num_mutations * num_regions:
        mutation_codes_check = pd.Categorical(
            df["mutation_id"], categories=mutation_ids
        ).codes
        region_codes_check = pd.Categorical(
            df["sample_id"], categories=region_ids
        ).codes
        observed_set = set(
            zip(mutation_codes_check.tolist(), region_codes_check.tolist())
        )
        expected_set = {
            (i, j) for i in range(num_mutations) for j in range(num_regions)
        }
        missing_pairs_coded = sorted(expected_set.difference(observed_set))[:5]
        missing_examples = ", ".join(
            f"({mutation_ids[i]}, {region_ids[j]})" for i, j in missing_pairs_coded
        )
        raise ValueError(
            f"Incomplete mutation-region matrix in {file_path}; "
            f"expected {num_mutations * num_regions} rows, got {len(df)}. "
            f"Missing examples: {missing_examples}"
        )

    # Vectorized fill using categorical integer codes
    mutation_codes = pd.Categorical(
        df["mutation_id"], categories=mutation_ids
    ).codes.copy()
    region_codes = pd.Categorical(df["sample_id"], categories=region_ids).codes.copy()

    alt_vals = df["alt_counts"].to_numpy(dtype=np.float64)
    ref_vals = df["ref_counts"].to_numpy(dtype=np.float64)
    purity_vals = df[purity_col].to_numpy(dtype=np.float64)
    major_vals = df["major_cn"].to_numpy(dtype=np.float64)
    minor_vals = df["minor_cn"].to_numpy(dtype=np.float64)
    normal_vals = df["normal_cn"].to_numpy(dtype=np.float64)

    alt_counts = np.full((num_mutations, num_regions), np.nan, dtype=np.float64)
    total_counts = np.full((num_mutations, num_regions), np.nan, dtype=np.float64)
    purity = np.full((num_mutations, num_regions), np.nan, dtype=np.float64)
    major_cn = np.full((num_mutations, num_regions), np.nan, dtype=np.float64)
    minor_cn = np.full((num_mutations, num_regions), np.nan, dtype=np.float64)
    normal_cn = np.full((num_mutations, num_regions), np.nan, dtype=np.float64)

    alt_counts[mutation_codes, region_codes] = alt_vals
    total_counts[mutation_codes, region_codes] = alt_vals + ref_vals
    purity[mutation_codes, region_codes] = purity_vals
    major_cn[mutation_codes, region_codes] = major_vals
    minor_cn[mutation_codes, region_codes] = minor_vals
    normal_cn[mutation_codes, region_codes] = normal_vals

    has_explicit_cna_mask = "has_cna" in df.columns or "cna_observed" in df.columns
    normalized_missing_cna_policy = str(missing_cna_policy).strip().lower()
    if normalized_missing_cna_policy not in {"error", "all_true"}:
        raise ValueError("missing_cna_policy must be one of {'error', 'all_true'}.")
    if has_explicit_cna_mask:
        has_cna = np.zeros((num_mutations, num_regions), dtype=bool)
        cna_col = "has_cna" if "has_cna" in df.columns else "cna_observed"
        has_cna_vals = np.array(
            [_parse_bool_like(v, column_name=cna_col) for v in df[cna_col]],
            dtype=bool,
        )
        has_cna[mutation_codes, region_codes] = has_cna_vals
    else:
        if normalized_missing_cna_policy == "error":
            raise ValueError(
                f"Missing CNA observability column in {file_path}; expected 'has_cna' or 'cna_observed'. "
                "Pass missing_cna_policy='all_true' only if that behavior is intentional."
            )
        has_cna = np.ones((num_mutations, num_regions), dtype=bool)

    # count_observed mask
    if "count_observed" in df.columns:
        count_obs_vals = np.array(
            [
                _parse_bool_like(v, column_name="count_observed")
                for v in df["count_observed"]
            ],
            dtype=bool,
        )
        count_observed = np.zeros((num_mutations, num_regions), dtype=bool)
        count_observed[mutation_codes, region_codes] = count_obs_vals
    else:
        count_observed = np.ones((num_mutations, num_regions), dtype=bool)

    for name, matrix in {
        "alt_counts": alt_counts,
        "total_counts": total_counts,
        "purity": purity,
        "major_cn": major_cn,
        "minor_cn": minor_cn,
        "normal_cn": normal_cn,
    }.items():
        if not np.all(np.isfinite(matrix)):
            raise ValueError(
                f"Non-finite values in matrix '{name}' after loading {file_path}."
            )

    if normalized_validation_mode == "strict":
        _validate_inputs_strict(
            file_path=file_path,
            alt_counts=alt_counts,
            total_counts=total_counts,
            purity=purity,
            major_cn=major_cn,
            minor_cn=minor_cn,
            normal_cn=normal_cn,
        )

    total_cn = major_cn + minor_cn
    denom = purity * total_cn + (1.0 - purity) * normal_cn
    bad_denom = ~(denom > 0.0)
    if bad_denom.any():
        raise ValueError(
            f"Non-positive copy-number denominator found in {file_path}. "
            "This can occur when purity=1 and total_cn=0 simultaneously."
        )
    scaling = purity / denom

    max_prob_scale = np.maximum(scaling * major_cn, scaling * minor_cn)
    phi_upper = np.minimum(1.0, (1.0 - eps) / np.clip(max_prob_scale, eps, None))
    phi_upper = np.clip(phi_upper, eps, 1.0)

    phi_init, init_major_mask = compute_phi_init_from_counts(
        alt_counts=alt_counts,
        total_counts=total_counts,
        scaling=scaling,
        major_cn=major_cn,
        minor_cn=minor_cn,
        phi_upper=phi_upper,
        eps=eps,
    )

    return TumorData(
        tumor_id=file_path.stem,
        mutation_ids=mutation_ids,
        region_ids=region_ids,
        alt_counts=alt_counts.astype(np.float64, copy=False),
        total_counts=total_counts.astype(np.float64, copy=False),
        purity=purity.astype(np.float64, copy=False),
        major_cn=major_cn.astype(np.float64, copy=False),
        minor_cn=minor_cn.astype(np.float64, copy=False),
        normal_cn=normal_cn.astype(np.float64, copy=False),
        has_cna=has_cna.astype(bool, copy=False),
        scaling=scaling.astype(np.float64, copy=False),
        phi_upper=phi_upper.astype(np.float64, copy=False),
        phi_init=phi_init.astype(np.float64, copy=False),
        init_major_mask=init_major_mask.astype(bool, copy=False),
        count_observed=count_observed.astype(bool, copy=False),
    )


__all__ = [
    "PathLikelihoodSpec",
    "TumorData",
    "canonicalize_single_switch_path_values",
    "compute_phi_init_from_counts",
    "legacy_path_likelihood_spec",
    "tumor_data_fingerprint",
    "tumor_objective_fingerprint",
]
