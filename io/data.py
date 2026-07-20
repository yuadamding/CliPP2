from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


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
    return digest.hexdigest()


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


def load_tumor_tsv(
    file_path: str | Path,
    eps: float = 1e-6,
    *,
    missing_cna_policy: str = "error",
    validation_mode: str = "strict",
) -> TumorData:
    file_path = Path(file_path)
    df = pd.read_csv(file_path, sep="\t").copy()

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
    "TumorData",
    "compute_phi_init_from_counts",
    "load_tumor_tsv",
    "tumor_data_fingerprint",
]
