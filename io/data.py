from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class PatientData:
    patient_id: str
    mutation_ids: list[str]
    sample_ids: list[str]
    alt_counts: np.ndarray
    total_counts: np.ndarray
    purity: np.ndarray
    major_cn: np.ndarray
    minor_cn: np.ndarray
    normal_cn: np.ndarray
    has_cna: np.ndarray
    scaling: np.ndarray
    phi_upper: np.ndarray
    phi_init: np.ndarray
    init_major_mask: np.ndarray

    @property
    def num_mutations(self) -> int:
        return int(self.alt_counts.shape[0])

    @property
    def num_samples(self) -> int:
        return int(self.alt_counts.shape[1])

    @property
    def tumor_id(self) -> str:
        return self.patient_id

    @property
    def region_ids(self) -> list[str]:
        return list(self.sample_ids)

    @property
    def num_regions(self) -> int:
        return self.num_samples

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
        return self.major_cn.astype(np.float32, copy=True)


TumorData = PatientData


def _first_seen(values: pd.Series) -> list[str]:
    return list(pd.Index(values.astype(str)).drop_duplicates())


def _safe_probability(scale: np.ndarray, multiplicity: np.ndarray, phi: np.ndarray, eps: float) -> np.ndarray:
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
    alt_counts = np.asarray(alt_counts, dtype=np.float32)
    total_counts = np.asarray(total_counts, dtype=np.float32)
    scaling = np.asarray(scaling, dtype=np.float32)
    major_cn = np.asarray(major_cn, dtype=np.float32)
    minor_cn = np.asarray(minor_cn, dtype=np.float32)
    phi_upper = np.asarray(phi_upper, dtype=np.float32)

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

    loglik_major = alt_counts * np.log(p_major) + (total_counts - alt_counts) * np.log1p(-p_major)
    loglik_minor = alt_counts * np.log(p_minor) + (total_counts - alt_counts) * np.log1p(-p_minor)

    init_major_mask = loglik_major >= loglik_minor
    phi_init = np.where(init_major_mask, phi_major, phi_minor).astype(np.float32)
    phi_init = np.clip(phi_init, eps, phi_upper)
    return phi_init.astype(np.float32), init_major_mask.astype(bool)


def load_patient_tsv(file_path: str | Path, eps: float = 1e-6) -> PatientData:
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
        raise ValueError(f"Missing purity column in {file_path}; expected 'purity' or 'tumour_content'.")

    if "normal_cn" not in df.columns:
        df["normal_cn"] = 2.0

    df["mutation_id"] = df["mutation_id"].astype(str)
    df["sample_id"] = df["sample_id"].astype(str)
    mutation_ids = _first_seen(df["mutation_id"])
    sample_ids = _first_seen(df["sample_id"])

    mut_index = {mutation_id: idx for idx, mutation_id in enumerate(mutation_ids)}
    sample_index = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}

    num_mutations = len(mutation_ids)
    num_samples = len(sample_ids)

    alt_counts = np.zeros((num_mutations, num_samples), dtype=np.float32)
    total_counts = np.zeros((num_mutations, num_samples), dtype=np.float32)
    purity = np.zeros((num_mutations, num_samples), dtype=np.float32)
    major_cn = np.zeros((num_mutations, num_samples), dtype=np.float32)
    minor_cn = np.zeros((num_mutations, num_samples), dtype=np.float32)
    normal_cn = np.zeros((num_mutations, num_samples), dtype=np.float32)
    has_cna = np.ones((num_mutations, num_samples), dtype=bool)

    for row in df.itertuples(index=False):
        i = mut_index[str(row.mutation_id)]
        j = sample_index[str(row.sample_id)]
        alt = float(row.alt_counts)
        ref = float(row.ref_counts)
        alt_counts[i, j] = alt
        total_counts[i, j] = alt + ref
        purity[i, j] = float(getattr(row, purity_col))
        major_cn[i, j] = float(row.major_cn)
        minor_cn[i, j] = float(row.minor_cn)
        normal_cn[i, j] = float(row.normal_cn)
        if "has_cna" in df.columns:
            has_cna[i, j] = bool(getattr(row, "has_cna"))
        elif "cna_observed" in df.columns:
            has_cna[i, j] = bool(getattr(row, "cna_observed"))

    purity = np.clip(purity, eps, 1.0 - eps)
    total_cn = major_cn + minor_cn
    denom = purity * total_cn + (1.0 - purity) * normal_cn
    scaling = purity / np.clip(denom, eps, None)

    max_prob_scale = np.maximum(scaling * major_cn, scaling * minor_cn)
    phi_upper = np.minimum(1.0, (1.0 - eps) / np.clip(max_prob_scale, eps, None)).astype(np.float32)
    phi_upper = np.clip(phi_upper, eps, 1.0).astype(np.float32)

    phi_init, init_major_mask = compute_phi_init_from_counts(
        alt_counts=alt_counts,
        total_counts=total_counts,
        scaling=scaling,
        major_cn=major_cn,
        minor_cn=minor_cn,
        phi_upper=phi_upper,
        eps=eps,
    )

    return PatientData(
        patient_id=file_path.stem,
        mutation_ids=mutation_ids,
        sample_ids=sample_ids,
        alt_counts=alt_counts,
        total_counts=total_counts,
        purity=purity.astype(np.float32),
        major_cn=major_cn.astype(np.float32),
        minor_cn=minor_cn.astype(np.float32),
        normal_cn=normal_cn.astype(np.float32),
        has_cna=has_cna.astype(bool),
        scaling=scaling.astype(np.float32),
        phi_upper=phi_upper.astype(np.float32),
        phi_init=phi_init.astype(np.float32),
        init_major_mask=init_major_mask.astype(bool),
    )


def load_tumor_tsv(file_path: str | Path, eps: float = 1e-6) -> TumorData:
    return load_patient_tsv(file_path=file_path, eps=eps)


__all__ = [
    "PatientData",
    "TumorData",
    "compute_phi_init_from_counts",
    "load_patient_tsv",
    "load_tumor_tsv",
]
