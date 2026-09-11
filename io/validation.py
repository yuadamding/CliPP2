"""Original-CN and biological validation shared by fitting entry points."""
from __future__ import annotations

import numpy as np
from ..config import _FitOptions, MAX_MAJOR_CN
from .data import TumorData
from .tumor_txt import CN_FILTER_POLICY_ID

def _validate_biological_arrays(data: TumorData) -> None:
    """Reject stale derived arrays and malformed programmatic replacements.

    Shape checks precede arithmetic so NumPy broadcasting cannot turn an
    inconsistent biological source into an apparently valid objective.
    """
    shape = (len(data.mutation_ids), len(data.region_ids))
    if 0 in shape or any(
        any(not isinstance(value, str) or not value.strip() for value in ids)
        or len(set(ids)) != len(ids)
        for ids in (data.mutation_ids, data.region_ids)
    ):
        raise ValueError("TumorData mutation/region identifiers must be nonempty and unique.")
    for name in (
        "alt_counts", "total_counts", "purity", "major_cn", "minor_cn",
        "normal_cn", "scaling", "phi_upper", "phi_init",
    ):
        values = np.asarray(getattr(data, name))
        if (
            not isinstance(getattr(data, name), np.ndarray)
            or values.shape != shape
            or values.dtype.kind not in "iuf"
            or not np.all(np.isfinite(values))
        ):
            raise ValueError(f"TumorData.{name} must be a finite numeric array of shape {shape}.")
    for name in ("count_observed",):
        values = getattr(data, name)
        if values is None and name == "count_observed":
            continue
        values = np.asarray(values)
        if values.shape != shape or values.dtype.kind != "b":
            raise ValueError(f"TumorData.{name} must be a Boolean array of shape {shape}.")
    for name in ("alt_counts", "total_counts", "major_cn", "minor_cn"):
        values = getattr(data, name)
        if np.any(values < 0) or np.any(values != np.rint(values)):
            raise ValueError(f"TumorData.{name} must contain nonnegative integers.")
    if np.any(data.alt_counts > data.total_counts):
        raise ValueError("TumorData.alt_counts cannot exceed total_counts.")
    if np.any((data.purity <= 0.0) | (data.purity > 1.0)):
        raise ValueError("TumorData.purity must lie in (0, 1].")
    if not np.all(data.purity == data.purity[:1]):
        raise ValueError("TumorData.purity must be constant within each region.")
    if np.any(data.normal_cn < 0.0):
        raise ValueError("TumorData.normal_cn must be nonnegative.")
    if (
        np.any((data.major_cn < 1) | (data.major_cn > MAX_MAJOR_CN))
        or np.any(data.minor_cn > data.major_cn)
    ):
        raise ValueError("TumorData must satisfy 0 <= minor_cn <= major_cn <= 6 and major_cn >= 1.")
    expected_scaling = data.purity / (
        (1.0 - data.purity) * data.normal_cn
        + data.purity * (data.major_cn + data.minor_cn)
    )
    if not np.allclose(data.scaling, expected_scaling, rtol=1e-12, atol=0.0):
        raise ValueError(
            "TumorData.scaling is inconsistent with purity and copy number; "
            "reload the input after changing biological values."
        )


def validate_public_tumor_data(data: TumorData, config: _FitOptions) -> None:
    """Require original-CN validation at the public fitting boundary.

    Retained CN arrays alone cannot establish original-input eligibility.
    """

    report = data.cn_filter_report
    if (
        report is None
        or report.policy_id != CN_FILTER_POLICY_ID
    ):
        raise ValueError(
            "The public fit requires clonal integer TumorData from "
            "load_tumor_txt, including its original-CN filtering report; "
            "legacy or unvalidated TumorData is not supported."
        )
    _validate_biological_arrays(data)
    excluded = set(report.excluded_mutation_ids)
    if (
        report.retained_mutation_count != data.num_mutations
        or report.input_mutation_count != data.num_mutations + len(excluded)
        or len(excluded) != len(report.excluded_mutation_ids)
        or excluded.intersection(data.mutation_ids)
    ):
        raise ValueError("CN filtering report is inconsistent with retained mutations.")
    epsilon = float(config.eps)
    expected_upper = np.clip(np.minimum(
        1.0, (1.0 - epsilon) / np.clip(data.scaling * data.major_cn, epsilon, None)
    ), epsilon, 1.0)
    if (
        not np.allclose(data.phi_upper, expected_upper, rtol=0.0, atol=1e-12)
        or not np.all(np.isfinite(data.phi_init))
        or np.any(data.phi_init < epsilon)
        or np.any(data.phi_init > expected_upper)
    ):
        raise ValueError(
            "Loaded CCF bounds/initialization do not match fit eps; "
            "reload the input with eps=config.eps."
        )
