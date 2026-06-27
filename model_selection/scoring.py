from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.model import FitOptions, FitResult
from ..io.data import TumorData
from ..core.bic import (
    compute_classic_bic,
    compute_extended_bic,
)

def _normalize_selection_score_name(selection_score: str) -> str:
    normalized = str(selection_score).strip().lower()
    if normalized == "bic":
        return "bic"
    raise ValueError(f"Unknown selection_score: {selection_score}")


def _selection_score_value(
    *,
    loglik: float,
    num_clusters: int,
    data: TumorData,
    bic_df_scale: float,
    bic_cluster_penalty: float,
    selection_score: str,
) -> tuple[float, float, float]:
    classic_bic = compute_classic_bic(loglik, num_clusters, data)
    extended_bic = compute_extended_bic(
        loglik,
        num_clusters,
        data,
        bic_df_scale=bic_df_scale,
        bic_cluster_penalty=bic_cluster_penalty,
    )
    normalized = _normalize_selection_score_name(selection_score)
    if normalized == "bic":
        return float(classic_bic), float(classic_bic), float(extended_bic)
    raise ValueError(f"Unknown normalized selection_score: {selection_score}")


def _is_bic_selection_eligible(
    *,
    raw_kkt_eligible: bool,
    classic_bic: float,
    bic_refit_finite_candidate_found: bool | None = None,
    bic_refit_converged: bool | None = None,
) -> bool:
    refit_finite = (
        bool(bic_refit_finite_candidate_found)
        if bic_refit_finite_candidate_found is not None
        else bool(bic_refit_converged)
    )
    return bool(
        bool(raw_kkt_eligible)
        and refit_finite
        and np.isfinite(float(classic_bic))
    )


def _bic_selection_eligible_mask(search_df: pd.DataFrame) -> np.ndarray:
    n_rows = int(search_df.shape[0])
    if "bic_selection_eligible" in search_df.columns:
        return search_df["bic_selection_eligible"].astype(bool).to_numpy(dtype=bool)
    if any(
        column in search_df.columns
        for column in (
            "raw_kkt_eligible",
            "bic_refit_finite_candidate_found",
            "bic_refit_converged",
            "classic_bic",
            "bic",
        )
    ):
        return _add_bic_selection_eligible(search_df)["bic_selection_eligible"].astype(bool).to_numpy(dtype=bool)
    if "selection_eligible" in search_df.columns:
        return search_df["selection_eligible"].astype(bool).to_numpy(dtype=bool)
    if "converged" in search_df.columns:
        return search_df["converged"].astype(bool).to_numpy(dtype=bool)
    return np.zeros(n_rows, dtype=bool)


def _row_bic_selection_eligible(row: pd.Series) -> bool:
    return bool(row.get("bic_selection_eligible", row.get("selection_eligible", row.get("converged", False))))


def _bool_with_default(value: object, default: bool = True) -> bool:
    if value is None:
        return bool(default)
    try:
        if pd.isna(value):
            return bool(default)
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"0", "false", "f", "no", "n", ""}:
            return False
        if normalized in {"1", "true", "t", "yes", "y"}:
            return True
    return bool(value)


def _row_lambda_applicable(row: pd.Series) -> bool:
    return _bool_with_default(row.get("lambda_applicable", True), default=True)


def _row_lambda_if_applicable(row: pd.Series) -> float | None:
    if not _row_lambda_applicable(row):
        return None
    try:
        value = float(row.get("lambda", np.nan))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value) or value < 0.0:
        return None
    return value


def _lambda_applicable_mask(frame: pd.DataFrame) -> np.ndarray:
    if frame.empty:
        return np.zeros(0, dtype=bool)
    if "lambda_applicable" in frame.columns:
        mask = frame["lambda_applicable"].map(_bool_with_default).to_numpy(dtype=bool, copy=True)
    else:
        mask = np.ones(frame.shape[0], dtype=bool)
    if "lambda" in frame.columns:
        lambdas = frame["lambda"].to_numpy(dtype=float)
        mask &= np.isfinite(lambdas) & (lambdas >= 0.0)
    return mask


def _add_bic_selection_eligible(search_df: pd.DataFrame) -> pd.DataFrame:
    if search_df.empty:
        return search_df.copy()
    enriched = search_df.copy()
    n_rows = int(enriched.shape[0])
    if "raw_kkt_eligible" in enriched.columns:
        raw_kkt = enriched["raw_kkt_eligible"].astype(bool).to_numpy(dtype=bool)
    elif "selection_eligible" in enriched.columns:
        raw_kkt = enriched["selection_eligible"].astype(bool).to_numpy(dtype=bool)
    elif "converged" in enriched.columns:
        raw_kkt = enriched["converged"].astype(bool).to_numpy(dtype=bool)
    else:
        raw_kkt = np.zeros(n_rows, dtype=bool)
    if "candidate_pool_source" in enriched.columns:
        candidate_source = enriched["candidate_pool_source"].astype(str)
        partition_candidate = candidate_source.eq("likelihood_partition").to_numpy(dtype=bool)
    elif "search_phase" in enriched.columns:
        partition_candidate = (
            enriched["search_phase"].astype(str).eq("likelihood_partition").to_numpy(dtype=bool)
        )
    else:
        partition_candidate = np.zeros(n_rows, dtype=bool)
    if "bic_refit_finite_candidate_found" in enriched.columns:
        bic_refit = enriched["bic_refit_finite_candidate_found"].astype(bool).to_numpy(dtype=bool)
    elif "bic_refit_converged" in enriched.columns:
        bic_refit = enriched["bic_refit_converged"].astype(bool).to_numpy(dtype=bool)
    else:
        # Absent certificate means unknown, treated as False (not True)
        bic_refit = np.zeros(n_rows, dtype=bool)
    if "classic_bic" in enriched.columns:
        classic_bic = enriched["classic_bic"].to_numpy(dtype=float)
    elif "bic" in enriched.columns:
        classic_bic = enriched["bic"].to_numpy(dtype=float)
    else:
        classic_bic = np.full(n_rows, np.nan, dtype=float)
    enriched["bic_selection_eligible"] = (raw_kkt | partition_candidate) & bic_refit & np.isfinite(classic_bic)
    return enriched


def _annotate_bic_diagnostics(search_df: pd.DataFrame) -> pd.DataFrame:
    if search_df.empty:
        return search_df.copy()
    enriched = _add_bic_selection_eligible(search_df)
    if {"bic_df", "bic_n_eff"}.issubset(enriched.columns):
        bic_df = enriched["bic_df"].to_numpy(dtype=float)
        bic_n_eff = np.maximum(enriched["bic_n_eff"].to_numpy(dtype=float), 1.0)
        enriched["bic_penalty"] = bic_df * np.log(bic_n_eff)
    elif "bic_penalty" not in enriched.columns:
        enriched["bic_penalty"] = np.nan

    for column in ("delta_loglik_vs_one_cluster", "delta_bic_vs_one_cluster"):
        if column not in enriched.columns:
            enriched[column] = np.nan
    if not {"n_clusters", "classic_bic", "bic_loglik"}.issubset(enriched.columns):
        return enriched
    n_clusters = enriched["n_clusters"].to_numpy(dtype=float)
    one_cluster = enriched.loc[(n_clusters == 1.0) & np.isfinite(enriched["classic_bic"].to_numpy(dtype=float))].copy()
    if one_cluster.empty:
        return enriched
    one_cluster["_bic_eligible_for_baseline"] = _bic_selection_eligible_mask(one_cluster)
    baseline = one_cluster.sort_values(
        ["_bic_eligible_for_baseline", "classic_bic", "lambda", "selection_step"],
        ascending=[False, True, True, True],
    ).iloc[0]
    baseline_loglik = float(baseline.get("bic_loglik", np.nan))
    baseline_bic = float(baseline.get("classic_bic", np.nan))
    if np.isfinite(baseline_loglik):
        enriched["delta_loglik_vs_one_cluster"] = enriched["bic_loglik"].to_numpy(dtype=float) - baseline_loglik
    if np.isfinite(baseline_bic):
        enriched["delta_bic_vs_one_cluster"] = enriched["classic_bic"].to_numpy(dtype=float) - baseline_bic
    return enriched


def _optimal_lambda_range(
    values: np.ndarray,
    lambdas: np.ndarray,
    *,
    maximize: bool,
) -> tuple[float | None, float | None, int, float | None, np.ndarray]:
    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        empty_mask = np.zeros_like(values, dtype=bool)
        return None, None, 0, None, empty_mask

    finite_values = values[finite_mask]
    best_value = float(np.max(finite_values) if maximize else np.min(finite_values))
    optimal_mask = finite_mask & np.isclose(values, best_value, rtol=0.0, atol=1e-12)
    lambda_values = np.unique(np.round(lambdas[optimal_mask].astype(float, copy=False), 12))
    return (
        float(np.min(lambda_values)),
        float(np.max(lambda_values)),
        int(lambda_values.size),
        best_value,
        optimal_mask,
    )


def _lambda_range_for_optimal_rows(
    frame: pd.DataFrame,
    optimal_mask: np.ndarray,
) -> tuple[float | None, float | None, int]:
    if frame.empty or optimal_mask.size == 0:
        return None, None, 0
    lambda_mask = _lambda_applicable_mask(frame)
    if lambda_mask.size != optimal_mask.size:
        return None, None, 0
    combined_mask = np.asarray(optimal_mask, dtype=bool) & lambda_mask
    if not np.any(combined_mask):
        return None, None, 0
    lambda_values = np.unique(np.round(frame.loc[combined_mask, "lambda"].to_numpy(dtype=float), 12))
    return float(np.min(lambda_values)), float(np.max(lambda_values)), int(lambda_values.size)


def _canonical_lambda(value: float) -> float:
    return float(np.round(float(value), 12))


def _lambda_warm_start_distance(*, source_lambda: float, target_lambda: float) -> float:
    source = float(source_lambda)
    target = float(target_lambda)
    if source > 0.0 and target > 0.0:
        return float(abs(np.log(source) - np.log(target)))
    if source <= 0.0 and target <= 0.0:
        return 0.0
    if target <= 0.0:
        return float(abs(source - target))
    return float("inf")


def _sorted_unique_lambdas(values: list[float] | np.ndarray) -> list[float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array) & (array >= 0.0)]
    if array.size == 0:
        return []
    return [float(value) for value in np.unique(np.round(np.sort(array), 12))]



def _ari_candidate_frame(search_df: pd.DataFrame) -> pd.DataFrame:
    if search_df.empty or "ARI" not in search_df.columns:
        return search_df.iloc[0:0].copy()
    ari_df = search_df.loc[np.isfinite(search_df["ARI"].to_numpy(dtype=float))].copy()
    return ari_df.sort_values(["lambda", "selection_step"]).reset_index(drop=True)


def _representative_optimal_row(
    tied_df: pd.DataFrame,
    *,
    lambda_min: float | None,
    lambda_max: float | None,
) -> pd.Series:
    if tied_df.empty:
        raise ValueError("tied_df must contain at least one optimal candidate row.")
    if lambda_min is None or lambda_max is None:
        return tied_df.sort_values(
            ["converged", "iterations", "lambda", "selection_step"],
            ascending=[False, False, True, True],
        ).iloc[0]

    if np.isclose(lambda_min, lambda_max, rtol=0.0, atol=1e-12):
        target_lambda = float(lambda_min)
    elif lambda_min <= 0.0 or lambda_max <= 0.0:
        target_lambda = 0.0
    else:
        target_lambda = float(np.sqrt(float(lambda_min) * float(lambda_max)))

    ranked_df = tied_df.copy()
    lambda_values = ranked_df["lambda"].to_numpy(dtype=float)
    if target_lambda > 0.0:
        distances = np.full(lambda_values.shape, np.inf, dtype=float)
        positive_lambda_mask = lambda_values > 0.0
        distances[positive_lambda_mask] = np.abs(
            np.log(lambda_values[positive_lambda_mask]) - np.log(target_lambda)
        )
        ranked_df["_repr_log_distance"] = distances
    else:
        ranked_df["_repr_log_distance"] = np.abs(lambda_values - target_lambda)
    return ranked_df.sort_values(
        ["_repr_log_distance", "converged", "iterations", "lambda", "selection_step"],
        ascending=[True, False, False, True, True],
    ).iloc[0]


def _prefer_fit_candidate(candidate: FitResult, incumbent: FitResult | None) -> bool:
    if incumbent is None:
        return True
    if candidate.selection_eligible and not incumbent.selection_eligible:
        return True
    if candidate.selection_eligible != incumbent.selection_eligible:
        return False
    if candidate.selection_eligible and incumbent.selection_eligible:
        return bool(candidate.penalized_objective < incumbent.penalized_objective - 1e-8)
    candidate_kkt = float(candidate.fixed_objective_kkt_residual)
    incumbent_kkt = float(incumbent.fixed_objective_kkt_residual)
    if np.isfinite(candidate_kkt) and np.isfinite(incumbent_kkt) and abs(candidate_kkt - incumbent_kkt) > 1e-8:
        return bool(candidate_kkt < incumbent_kkt)
    if np.isfinite(candidate_kkt) and not np.isfinite(incumbent_kkt):
        return True
    if not np.isfinite(candidate_kkt) and np.isfinite(incumbent_kkt):
        return False
    return bool(candidate.penalized_objective < incumbent.penalized_objective - 1e-8)




def _effective_bic_partition_tol(options: FitOptions) -> float:
    value = options.bic_partition_tol
    if value is None:
        value = 1e-4
    return float(max(float(value), 1e-12))


def _profile_penalty_from_fit(fit: FitResult) -> tuple[float, float]:
    penalty = max(float(fit.penalized_objective + fit.loglik), 0.0)
    if float(fit.lambda_value) > 0.0:
        return penalty, float(penalty / float(fit.lambda_value))
    return penalty, float("nan")
