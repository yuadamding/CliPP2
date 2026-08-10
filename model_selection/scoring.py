from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.model import FitOptions, FitResult
from ..io.data import TumorData
from ..core.bic import (
    compute_classic_bic,
    compute_extended_bic,
    compute_partition_icl,
)
from .config import PARTITION_ICL_DIRICHLET_ALPHA, SELECTION_SCORE_NAMES


_EXACT_OBSERVED_OBJECTIVE_GRADIENT_SCOPES = frozenset(
    {
        "observed_objective",
        "clarke_piecewise_observed_objective_subgradient",
    }
)


def _normalize_selection_score_name(selection_score: str) -> str:
    normalized = str(selection_score).strip().lower()
    if normalized == "ebic":
        normalized = "extended_bic"
    if normalized in SELECTION_SCORE_NAMES:
        return normalized
    allowed = ", ".join(SELECTION_SCORE_NAMES)
    raise ValueError(
        f"Unknown selection_score: {selection_score}. Expected one of: {allowed}."
    )


def _selection_score_value(
    *,
    loglik: float,
    num_clusters: int,
    data: TumorData,
    bic_df_scale: float,
    bic_cluster_penalty: float,
    selection_score: str,
    cluster_sizes: np.ndarray | None = None,
    partition_icl_alpha: float = PARTITION_ICL_DIRICHLET_ALPHA,
    marginal_loglik: float | None = None,
) -> tuple[float, float, float, float]:
    classic_bic = compute_classic_bic(loglik, num_clusters, data)
    extended_bic = compute_extended_bic(
        loglik,
        num_clusters,
        data,
        bic_df_scale=bic_df_scale,
        bic_cluster_penalty=bic_cluster_penalty,
    )
    partition_icl = (
        float("nan")
        if cluster_sizes is None
        else compute_partition_icl(
            loglik,
            cluster_sizes,
            data,
            alpha=float(partition_icl_alpha),
        )
    )
    normalized = _normalize_selection_score_name(selection_score)
    if normalized == "bic":
        selected_score = classic_bic
    elif normalized == "extended_bic":
        selected_score = extended_bic
    elif normalized == "marginal_bic":
        if marginal_loglik is None or not np.isfinite(float(marginal_loglik)):
            raise ValueError(
                "marginal_bic selection requires a finite marginal "
                "mixture log-likelihood."
            )
        # Same (K-1)*S*log(n_eff) penalty as classic BIC; only the likelihood
        # differs (assignments integrated out at the CEM-stabilized centers).
        selected_score = compute_classic_bic(
            float(marginal_loglik), num_clusters, data
        )
    elif normalized == "partition_icl":
        if not np.isfinite(partition_icl):
            raise ValueError(
                "partition_icl selection requires candidate cluster sizes or labels."
            )
        selected_score = partition_icl
    else:
        raise ValueError(f"Unknown normalized selection_score: {selection_score}")
    return (
        float(selected_score),
        float(classic_bic),
        float(extended_bic),
        float(partition_icl),
    )


def _is_bic_selection_eligible(
    *,
    raw_kkt_eligible: bool,
    classic_bic: float,
    selection_score_value: float | None = None,
    bic_refit_finite_candidate_found: bool | None = None,
    bic_refit_converged: bool | None = None,
) -> bool:
    refit_finite = (
        _bool_with_default(bic_refit_finite_candidate_found, default=False)
        if bic_refit_finite_candidate_found is not None
        else _bool_with_default(bic_refit_converged, default=False)
    )
    selected_score_finite = (
        np.isfinite(float(classic_bic))
        if selection_score_value is None
        else np.isfinite(float(selection_score_value))
    )
    return bool(
        _bool_with_default(raw_kkt_eligible, default=False)
        and refit_finite
        and np.isfinite(float(classic_bic))
        and selected_score_finite
    )


def _bic_selection_eligible_mask(search_df: pd.DataFrame) -> np.ndarray:
    n_rows = int(search_df.shape[0])
    if "bic_selection_eligible" in search_df.columns:
        return _strict_bool_mask(search_df["bic_selection_eligible"])
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
        return (
            _add_bic_selection_eligible(search_df)["bic_selection_eligible"]
            .astype(bool)
            .to_numpy(dtype=bool)
        )
    if "selection_eligible" in search_df.columns:
        return _strict_bool_mask(search_df["selection_eligible"])
    if "converged" in search_df.columns:
        return _strict_bool_mask(search_df["converged"])
    return np.zeros(n_rows, dtype=bool)


def _false_mask(search_df: pd.DataFrame) -> np.ndarray:
    return np.zeros(search_df.shape[0], dtype=bool)


def _required_bool_mask(search_df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in search_df.columns:
        return _false_mask(search_df)
    return _strict_bool_mask(search_df[column])


def _required_text_mask(
    search_df: pd.DataFrame,
    column: str,
    expected: str | None = None,
) -> np.ndarray:
    if column not in search_df.columns:
        return _false_mask(search_df)

    def is_valid(value: object) -> bool:
        try:
            if pd.isna(value):
                return False
        except (TypeError, ValueError):
            return False
        normalized = str(value).strip()
        if not normalized:
            return False
        return expected is None or normalized == expected

    return search_df[column].map(is_valid).to_numpy(dtype=bool)


def _required_text_membership_mask(
    search_df: pd.DataFrame,
    column: str,
    accepted: frozenset[str],
) -> np.ndarray:
    if column not in search_df.columns:
        return _false_mask(search_df)
    return (
        search_df[column]
        .map(lambda value: str(value).strip() in accepted)
        .to_numpy(dtype=bool)
    )


def _exact_fusion_certificate_mask(search_df: pd.DataFrame) -> np.ndarray:
    """Return rows carrying an accepted full fixed-objective certificate.

    Versioned provenance is authoritative and deliberately independent of the
    inner backend. Rows written before provenance schema v1 retain the previous
    dense-ADMM rule. A present but invalid/unsupported schema value fails closed
    rather than falling back to solver identity.
    """

    if search_df.empty:
        return _false_mask(search_df)

    raw_kkt_ok = _required_bool_mask(search_df, "raw_kkt_eligible")
    if "exactness_provenance_version" in search_df.columns:
        schema_values = search_df["exactness_provenance_version"]
        schema_present = schema_values.notna().to_numpy(dtype=bool)
        schema_version = pd.to_numeric(schema_values, errors="coerce").to_numpy(
            dtype=float
        )
    else:
        schema_present = _false_mask(search_df)
        schema_version = np.full(search_df.shape[0], np.nan, dtype=float)

    explicit = (
        schema_present
        & np.isfinite(schema_version)
        & (schema_version == 1.0)
        & raw_kkt_ok
        & _required_text_mask(search_df, "estimator_role", "raw_fused_lambda_path")
        & _required_bool_mask(search_df, "objective_faithful")
        & _required_text_mask(search_df, "objective_spec_hash")
        & _required_text_mask(search_df, "original_graph_hash")
        & _required_text_mask(search_df, "certificate_problem_hash")
        & _required_text_mask(search_df, "certificate_scope", "full_original_graph")
        & _required_text_membership_mask(
            search_df,
            "certificate_gradient_scope",
            _EXACT_OBSERVED_OBJECTIVE_GRADIENT_SCOPES,
        )
        & _required_bool_mask(search_df, "full_kkt_certified")
        & _required_text_membership_mask(
            search_df,
            "full_kkt_certificate_status",
            frozenset(
                {
                    "certified",
                    "input_dual_retained",
                    "analytic_nonfused_dual",
                    "refined_fused_edge_dual",
                    "zero_penalty_no_dual_needed",
                }
            ),
        )
    )

    if "fixed_objective_kkt_residual" in search_df.columns:
        residual = pd.to_numeric(
            search_df["fixed_objective_kkt_residual"], errors="coerce"
        ).to_numpy(dtype=float)
    else:
        residual = np.full(search_df.shape[0], np.nan, dtype=float)
    if "full_kkt_tolerance" in search_df.columns:
        tolerance = pd.to_numeric(
            search_df["full_kkt_tolerance"], errors="coerce"
        ).to_numpy(dtype=float)
    else:
        tolerance = np.full(search_df.shape[0], np.nan, dtype=float)
    explicit &= (
        np.isfinite(residual)
        & np.isfinite(tolerance)
        & (tolerance > 0.0)
        & (residual <= tolerance)
    )

    legacy = ~schema_present & raw_kkt_ok
    if (
        "inner_solver" not in search_df.columns
        or "admm_iterations" not in search_df.columns
    ):
        legacy &= False
    else:
        solver_ok = (
            search_df["inner_solver"]
            .astype(str)
            .eq("admm_complete_graph")
            .to_numpy(dtype=bool)
        )
        admm_iterations = pd.to_numeric(
            search_df["admm_iterations"], errors="coerce"
        ).to_numpy(dtype=float)
        legacy &= solver_ok & np.isfinite(admm_iterations) & (admm_iterations > 0.0)
    return explicit | legacy


def _positive_exact_fusion_selection_mask(search_df: pd.DataFrame) -> np.ndarray:
    """Strict backend-neutral contract for a final positive-fusion estimator."""

    eligible = _bic_selection_eligible_mask(search_df)
    if search_df.empty:
        return eligible
    if "candidate_pool_source" not in search_df.columns:
        return _false_mask(search_df)
    source_ok = (
        search_df["candidate_pool_source"]
        .astype(str)
        .eq("raw_fused_lambda_path")
        .to_numpy(dtype=bool)
    )
    if "lambda" not in search_df.columns:
        return _false_mask(search_df)
    lambdas = pd.to_numeric(search_df["lambda"], errors="coerce").to_numpy(dtype=float)
    lambda_ok = np.isfinite(lambdas) & (lambdas > 0.0)
    return eligible & source_ok & lambda_ok & _exact_fusion_certificate_mask(search_df)


def _row_bic_selection_eligible(row: pd.Series) -> bool:
    value = row.get(
        "bic_selection_eligible",
        row.get("selection_eligible", row.get("converged", False)),
    )
    return _bool_with_default(value, default=False)


def _bool_with_default(value: object, default: bool = False) -> bool:
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
        return bool(default)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer, float, np.floating)):
        numeric = float(value)
        if numeric == 0.0:
            return False
        if numeric == 1.0:
            return True
    return bool(default)


def _strict_bool_mask(values: pd.Series) -> np.ndarray:
    return values.map(lambda value: _bool_with_default(value, default=False)).to_numpy(
        dtype=bool, copy=True
    )


def _row_lambda_applicable(row: pd.Series) -> bool:
    if "lambda_applicable" not in row:
        return True
    return _bool_with_default(row["lambda_applicable"], default=False)


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
        mask = _strict_bool_mask(frame["lambda_applicable"])
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
        raw_kkt = _strict_bool_mask(enriched["raw_kkt_eligible"])
    elif "selection_eligible" in enriched.columns:
        raw_kkt = _strict_bool_mask(enriched["selection_eligible"])
    elif "converged" in enriched.columns:
        raw_kkt = _strict_bool_mask(enriched["converged"])
    else:
        raw_kkt = np.zeros(n_rows, dtype=bool)
    if "candidate_pool_source" in enriched.columns:
        candidate_source = enriched["candidate_pool_source"].astype(str)
        partition_candidate = candidate_source.eq("likelihood_partition").to_numpy(
            dtype=bool
        )
    elif "search_phase" in enriched.columns:
        partition_candidate = (
            enriched["search_phase"]
            .astype(str)
            .eq("likelihood_partition")
            .to_numpy(dtype=bool)
        )
    else:
        partition_candidate = np.zeros(n_rows, dtype=bool)
    if "bic_refit_finite_candidate_found" in enriched.columns:
        bic_refit = _strict_bool_mask(enriched["bic_refit_finite_candidate_found"])
    elif "bic_refit_converged" in enriched.columns:
        bic_refit = _strict_bool_mask(enriched["bic_refit_converged"])
    else:
        # Absent certificate means unknown, treated as False (not True)
        bic_refit = np.zeros(n_rows, dtype=bool)
    if "classic_bic" in enriched.columns:
        classic_bic = enriched["classic_bic"].to_numpy(dtype=float)
    elif "bic" in enriched.columns:
        classic_bic = enriched["bic"].to_numpy(dtype=float)
    else:
        classic_bic = np.full(n_rows, np.nan, dtype=float)
    if "bic" in enriched.columns:
        selected_score = enriched["bic"].to_numpy(dtype=float)
    else:
        selected_score = classic_bic
    enriched["bic_selection_eligible"] = (
        (raw_kkt | partition_candidate)
        & bic_refit
        & np.isfinite(classic_bic)
        & np.isfinite(selected_score)
    )
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
    one_cluster = enriched.loc[
        (n_clusters == 1.0) & np.isfinite(enriched["classic_bic"].to_numpy(dtype=float))
    ].copy()
    if one_cluster.empty:
        return enriched
    one_cluster["_bic_eligible_for_baseline"] = _bic_selection_eligible_mask(
        one_cluster
    )
    baseline = one_cluster.sort_values(
        ["_bic_eligible_for_baseline", "classic_bic", "lambda", "selection_step"],
        ascending=[False, True, True, True],
    ).iloc[0]
    baseline_loglik = float(baseline.get("bic_loglik", np.nan))
    baseline_bic = float(baseline.get("classic_bic", np.nan))
    if np.isfinite(baseline_loglik):
        enriched["delta_loglik_vs_one_cluster"] = (
            enriched["bic_loglik"].to_numpy(dtype=float) - baseline_loglik
        )
    if np.isfinite(baseline_bic):
        enriched["delta_bic_vs_one_cluster"] = (
            enriched["classic_bic"].to_numpy(dtype=float) - baseline_bic
        )
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
    lambda_values = np.unique(
        np.round(lambdas[optimal_mask].astype(float, copy=False), 12)
    )
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
    lambda_values = np.unique(
        np.round(frame.loc[combined_mask, "lambda"].to_numpy(dtype=float), 12)
    )
    return (
        float(np.min(lambda_values)),
        float(np.max(lambda_values)),
        int(lambda_values.size),
    )


def _canonical_lambda(value: float) -> float:
    return float(np.round(float(value), 12))


def _sorted_unique_lambdas(values: list[float] | np.ndarray) -> list[float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array) & (array >= 0.0)]
    if array.size == 0:
        return []
    return [float(value) for value in np.unique(np.round(np.sort(array), 12))]


def _prefer_fit_candidate(candidate: FitResult, incumbent: FitResult | None) -> bool:
    if incumbent is None:
        return True
    if candidate.selection_eligible and not incumbent.selection_eligible:
        return True
    if candidate.selection_eligible != incumbent.selection_eligible:
        return False
    candidate_objective = float(candidate.penalized_objective)
    incumbent_objective = float(incumbent.penalized_objective)
    if (
        np.isfinite(candidate_objective)
        and np.isfinite(incumbent_objective)
        and abs(candidate_objective - incumbent_objective) > 1e-8
    ):
        return bool(candidate_objective < incumbent_objective)
    if np.isfinite(candidate_objective) and not np.isfinite(incumbent_objective):
        return True
    if not np.isfinite(candidate_objective) and np.isfinite(incumbent_objective):
        return False
    candidate_kkt = float(candidate.fixed_objective_kkt_residual)
    incumbent_kkt = float(incumbent.fixed_objective_kkt_residual)
    if (
        np.isfinite(candidate_kkt)
        and np.isfinite(incumbent_kkt)
        and abs(candidate_kkt - incumbent_kkt) > 1e-8
    ):
        return bool(candidate_kkt < incumbent_kkt)
    if np.isfinite(candidate_kkt) and not np.isfinite(incumbent_kkt):
        return True
    return False


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

def _adaptive_score_column(normalized_score: str) -> str:
    if normalized_score == "bic":
        return "classic_bic"
    if normalized_score == "extended_bic":
        return "extended_bic"
    if normalized_score == "partition_icl":
        return "partition_icl"
    if normalized_score == "marginal_bic":
        return "marginal_bic"
    raise ValueError(f"Unknown normalized selection score: {normalized_score}")

def _score_strictly_better(score: float, reference: float) -> bool:
    if not np.isfinite(score) or not np.isfinite(reference):
        return False
    margin = 1e-8 * (1.0 + abs(float(reference)))
    return bool(float(score) < float(reference) - margin)

def _best_candidate_rows_by_lambda(
    search_df: pd.DataFrame,
    *,
    normalized_score: str = "bic",
) -> pd.DataFrame:
    if search_df.empty:
        return search_df.copy()
    ranked = _add_bic_selection_eligible(search_df)
    if (
        "bic_refit_finite_candidate_found" not in ranked.columns
        and "bic_refit_converged" in ranked.columns
    ):
        ranked["bic_refit_finite_candidate_found"] = ranked[
            "bic_refit_converged"
        ].astype(bool)
    if (
        "bic_refit_converged" not in ranked.columns
        and "bic_refit_finite_candidate_found" in ranked.columns
    ):
        ranked["bic_refit_converged"] = ranked[
            "bic_refit_finite_candidate_found"
        ].astype(bool)
    ranked["_lambda_key"] = np.round(ranked["lambda"].to_numpy(dtype=float), 12)
    score_column = _adaptive_score_column(normalized_score)
    defaults = {
        "selection_eligible": False,
        "bic_refit_finite_candidate_found": False,
        "bic_refit_converged": False,
        "converged": False,
        score_column: np.inf,
        "penalized_objective": np.inf,
        "selection_step": np.arange(ranked.shape[0], dtype=int),
    }
    for column, default in defaults.items():
        if column not in ranked.columns:
            ranked[column] = default
    sort_columns = [
        "_lambda_key",
        "bic_selection_eligible",
        "selection_eligible",
        "bic_refit_finite_candidate_found",
        "bic_refit_converged",
        score_column,
        "penalized_objective",
        "selection_step",
    ]
    ascending = [True, False, False, False, False, True, True, True]
    ranked = ranked.sort_values(sort_columns, ascending=ascending)
    return (
        ranked.drop_duplicates("_lambda_key", keep="first")
        .drop(columns=["_lambda_key"])
        .reset_index(drop=True)
    )

def _selected_lambda_signature_interval(
    search_df: pd.DataFrame,
    *,
    selected_candidate_id: int,
    normalized_score: str = "bic",
) -> tuple[float | None, float | None, float | None]:
    if search_df.empty or "_candidate_id" not in search_df.columns:
        return None, None, None
    selected = search_df.loc[
        search_df["_candidate_id"].astype(int) == int(selected_candidate_id)
    ]
    if selected.empty:
        return None, None, None
    selected_row = selected.iloc[0]
    if "lambda_applicable" in selected_row and not bool(
        selected_row.get("lambda_applicable", True)
    ):
        return None, None, None
    selected_lambda = float(selected_row["lambda"])
    signature = str(selected_row.get("partition_signature", ""))
    eligible = search_df.loc[_bic_selection_eligible_mask(search_df)].copy()
    if eligible.empty:
        eligible = search_df.copy()
    eligible = (
        _best_candidate_rows_by_lambda(
            eligible,
            normalized_score=normalized_score,
        )
        .sort_values("lambda")
        .reset_index(drop=True)
    )
    lambdas = eligible["lambda"].to_numpy(dtype=float)
    signatures = (
        eligible["partition_signature"].astype(str).to_numpy()
        if "partition_signature" in eligible.columns
        else np.full(eligible.shape[0], signature)
    )
    if lambdas.size == 0:
        return selected_lambda, selected_lambda, 0.0
    if selected_lambda > 0.0:
        distances = np.full(lambdas.shape, np.inf, dtype=float)
        positive_lambda_mask = lambdas > 0.0
        distances[positive_lambda_mask] = np.abs(
            np.log(lambdas[positive_lambda_mask]) - np.log(selected_lambda)
        )
        if not np.any(np.isfinite(distances)):
            distances = np.abs(lambdas - selected_lambda)
    else:
        distances = np.abs(lambdas - selected_lambda)
    idx = int(np.argmin(distances))
    left_idx = idx
    while left_idx > 0 and signatures[left_idx - 1] == signature:
        left_idx -= 1
    right_idx = idx
    while right_idx + 1 < lambdas.size and signatures[right_idx + 1] == signature:
        right_idx += 1
    left_lambda = float(lambdas[left_idx])
    right_lambda = float(lambdas[right_idx])
    log_width = (
        float(np.log10(right_lambda) - np.log10(left_lambda))
        if right_lambda > 0.0 and left_lambda > 0.0
        else 0.0
    )
    return left_lambda, right_lambda, log_width

def _lambda_boundary_flags(
    evaluated_lambdas: list[float],
    *,
    best_lambda_min: float | None,
    best_lambda_max: float | None,
) -> tuple[bool, bool]:
    sorted_lambdas = _sorted_unique_lambdas(evaluated_lambdas)
    if not sorted_lambdas or best_lambda_min is None or best_lambda_max is None:
        return False, False
    lower_hit = np.isclose(best_lambda_min, sorted_lambdas[0], rtol=0.0, atol=1e-12)
    upper_hit = np.isclose(best_lambda_max, sorted_lambdas[-1], rtol=0.0, atol=1e-12)
    return bool(lower_hit), bool(upper_hit)

def _lambda_boundary_unresolved(
    *,
    evaluated_lambdas: list[float],
    lower_hit: bool,
    upper_hit: bool,
) -> bool:
    sorted_lambdas = _sorted_unique_lambdas(evaluated_lambdas)
    if not sorted_lambdas:
        return False
    return bool(lower_hit or upper_hit)
