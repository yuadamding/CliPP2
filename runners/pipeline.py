from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .._version import __version__ as _SOFTWARE_VERSION
from ..core.fusion.partition_starts import _loss_to_centers
from ..core.fusion.refit import (
    _canonical_labels,
    partition_constrained_observed_refit,
)
from ..core.fusion.solver import _cluster_summary_from_labels
from ..core.model import FitOptions, FitResult
from ..io.tumor_txt import DEFAULT_DOSAGE_PRIOR_PENALTY, load_tumor_txt
from ..model_selection.config import FINAL_PHI_WARD_LADDER_KMAX
from .model_selection import select_model
from .outputs import write_fit_outputs


def _cluster_diameters_dense(
    phi: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Exact per-cluster L2 diameters, chunked so memory stays O(block * M)."""
    n_clusters = int(labels.max()) + 1 if labels.size else 0
    diameters = np.zeros(n_clusters, dtype=np.float64)
    for cluster in range(n_clusters):
        rows = np.asarray(phi[labels == cluster], dtype=np.float64)
        best = 0.0
        for start in range(0, rows.shape[0], 512):
            block = rows[start : start + 512]
            distances = np.linalg.norm(block[:, None, :] - rows[None, :, :], axis=-1)
            if distances.size:
                best = max(best, float(distances.max()))
        diameters[cluster] = best
    return diameters, True


def _polish_selected_labels(
    data,
    fit: FitResult,
    bic_refit_phi: np.ndarray | None,
    bic_refit_cluster_centers: np.ndarray | None,
    fit_options: FitOptions,
) -> tuple[FitResult, np.ndarray | None, np.ndarray | None]:
    """One hard E-step against the anchored refit's assignment cost, then a
    final anchored refit, applied to the selected partition before outputs.

    Only boundary mutations whose observed reads fit another cluster's center
    strictly better are moved (~2% of mutations corpus-wide), which repairs
    the absorption of near-clonal mutations into the pinned clonal cluster.
    K is held fixed — a step that would empty a cluster is rejected — so model
    selection is untouched. Any failure falls back to the unpolished fit.
    """
    labels = np.asarray(fit.cluster_labels, dtype=np.int64).reshape(-1)
    if labels.size == 0 or int(fit.n_clusters) < 2:
        return fit, bic_refit_phi, bic_refit_cluster_centers
    if not np.array_equal(_canonical_labels(labels), labels):
        # Refit centers are indexed in canonical label order; with any other
        # numbering they would not align with the output cluster_label order.
        return fit, bic_refit_phi, bic_refit_cluster_centers
    try:
        refit_kwargs = dict(
            major_prior=float(fit_options.major_prior),
            eps=float(fit_options.eps),
            tol=float(fit_options.tol),
            max_iter=max(int(fit_options.inner_max_iter), 32),
        )
        hint_phi = np.asarray(fit.phi, dtype=np.float64)
        base = partition_constrained_observed_refit(
            data, labels, hint_phi=hint_phi, **refit_kwargs
        )
        cost = _loss_to_centers(
            data,
            np.asarray(base.cluster_centers, dtype=np.float64),
            major_prior=float(fit_options.major_prior),
            eps=float(fit_options.eps),
        )
        proposed = np.argmin(cost, axis=1).astype(np.int64)
        if (
            np.array_equal(proposed, labels)
            or np.unique(proposed).size != int(fit.n_clusters)
        ):
            return fit, bic_refit_phi, bic_refit_cluster_centers
        proposed = _canonical_labels(proposed)
        polished = partition_constrained_observed_refit(
            data, proposed, hint_phi=hint_phi, **refit_kwargs
        )
    except Exception:
        return fit, bic_refit_phi, bic_refit_cluster_centers
    phi = np.asarray(fit.phi, dtype=np.float64)
    centers, phi_clustered = _cluster_summary_from_labels(phi, proposed)
    diameters, diameter_exact = _cluster_diameters_dense(phi, proposed)
    fit = replace(
        fit,
        cluster_labels=proposed,
        cluster_centers=centers.astype(np.asarray(fit.cluster_centers).dtype, copy=False),
        phi_clustered=phi_clustered.astype(np.asarray(fit.phi_clustered).dtype, copy=False),
        cluster_diameters=diameters,
        max_cluster_diameter=float(np.max(diameters)) if diameters.size else 0.0,
        cluster_diameter_exact=bool(diameter_exact),
    )
    return (
        fit,
        np.asarray(polished.phi, dtype=np.float64),
        np.asarray(polished.cluster_centers, dtype=np.float64),
    )


def process_tumor_bundle(
    tumor_file: str | Path,
    outdir: str | Path,
    fit_options: FitOptions | None = None,
    use_warm_starts: bool = True,
    write_outputs: bool = True,
    unsupported_policy: str = "error",
    dosage_prior_penalty: float = DEFAULT_DOSAGE_PRIOR_PENALTY,
    ward_ladder_kmax: int = FINAL_PHI_WARD_LADDER_KMAX,
) -> tuple[dict[str, float | int | str | bool], pd.DataFrame]:
    """Fit one canonical tumor TSV file with the default workflow."""

    start_time = perf_counter()
    tumor_file = Path(tumor_file)
    outdir = Path(outdir)
    if not tumor_file.is_file():
        raise FileNotFoundError(f"Tumor input must be a file: {tumor_file}")
    data = load_tumor_txt(
        tumor_file,
        unsupported_policy=unsupported_policy,
        dosage_prior_penalty=dosage_prior_penalty,
    )

    if fit_options is None:
        fit_options = FitOptions(lambda_value=0.0)
    selection_result = select_model(
        data=data,
        fit_options=fit_options,
        use_warm_starts=use_warm_starts,
        ward_ladder_kmax=int(ward_ladder_kmax),
    )
    best_fit = selection_result.best_fit
    selection_artifact = selection_result.selected_artifact
    search_df = selection_result.search_df
    elapsed_seconds = float(perf_counter() - start_time)
    selected_lambda = selection_result.selected_lambda_representative
    summary: dict[str, float | int | str | bool] = {
        "tumor_id": data.tumor_id,
        "input_file": str(tumor_file),
        "selected_lambda": (
            np.nan if selected_lambda is None else float(selected_lambda)
        ),
        "selected_n_clusters": int(best_fit.n_clusters),
        "selection_score_name": str(
            selection_artifact.selection_score_name or "marginal_bic"
        ),
        "selection_metric_value": (
            np.nan
            if selection_result.selection_metric_value is None
            else float(selection_result.selection_metric_value)
        ),
        "selection_method": str(selection_result.selection_method),
        "num_candidates": int(selection_result.num_candidates),
        "num_candidates_certified": int(selection_result.num_candidates_certified),
        "selected_kkt_residual": (
            np.nan
            if selection_result.selected_kkt_residual is None
            else float(selection_result.selected_kkt_residual)
        ),
        "search_stop_reason": str(selection_result.adaptive_search_stop_reason),
        "device": str(best_fit.device),
        "dtype": str(best_fit.dtype),
        "elapsed_seconds": elapsed_seconds,
        "software_version": _SOFTWARE_VERSION,
    }

    if write_outputs:
        polished_fit, polished_refit_phi, polished_refit_centers = (
            _polish_selected_labels(
                data,
                best_fit,
                selection_artifact.bic_refit_phi,
                selection_artifact.bic_refit_cluster_centers,
                fit_options,
            )
        )
        write_fit_outputs(
            outdir=outdir,
            data=data,
            fit=polished_fit,
            bic_refit_phi=polished_refit_phi,
            bic_refit_cluster_centers=polished_refit_centers,
        )
    return summary, search_df


def process_tumor(
    tumor_file: str | Path,
    outdir: str | Path,
    fit_options: FitOptions | None = None,
    use_warm_starts: bool = True,
    write_outputs: bool = True,
    unsupported_policy: str = "error",
    dosage_prior_penalty: float = DEFAULT_DOSAGE_PRIOR_PENALTY,
    ward_ladder_kmax: int = FINAL_PHI_WARD_LADDER_KMAX,
) -> dict[str, float | int | str | bool]:
    """Fit one tumor TSV file."""

    summary, _ = process_tumor_bundle(
        tumor_file=tumor_file,
        outdir=outdir,
        fit_options=fit_options,
        use_warm_starts=use_warm_starts,
        write_outputs=write_outputs,
        unsupported_policy=unsupported_policy,
        dosage_prior_penalty=dosage_prior_penalty,
        ward_ladder_kmax=int(ward_ladder_kmax),
    )
    return summary
