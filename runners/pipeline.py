from __future__ import annotations

import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .._version import __version__ as _SOFTWARE_VERSION
from ..core.bic import uniform_all_blocks_anchor_prior_adjusted_score
from ..core.model import FitOptions, effective_raw_clonal_equality_tolerance
from ..io.tumor_txt import DEFAULT_DOSAGE_PRIOR_PENALTY, load_tumor_txt
from ..model_selection.config import FINAL_PHI_WARD_LADDER_KMAX
from .model_selection import select_model
from ..model_selection.candidates import validate_candidate_identity
from .outputs import write_fit_outputs


def _array_fingerprint(values: np.ndarray, *, dtype: np.dtype) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


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
    selected_model = selection_result.selected_model
    selected_candidate = selected_model.candidate
    validate_candidate_identity(selected_candidate)
    best_fit = selected_candidate.raw_fit
    partition = selected_candidate.partition
    refit = selected_candidate.refit
    score = selected_candidate.score
    clonal_block = selected_candidate.clonal_block
    clonal_block_evidence = selected_candidate.clonal_block_evidence
    search_df = selection_result.search_df
    elapsed_seconds = float(perf_counter() - start_time)
    selected_lambda = selection_result.selected_lambda_representative
    summary: dict[str, float | int | str | bool] = {
        "tumor_id": data.tumor_id,
        "input_file": str(tumor_file),
        "selected_lambda": (
            np.nan if selected_lambda is None else float(selected_lambda)
        ),
        "selected_n_clusters": int(partition.n_clusters),
        "selected_partition_signature": str(partition.signature),
        "selected_partition_certified": bool(partition.certified),
        "selected_partition_maximal": bool(partition.maximal),
        "selected_labels_hash": _array_fingerprint(
            partition.labels,
            dtype=np.dtype(np.int64),
        ),
        "selected_raw_phi_hash": _array_fingerprint(
            best_fit.phi,
            dtype=np.dtype(np.float64),
        ),
        "selected_fixed_partition_refit_phi_hash": _array_fingerprint(
            refit.phi,
            dtype=np.dtype(np.float64),
        ),
        "selected_fixed_partition_refit_centers_hash": _array_fingerprint(
            refit.cluster_centers,
            dtype=np.dtype(np.float64),
        ),
        "selection_score_name": str(score.name),
        "selection_score": float(score.value),
        "uniform_all_blocks_anchor_prior_adjusted_score": float(
            uniform_all_blocks_anchor_prior_adjusted_score(
                score,
                num_clusters=int(partition.n_clusters),
            )
        ),
        "anchor_prior_assumption": "uniform_over_all_partition_blocks",
        "selection_score_numerical_uncertainty": float(
            score.numerical_uncertainty
        ),
        "selection_loglik": float(score.loglik),
        "selection_df": int(score.degrees_of_freedom),
        "selection_penalty": float(score.penalty),
        "selection_n_eff": int(score.n_eff),
        "selected_raw_penalized_objective": float(best_fit.penalized_objective),
        "selected_refit_numerically_resolved": bool(
            refit.refit_numerically_resolved
        ),
        "selected_refit_global_optimum_certified": bool(
            refit.global_optimum_certified
        ),
        "selected_refit_global_optimality_gap": float(
            refit.global_optimality_gap
        ),
        "selected_refit_global_lower_bound": float(refit.global_lower_bound),
        "selected_refit_global_certificate_method": str(
            refit.global_certificate_method
        ),
        "selected_raw_clonal_anchor_mutation_index": (
            -1
            if best_fit.raw_clonal_anchor_mutation_index is None
            else int(best_fit.raw_clonal_anchor_mutation_index)
        ),
        "selected_raw_clonal_anchor_mutation_id": (
            "none"
            if best_fit.raw_clonal_anchor_mutation_index is None
            else str(data.mutation_ids[int(best_fit.raw_clonal_anchor_mutation_index)])
        ),
        "selected_raw_clonal_anchor_cluster": (
            -1 if refit.clonal_cluster is None else int(refit.clonal_cluster)
        ),
        "selected_raw_clonal_cluster_signature": (
            "none" if clonal_block is None else str(clonal_block.block_signature)
        ),
        "selected_raw_clonal_cluster_size": (
            0 if clonal_block is None else int(clonal_block.cluster_size)
        ),
        "selected_raw_clonal_cluster_mathematically_certified": bool(
            clonal_block is not None and clonal_block.mathematically_certified
        ),
        "raw_clonal_model_fitted": bool(
            clonal_block is not None and clonal_block.mathematically_certified
        ),
        "clonal_constraint_satisfied": bool(
            clonal_block is not None and clonal_block.mathematically_certified
        ),
        "selected_raw_clonal_cluster_equality_tol": float(
            effective_raw_clonal_equality_tolerance(fit_options)
        ),
        "selected_raw_solver_primal_tol": float(fit_options.tol),
        "selected_full_kkt_tolerance": float(best_fit.full_kkt_tolerance),
        "selected_raw_clonal_cluster_centroid_residual": (
            np.nan if clonal_block is None else float(clonal_block.centroid_residual)
        ),
        "selected_raw_clonal_cluster_max_member_residual": (
            np.nan
            if clonal_block is None
            else float(clonal_block.maximum_member_residual)
        ),
        "selected_raw_clonal_cluster_observed_support_per_region": (
            "none"
            if clonal_block_evidence is None
            else ",".join(
                str(int(value))
                for value in clonal_block_evidence.observed_support_per_region
            )
        ),
        "selected_raw_clonal_cluster_evidence_supported": bool(
            clonal_block_evidence is not None
            and clonal_block_evidence.evidence_gate_passed
        ),
        "clonal_block_biologically_supported": bool(
            clonal_block_evidence is not None
            and clonal_block_evidence.evidence_gate_passed
        ),
        "clonal_cluster_statistically_identified": bool(
            clonal_block_evidence is not None
            and clonal_block_evidence.evidence_gate_passed
        ),
        "clonal_cluster_statistical_identification_reason": (
            "insufficient_support"
            if clonal_block_evidence is None
            or not clonal_block_evidence.evidence_gate_passed
            else "support_thresholds_satisfied"
        ),
        "selected_raw_clonal_cluster_evidence_failure_reason": (
            "none"
            if clonal_block_evidence is None
            else str(clonal_block_evidence.evidence_failure_reason)
        ),
        "selected_raw_clonal_cluster_total_depth_per_region": (
            "none"
            if clonal_block_evidence is None
            else ",".join(
                format(float(value), ".17g")
                for value in clonal_block_evidence.total_depth_per_region
            )
        ),
        "selected_raw_clonal_cluster_median_depth_per_region": (
            "none"
            if clonal_block_evidence is None
            else ",".join(
                format(float(value), ".17g")
                for value in clonal_block_evidence.median_depth_per_region
            )
        ),
        "selected_raw_clonal_cluster_common_center": (
            "none"
            if clonal_block is None
            else ",".join(
                format(float(value), ".17g")
                for value in clonal_block.common_center
            )
        ),
        "selected_raw_clonal_cluster_centroid": (
            "none"
            if clonal_block is None
            else ",".join(
                format(float(value), ".17g") for value in clonal_block.centroid
            )
        ),
        "selected_raw_clonal_witness_mutation_id": (
            "none"
            if clonal_block is None
            else str(clonal_block.witness_mutation_id)
        ),
        "selected_raw_clonal_anchor_target": (
            "none"
            if best_fit.raw_clonal_anchor_target is None
            else ",".join(
                format(float(value), ".17g")
                for value in np.asarray(best_fit.raw_clonal_anchor_target).reshape(-1)
            )
        ),
        "selected_raw_clonal_anchor_source": str(
            best_fit.raw_clonal_anchor_source
        ),
        "selected_raw_clonal_anchor_mode": str(best_fit.raw_clonal_anchor_mode),
        "selected_raw_clonal_anchor_constraint_residual": float(
            best_fit.raw_clonal_anchor_constraint_residual
        ),
        "selected_raw_clonal_anchor_frozen_coordinate_count": int(
            best_fit.raw_clonal_anchor_frozen_coordinate_count
        ),
        "selected_raw_clonal_anchor_frozen_mutation_count": int(
            len(best_fit.raw_clonal_anchor_frozen_mutation_indices)
        ),
        "selected_raw_clonal_anchor_search_complete": bool(
            best_fit.raw_clonal_anchor_search_complete
        ),
        "selected_raw_clonal_witness_coverage_certified": bool(
            best_fit.raw_clonal_witness_coverage_certified
        ),
        "selected_raw_clonal_branch_stationarity_certified": bool(
            best_fit.raw_clonal_branch_stationarity_certified
        ),
        "selected_raw_clonal_union_global_optimum_certified": bool(
            best_fit.raw_clonal_union_global_optimum_certified
        ),
        "selected_raw_clonal_anchor_total_eligible_candidates": int(
            best_fit.raw_clonal_anchor_total_eligible_candidates
        ),
        "selected_raw_clonal_anchor_candidates_evaluated": int(
            best_fit.raw_clonal_anchor_candidates_evaluated
        ),
        "selected_raw_clonal_anchor_objective_gap_to_second": float(
            best_fit.raw_clonal_anchor_objective_gap_to_second
        ),
        "selected_raw_clonal_anchor_screening_rule": str(
            best_fit.raw_clonal_anchor_screening_rule
        ),
        "selected_anchor_block_signature": str(score.anchor_block_signature),
        "selected_objective_spec_hash": str(best_fit.objective_spec_hash),
        "selected_base_fusion_objective_hash": str(
            best_fit.base_fusion_objective_hash
        ),
        "selected_raw_clonal_union_model_hash": str(
            best_fit.raw_clonal_union_model_hash
        ),
        "selected_witness_subproblem_hash": str(
            best_fit.witness_subproblem_hash
        ),
        "selected_original_graph_hash": str(best_fit.original_graph_hash),
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
        validate_candidate_identity(selected_candidate)
        write_fit_outputs(
            outdir=outdir,
            data=data,
            raw_fit=best_fit,
            partition=partition,
            refit=refit,
            clonal_block=clonal_block,
            clonal_block_evidence=clonal_block_evidence,
            major_prior=float(fit_options.major_prior),
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
