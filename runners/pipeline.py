from __future__ import annotations

import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .._version import __version__ as _SOFTWARE_VERSION
from ..core.fusion.profiles import get_computation_profile
from ..core.model import FitOptions
from ..io.tumor_txt import DEFAULT_DOSAGE_PRIOR_PENALTY, load_tumor_txt
from .model_selection import select_model
from ..model_selection.candidates import validate_candidate_identity
from ..model_selection.contracts import get_selection_contract
from ..model_selection.types import DirectPartition, FusionPartition
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
    computation_profile = get_computation_profile(fit_options.computation_profile)
    selection_result = select_model(
        data=data,
        fit_options=fit_options,
        use_warm_starts=use_warm_starts,
    )
    selected_model = selection_result.selected_model
    selected_candidate = getattr(selected_model, "partition_candidate", None)
    if selected_candidate is None:  # Compatibility for lightweight test doubles.
        selected_candidate = selected_model.candidate
    validate_candidate_identity(selected_candidate)
    raw_reference = getattr(selected_model, "raw_reference", selected_candidate)
    validate_candidate_identity(raw_reference)
    best_fit = raw_reference.raw_fit
    raw_partition = raw_reference.partition
    partition_parent_raw = getattr(selected_model, "partition_parent_raw", None)
    partition = selected_candidate.partition
    refit = selected_candidate.refit
    score = selected_candidate.score
    search_df = selection_result.search_df
    elapsed_seconds = float(perf_counter() - start_time)
    selected_lambda = selection_result.selected_lambda_representative
    selection_optimum_resolved = bool(selection_result.selection_optimum_resolved)
    selection_status = (
        "resolved" if selection_optimum_resolved else "provisional_unresolved"
    )
    selection_contract = get_selection_contract(
        getattr(fit_options, "selection_contract", "hybrid-ward-cem-v1")
    )
    selected_partition_certified = bool(
        isinstance(partition, FusionPartition) and partition.certified
    )
    summary: dict[str, float | int | str | bool] = {
        "tumor_id": data.tumor_id,
        "input_file": str(tumor_file),
        "computation_profile": str(computation_profile.name),
        "target_estimator": "complete_graph_pairwise_fusion",
        "solution_mode": (
            "strict_certified"
            if computation_profile.is_strict
            else "approximate_single_tumor_search"
        ),
        "selection_status": selection_status,
        "selection_constraint": "none",
        "selection_contract_id": str(selection_contract.contract_id),
        "selection_contract_json": str(selection_contract.to_json()),
        "selection_optimum_resolved": selection_optimum_resolved,
        "selection_boundary_unresolved": bool(
            selection_result.selection_boundary_unresolved
        ),
        "selection_hits_lower_boundary": bool(
            selection_result.selection_hits_lower_boundary
        ),
        "selection_hits_upper_boundary": bool(
            selection_result.selection_hits_upper_boundary
        ),
        "objective_equivalent_to_strict_graph": bool(
            computation_profile.objective_equivalent_to_strict
        ),
        "scalar_mode": str(computation_profile.scalar_mode),
        "selected_refit_mode": str(refit.refit_mode),
        "selected_lambda": (
            np.nan if selected_lambda is None else float(selected_lambda)
        ),
        "selected_lambda_applicable": bool(selected_lambda is not None),
        "raw_reference_lambda": float(
            getattr(
                best_fit,
                "lambda_value",
                np.nan if selected_lambda is None else selected_lambda,
            )
        ),
        "raw_reference_partition_signature": str(raw_partition.signature),
        "raw_reference_objective_certified": bool(
            getattr(raw_reference, "raw_objective_certified", True)
        ),
        "selected_candidate_family": str(
            getattr(selected_model, "selected_candidate_family", "raw_fusion")
        ),
        "selected_partition_source": str(partition.source),
        "selected_partition_parent_lambda": (
            float(partition.parent_raw_lambda)
            if isinstance(partition, DirectPartition)
            and partition.parent_raw_lambda is not None
            else np.nan
        ),
        "selected_partition_parent_phi_hash": (
            str(partition.parent_raw_phi_hash)
            if isinstance(partition, DirectPartition)
            and partition_parent_raw is not None
            else ""
        ),
        "selected_partition_parent_signature": (
            str(partition_parent_raw.partition.signature)
            if partition_parent_raw is not None
            else ""
        ),
        "selected_n_clusters": int(partition.n_clusters),
        "selected_partition_signature": str(partition.signature),
        "selected_partition_certified": bool(selected_partition_certified),
        "selected_partition_certification_applicable": bool(
            isinstance(partition, FusionPartition)
        ),
        "selected_partition_maximal": bool(
            partition.maximal if isinstance(partition, FusionPartition) else False
        ),
        "selected_direct_partition_identity_certified": bool(
            isinstance(partition, DirectPartition)
            and partition.deterministic_generation
        ),
        "selected_labels_hash": _array_fingerprint(
            partition.labels,
            dtype=np.dtype(np.int64),
        ),
        "raw_reference_phi_hash": _array_fingerprint(
            best_fit.phi,
            dtype=np.dtype(np.float64),
        ),
        "selected_fixed_partition_refit_centers_hash": _array_fingerprint(
            refit.cluster_centers,
            dtype=np.dtype(np.float64),
        ),
        "selection_score_name": str(score.name),
        "selection_score": float(score.value),
        "selection_score_numerical_uncertainty": float(score.numerical_uncertainty),
        "selection_loglik": float(score.loglik),
        "selection_df": int(score.degrees_of_freedom),
        "selection_penalty": float(score.penalty),
        "selection_n_eff": int(score.n_eff),
        "selection_assignment_log_evidence": float(score.assignment_log_evidence),
        "selection_assignment_code_weight": float(score.assignment_code_weight),
        "selection_assignment_penalty": float(score.assignment_penalty),
        "selection_assignment_dirichlet_alpha": float(score.assignment_dirichlet_alpha),
        "selection_assignment_model_id": str(score.assignment_model_id),
        "selection_assignment_symmetry_mode": str(score.assignment_symmetry_mode),
        "selection_assignment_arithmetic_uncertainty": float(
            score.assignment_arithmetic_uncertainty
        ),
        "selected_raw_penalized_objective": float(best_fit.penalized_objective),
        "selected_refit_numerically_resolved": bool(refit.refit_numerically_resolved),
        "selected_refit_global_optimum_certified": bool(refit.global_optimum_certified),
        "selected_refit_global_optimality_gap": float(refit.global_optimality_gap),
        "selected_refit_global_lower_bound": float(refit.global_lower_bound),
        "selected_refit_global_certificate_method": str(
            refit.global_certificate_method
        ),
        "selected_raw_solver_primal_tol": float(fit_options.tol),
        "selected_full_kkt_tolerance": float(best_fit.full_kkt_tolerance),
        "selected_objective_spec_hash": str(best_fit.objective_spec_hash),
        "selected_base_fusion_objective_hash": str(best_fit.base_fusion_objective_hash),
        "selected_original_graph_hash": str(best_fit.original_graph_hash),
        "selection_method": str(selection_result.selection_method),
        "num_candidates": int(selection_result.num_candidates),
        "num_candidates_certified": int(selection_result.num_candidates_certified),
        "ward_candidate_pool_complete": bool(
            getattr(selection_result, "ward_candidate_pool_complete", False)
        ),
        "raw_lambda_path_complete": bool(
            getattr(selection_result, "raw_lambda_path_resolved", False)
        ),
        "global_hybrid_optimum_certified": bool(
            getattr(selection_result, "global_hybrid_optimum_certified", False)
        ),
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
    )
    return summary
