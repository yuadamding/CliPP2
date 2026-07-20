from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

import CliPP2.core.fusion.solver as solver_module
from CliPP2.core.fusion.solver import fit_torch, prepare_torch_problem
from CliPP2.core.fusion.starts import (
    compute_pooled_observed_data_start,
    compute_pooled_observed_data_start_torch,
    compute_scalar_mutation_region_wells,
    compute_scalar_mutation_region_wells_torch,
    compute_scalar_well_start_bank,
    compute_scalar_well_start_bank_torch,
)
from CliPP2.core.fusion.torch_backend import (
    refine_graph_fusion_dual_certificate_torch,
    resolve_runtime,
    to_torch_tumor_data,
)
from CliPP2.core.fusion.types import (
    CompressedEdgeCertificate,
    PrimalOnlyWarmState,
    QuotientWorksetWarmState,
)
from CliPP2.core.model import FitOptions, fit_fixed_objective
from CliPP2.io.data import TumorData, load_tumor_tsv


def _subset_tumor_mutations(data: TumorData, mutation_indices: np.ndarray) -> TumorData:
    idx = np.asarray(mutation_indices, dtype=np.int64)
    return TumorData(
        tumor_id=data.tumor_id,
        mutation_ids=[data.mutation_ids[int(i)] for i in idx],
        region_ids=list(data.region_ids),
        alt_counts=np.asarray(data.alt_counts)[idx].copy(),
        total_counts=np.asarray(data.total_counts)[idx].copy(),
        purity=np.asarray(data.purity)[idx].copy(),
        major_cn=np.asarray(data.major_cn)[idx].copy(),
        minor_cn=np.asarray(data.minor_cn)[idx].copy(),
        normal_cn=np.asarray(data.normal_cn)[idx].copy(),
        has_cna=np.asarray(data.has_cna)[idx].copy(),
        scaling=np.asarray(data.scaling)[idx].copy(),
        phi_upper=np.asarray(data.phi_upper)[idx].copy(),
        phi_init=np.asarray(data.phi_init)[idx].copy(),
        init_major_mask=np.asarray(data.init_major_mask)[idx].copy(),
    )


def _toy_tumor_data(*, tumor_id: str = "toy", alt_shift: float = 0.0) -> TumorData:
    alt_counts = np.array([[2.0], [7.0]], dtype=np.float32) + float(alt_shift)
    total_counts = np.array([[10.0], [12.0]], dtype=np.float32)
    return TumorData(
        tumor_id=tumor_id,
        mutation_ids=["m0", "m1"],
        region_ids=["r0"],
        alt_counts=alt_counts,
        total_counts=total_counts,
        purity=np.ones_like(alt_counts, dtype=np.float32),
        major_cn=np.ones_like(alt_counts, dtype=np.float32),
        minor_cn=np.ones_like(alt_counts, dtype=np.float32),
        normal_cn=np.full_like(alt_counts, 2.0, dtype=np.float32),
        has_cna=np.zeros_like(alt_counts, dtype=bool),
        scaling=np.full_like(alt_counts, 0.5, dtype=np.float32),
        phi_upper=np.ones_like(alt_counts, dtype=np.float32),
        phi_init=np.full_like(alt_counts, 0.5, dtype=np.float32),
        init_major_mask=np.ones_like(alt_counts, dtype=bool),
    )


@pytest.mark.parametrize("edge_work_bytes", [None, 1])
def test_fixed_primal_certificate_stops_on_a_deterministic_plateau(
    edge_work_bytes: int | None,
) -> None:
    phi = torch.full((2, 1), -1.0, dtype=torch.float64)
    audit = refine_graph_fusion_dual_certificate_torch(
        phi=phi,
        grad_smooth=torch.zeros_like(phi),
        dual_kkt=None,
        lower=torch.zeros_like(phi),
        upper=torch.ones_like(phi),
        edge_u=torch.tensor([0]),
        edge_v=torch.tensor([1]),
        edge_w=torch.ones(1, dtype=phi.dtype),
        lambda_value=1.0,
        atol=1e-8,
        max_iter=100,
        edge_work_bytes=edge_work_bytes,
    )

    assert audit["diag"]["kkt_residual"] > 5e-8
    assert audit["refinement_iterations"] == 8


@pytest.mark.parametrize("edge_work_bytes", [None, 1])
def test_fixed_primal_certificate_does_not_confuse_dual_motion_with_a_plateau(
    edge_work_bytes: int | None,
) -> None:
    phi = torch.tensor([[1.0], [1.0], [0.5]], dtype=torch.float64)
    audit = refine_graph_fusion_dual_certificate_torch(
        phi=phi,
        grad_smooth=torch.tensor(
            [[0.2365472727217759], [-2.00356390703028], [0.9668085873661573]],
            dtype=phi.dtype,
        ),
        dual_kkt=torch.tensor(
            [[1.4403661842651112], [0.7074348754654124], [-1.0394116546875747]],
            dtype=phi.dtype,
        ),
        lower=torch.zeros_like(phi),
        upper=torch.ones_like(phi),
        edge_u=torch.tensor([0, 0, 1]),
        edge_v=torch.tensor([1, 2, 2]),
        edge_w=torch.tensor(
            [1.0748447666506027, 0.5279092788672166, 0.7756403820370112],
            dtype=phi.dtype,
        ),
        lambda_value=1.3400690303898812,
        atol=0.025,
        max_iter=64,
        edge_work_bytes=edge_work_bytes,
    )

    assert audit["diag"]["kkt_residual"] <= 5 * 0.025
    assert audit["refinement_iterations"] == 17


def test_damped_state_transition_invalidates_trial_certificate_and_dual() -> None:
    phi = torch.tensor([[0.4], [0.4]], dtype=torch.float64)
    labels = torch.zeros(2, dtype=torch.long)
    trial_state = QuotientWorksetWarmState(
        phi=torch.tensor([[0.2], [0.2]], dtype=torch.float64),
        labels=labels,
        centers=torch.tensor([[0.2]], dtype=torch.float64),
        quotient_dual=torch.tensor([[0.7]], dtype=torch.float64),
        internal_edge_ids=torch.tensor([0], dtype=torch.long),
        internal_dual=torch.tensor([[0.3]], dtype=torch.float64),
        graph_hash="graph",
        previous_lambda=1.0,
    )

    dual, dual_kkt, certificate, warm_state, dual_is_actual = (
        solver_module._invalidate_damped_trial_state(
            phi=phi,
            trial_warm_state=trial_state,
        )
    )

    assert dual is None
    assert dual_kkt is None
    assert certificate is None
    assert not dual_is_actual
    assert isinstance(warm_state, PrimalOnlyWarmState)
    assert warm_state.phi is phi
    assert warm_state.structure_hint is labels
    assert warm_state.structure_hint_is_heuristic
    assert isinstance(warm_state.certificate_hint, CompressedEdgeCertificate)
    torch.testing.assert_close(
        warm_state.certificate_hint.internal_dual,
        trial_state.internal_dual,
    )


@pytest.mark.parametrize("bad_lambda", [-1.0, float("nan"), float("inf")])
def test_fit_rejects_invalid_lambda_at_public_boundary(bad_lambda: float) -> None:
    data = _toy_tumor_data()
    options = FitOptions(
        lambda_value=bad_lambda,
        outer_max_iter=1,
        inner_max_iter=16,
        tol=1e-4,
        device="cpu",
        dtype="float64",
    )

    with pytest.raises(ValueError, match="lambda_value"):
        fit_fixed_objective(data, options, compute_summary=False)


def test_solver_context_rejects_mismatched_tumor_data() -> None:
    data_a = _toy_tumor_data(tumor_id="toy_a")
    data_b = _toy_tumor_data(tumor_id="toy_b", alt_shift=1.0)
    options = FitOptions(
        lambda_value=0.1,
        outer_max_iter=1,
        inner_max_iter=16,
        tol=1e-4,
        device="cpu",
        dtype="float64",
    )
    context = prepare_torch_problem(
        data_a,
        major_prior=options.major_prior,
        eps=options.eps,
        tol=options.tol,
        inner_max_iter=options.inner_max_iter,
        graph=options.graph,
        adaptive_weight_gamma=options.adaptive_weight_gamma,
        adaptive_weight_floor=options.adaptive_weight_floor,
        adaptive_weight_baseline=options.adaptive_weight_baseline,
        device=options.device,
        dtype=options.dtype,
    )

    with pytest.raises(ValueError, match="data fingerprint"):
        fit_fixed_objective(
            data_b, options, solver_context=context, compute_summary=False
        )


def test_final_outer_kkt_audit_certifies_accepted_full_step() -> None:
    path = Path("/data/CliPP2/CliPP2Sim1K_TSV/50_6_0.6_0.0_S2_Lm300_M273_rep0.tsv")
    if not path.exists():
        pytest.skip(f"Missing regression fixture: {path}")

    fit = fit_fixed_objective(
        load_tumor_tsv(path),
        FitOptions(
            lambda_value=0.1,
            outer_max_iter=2,
            inner_max_iter=30,
            tol=1e-4,
            device="cpu",
            dtype="float64",
        ),
    )

    assert fit.accepted_outer_steps == 1
    assert fit.accepted_full_steps == 1
    assert fit.accepted_damped_steps == 0
    assert fit.attempted_outer_steps >= fit.accepted_outer_steps
    assert np.isfinite(fit.accepted_inner_kkt_residual)
    assert np.isfinite(fit.inner_kkt_residual)
    assert np.isfinite(fit.best_attempted_inner_kkt_residual)
    assert fit.accepted_inner_kkt_residual < 1e-5
    assert fit.best_attempted_inner_kkt_residual < 1e-5
    assert np.isfinite(fit.best_attempted_inner_model_gap)
    assert fit.best_attempted_inner_model_gap <= 1e-8
    assert fit.outer_stationarity_residual_before_dual_refine < 1e-4
    assert fit.outer_projected_stationarity_residual == fit.outer_stationarity_residual
    assert np.isfinite(fit.outer_projected_stationarity_norm)
    assert np.isfinite(fit.outer_stationarity_normalizer)
    assert np.isfinite(fit.outer_smooth_gradient_norm)
    assert np.isfinite(fit.outer_fusion_adjustment_norm)
    assert fit.outer_stationarity_normalizer >= 1.0
    assert fit.outer_box_primal_violation == 0.0
    assert (
        fit.outer_num_interior_coordinates
        + fit.outer_num_lower_active_coordinates
        + fit.outer_num_upper_active_coordinates
        + fit.outer_num_frozen_coordinates
    ) == fit.phi.size
    assert fit.fixed_objective_kkt_residual <= 5.0 * 1e-4
    assert fit.outer_kkt_certificate_status == "input_dual_retained"
    assert not fit.outer_kkt_dual_refined
    assert fit.outer_kkt_fused_edges > 0
    assert fit.selection_eligible
    assert fit.exactness_provenance_version == 1
    assert fit.estimator_role == "raw_fused_lambda_path"
    assert fit.objective_faithful
    assert fit.objective_spec_hash
    assert fit.original_graph_hash
    assert fit.certificate_problem_hash
    assert fit.certificate_scope == "full_original_graph"
    assert fit.certificate_gradient_scope == "observed_objective"
    assert fit.full_kkt_certified
    assert fit.full_kkt_certificate_status == fit.outer_kkt_certificate_status
    assert fit.full_kkt_tolerance == pytest.approx(5.0e-4)
    assert fit.inner_backend == "admm_complete_graph"
    assert fit.stationarity_certified
    assert fit.global_optimality_certified
    assert fit.global_optimality_basis == "assumed_unimodal_objective_plus_kkt"
    assert fit.number_of_starts == 1
    assert 1 <= fit.number_of_finite_starts <= fit.number_of_starts
    assert np.isfinite(fit.best_start_objective)
    assert fit.best_start_objective <= fit.penalized_objective + 1e-8
    assert fit.objective_spread_across_starts >= 0.0
    assert fit.selected_start_objective_rank >= 1
    assert fit.converged
    assert fit.converged_outer
    assert fit.converged_inner
    assert fit.failure_reason == "converged"


def test_full_step_recovery_accepts_only_matching_admm_endpoints() -> None:
    data = _toy_tumor_data()
    fit = fit_fixed_objective(
        data,
        FitOptions(
            lambda_value=0.2,
            outer_max_iter=6,
            inner_max_iter=60,
            tol=1e-4,
            device="cpu",
            dtype="float64",
            objective_shape="unimodal_full_step_backtracking",
        ),
        start_mode="warm_only",
        compute_summary=False,
    )

    assert fit.inner_solver == "admm_complete_graph"
    assert fit.admm_iterations > 0
    assert fit.accepted_outer_steps == fit.accepted_full_steps
    assert fit.accepted_full_steps > 0
    assert fit.accepted_damped_steps == 0
    assert fit.attempted_outer_steps >= fit.accepted_outer_steps
    assert np.all(np.diff(np.asarray(fit.history, dtype=np.float64)) <= 1e-10)
    assert fit.selection_eligible
    assert fit.fixed_objective_kkt_residual <= 5.0 * 1e-4


def test_fit_reuses_prepared_solver_context_with_tensor_start() -> None:
    path = Path("/data/CliPP2/CliPP2Sim1K_TSV/50_6_0.6_0.0_S2_Lm300_M273_rep0.tsv")
    if not path.exists():
        pytest.skip(f"Missing regression fixture: {path}")

    data = load_tumor_tsv(path)
    options = FitOptions(
        lambda_value=0.1,
        outer_max_iter=2,
        inner_max_iter=30,
        tol=1e-4,
        device="cpu",
        dtype="float64",
    )
    context = prepare_torch_problem(
        data,
        major_prior=options.major_prior,
        eps=options.eps,
        tol=options.tol,
        inner_max_iter=options.inner_max_iter,
        graph=options.graph,
        adaptive_weight_gamma=options.adaptive_weight_gamma,
        adaptive_weight_floor=options.adaptive_weight_floor,
        adaptive_weight_baseline=options.adaptive_weight_baseline,
        device=options.device,
        dtype=options.dtype,
    )

    fit = fit_fixed_objective(
        data,
        options,
        phi_start=context.exact_pilot,
        start_mode="warm_only",
        solver_context=context,
    )

    assert context.graph.edge_u.device.type == "cpu"
    assert context.graph_spec.edge_u.shape == context.graph.edge_u.cpu().numpy().shape
    assert context.scalar_well_starts == ()
    assert np.isfinite(fit.penalized_objective)
    assert fit.graph_name == context.graph_spec.name
    assert fit.number_of_starts == 1


def test_solver_context_honors_per_call_scalar_starts(monkeypatch) -> None:
    data = _toy_tumor_data()
    options = FitOptions(
        lambda_value=0.1,
        outer_max_iter=1,
        inner_max_iter=16,
        tol=1e-4,
        device="cpu",
        dtype="float64",
        objective_shape="generic_nonconvex",
    )
    context = prepare_torch_problem(
        data,
        major_prior=options.major_prior,
        eps=options.eps,
        tol=options.tol,
        inner_max_iter=options.inner_max_iter,
        graph=options.graph,
        adaptive_weight_gamma=options.adaptive_weight_gamma,
        adaptive_weight_floor=options.adaptive_weight_floor,
        adaptive_weight_baseline=options.adaptive_weight_baseline,
        device=options.device,
        dtype=options.dtype,
        objective_shape=options.objective_shape,
    )
    custom_start = torch.full_like(context.exact_pilot, 0.37)
    observed_starts: list[torch.Tensor] = []
    original_fit_from_start = solver_module._fit_from_start

    def recording_fit_from_start(*args, **kwargs):
        phi_start = kwargs["phi_start"]
        if torch.is_tensor(phi_start):
            observed_starts.append(phi_start.detach().clone())
        return original_fit_from_start(*args, **kwargs)

    monkeypatch.setattr(solver_module, "_fit_from_start", recording_fit_from_start)

    fit_fixed_objective(
        data,
        options,
        solver_context=context,
        scalar_well_starts=[custom_start],
        start_mode="full",
        compute_summary=False,
    )

    assert any(torch.allclose(start, custom_start) for start in observed_starts)


def test_fit_reuses_solver_state_actual_dual_for_alm_warm_start(monkeypatch) -> None:
    data = _toy_tumor_data()
    options = FitOptions(
        lambda_value=0.2,
        outer_max_iter=1,
        inner_max_iter=16,
        tol=1e-4,
        device="cpu",
        dtype="float64",
    )
    first_fit = fit_fixed_objective(
        data, options, start_mode="warm_only", compute_summary=False
    )
    assert first_fit.solver_state is not None
    assert first_fit.solver_state.dual is not None
    assert not hasattr(first_fit.solver_state, "split")
    assert first_fit.solver_state.dual.shape[0] == 1

    original_alm = solver_module.solve_majorized_subproblem_alm_torch
    first_call: list[dict[str, object]] = []

    def recording_alm(**kwargs):
        if not first_call:
            first_call.append(
                {
                    "dual_start_is_actual": bool(
                        kwargs.get("dual_start_is_actual", False)
                    ),
                    "dual_start_is_none": kwargs.get("dual_start") is None,
                }
            )
        return original_alm(**kwargs)

    monkeypatch.setattr(
        solver_module, "solve_majorized_subproblem_alm_torch", recording_alm
    )

    second_fit = fit_fixed_objective(
        data,
        replace(options, lambda_value=0.1),
        phi_start=first_fit.solver_state.phi,
        solver_state=first_fit.solver_state,
        start_mode="warm_only",
        compute_summary=False,
    )

    assert first_call == [{"dual_start_is_actual": True, "dual_start_is_none": False}]
    assert second_fit.solver_state is not None
    assert second_fit.solver_state.dual is not None
    assert not hasattr(second_fit.solver_state, "split")


def test_complete_graph_fit_reports_accumulated_admm_iterations(monkeypatch) -> None:
    data = _toy_tumor_data()
    options = FitOptions(
        lambda_value=0.2,
        outer_max_iter=2,
        inner_max_iter=16,
        tol=1e-4,
        device="cpu",
        dtype="float64",
    )
    original_alm = solver_module.solve_majorized_subproblem_alm_torch
    backend_iterations: list[int] = []

    def recording_alm(**kwargs):
        result = original_alm(**kwargs)
        backend_iterations.append(int(result[3]))
        return result

    monkeypatch.setattr(
        solver_module, "solve_majorized_subproblem_alm_torch", recording_alm
    )

    fit = fit_fixed_objective(
        data, options, start_mode="warm_only", compute_summary=False
    )

    assert backend_iterations
    assert all(value > 0 for value in backend_iterations)
    assert fit.inner_solver == "admm_complete_graph"
    assert fit.inner_iterations == sum(backend_iterations)
    assert fit.admm_iterations == sum(backend_iterations)
    assert fit.diagnostics.inner_solver == "admm_complete_graph"
    assert fit.diagnostics.inner_iterations == sum(backend_iterations)
    assert fit.diagnostics.admm_iterations == sum(backend_iterations)
    assert fit.iterations <= options.outer_max_iter
    assert fit.admm_iterations > fit.iterations


def test_zero_lambda_fit_reports_closed_form_and_zero_admm_iterations() -> None:
    data = _toy_tumor_data()
    fit = fit_fixed_objective(
        data,
        FitOptions(
            lambda_value=0.0,
            outer_max_iter=2,
            inner_max_iter=16,
            tol=1e-4,
            device="cpu",
            dtype="float64",
        ),
        start_mode="warm_only",
        compute_summary=False,
    )

    assert fit.inner_solver == "closed_form_projection"
    assert fit.inner_iterations == 0
    assert fit.admm_iterations == 0
    assert fit.diagnostics.inner_solver == "closed_form_projection"
    assert fit.diagnostics.inner_iterations == 0
    assert fit.diagnostics.admm_iterations == 0


def test_fit_torch_returns_tensor_result_and_reusable_state() -> None:
    data = _toy_tumor_data()
    context = prepare_torch_problem(
        data,
        major_prior=0.5,
        eps=1e-6,
        tol=1e-4,
        inner_max_iter=16,
        device="cpu",
        dtype="float64",
    )

    first_result, first_state = fit_torch(
        data,
        context=context,
        lambda_value=0.2,
        outer_max_iter=1,
        inner_max_iter=16,
        tol=1e-4,
    )
    second_result, second_state = fit_torch(
        data,
        context=context,
        lambda_value=0.1,
        state=first_state,
        outer_max_iter=1,
        inner_max_iter=16,
        tol=1e-4,
    )

    assert first_result.phi_raw.device == context.runtime.device
    assert first_result.objective.ndim == 0
    assert first_result.fit_loss.ndim == 0
    assert first_result.fusion_penalty.ndim == 0
    assert first_result.dual is not None
    assert first_result.inner_solver == "admm_complete_graph"
    assert first_result.inner.iterations > 0
    assert first_result.admm_iterations == first_result.inner.iterations
    assert torch.allclose(first_result.phi_raw, first_state.phi)
    assert torch.allclose(first_result.dual, first_state.dual)
    assert first_state.previous_lambda == pytest.approx(0.2)
    assert second_state.previous_lambda == pytest.approx(0.1)
    assert second_result.phi_raw.device == context.runtime.device
    assert torch.allclose(second_result.phi_raw, second_state.phi)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fit_torch_keeps_active_core_state_on_cuda() -> None:
    data = _toy_tumor_data()
    context = prepare_torch_problem(
        data,
        major_prior=0.5,
        eps=1e-6,
        tol=1e-4,
        inner_max_iter=16,
        device="cuda",
        dtype="float64",
    )

    result, state = fit_torch(
        data,
        context=context,
        lambda_value=0.2,
        outer_max_iter=1,
        inner_max_iter=16,
        tol=1e-4,
    )

    assert context.runtime.device.type == "cuda"
    assert context.graph.edge_u.device.type == "cuda"
    assert context.graph.edge_v.device.type == "cuda"
    assert context.graph.weight.device.type == "cuda"
    assert result.phi_raw.device.type == "cuda"
    assert result.gamma_major.device.type == "cuda"
    assert result.dual is not None and result.dual.device.type == "cuda"
    assert state.phi.device.type == "cuda"
    assert state.dual is not None and state.dual.device.type == "cuda"
    assert state.warm_state is not None
    assert state.warm_state.phi.device.type == "cuda"
    assert state.certificate is not None
    assert state.certificate.dual.device.type == "cuda"


def test_torch_pooled_start_matches_legacy_start_and_context_storage() -> None:
    path = Path("/data/CliPP2/CliPP2Sim1K_TSV/50_6_0.6_0.0_S2_Lm300_M273_rep0.tsv")
    if not path.exists():
        pytest.skip(f"Missing regression fixture: {path}")

    data = load_tumor_tsv(path)
    runtime = resolve_runtime("cpu", dtype="float64")
    pilot, _, _ = compute_scalar_mutation_region_wells(
        data,
        major_prior=0.5,
        eps=1e-6,
        tol=1e-4,
        max_iter=30,
    )
    legacy = compute_pooled_observed_data_start(
        data,
        runtime=runtime,
        major_prior=0.5,
        eps=1e-6,
        tol=1e-4,
        max_iter=30,
        beta_hints=pilot,
    )
    torch_data = to_torch_tumor_data(data, runtime)
    tensor_start = compute_pooled_observed_data_start_torch(
        torch_data,
        major_prior=0.5,
        eps=1e-6,
        tol=1e-4,
        max_iter=30,
        beta_hints=pilot,
    )
    context = prepare_torch_problem(
        data,
        major_prior=0.5,
        eps=1e-6,
        tol=1e-4,
        inner_max_iter=30,
        exact_pilot=pilot,
        device="cpu",
        dtype="float64",
        objective_shape="generic_nonconvex",
    )

    assert np.allclose(tensor_start.detach().cpu().numpy(), legacy, atol=1e-10)
    assert np.allclose(context.pooled_start.detach().cpu().numpy(), legacy, atol=1e-10)


def test_torch_scalar_mutation_region_wells_match_legacy_on_ambiguous_subset() -> None:
    path = Path("/data/CliPP2/CliPP2Sim1K_TSV/300_5_0.3_0.2_S20_Lm4000_M4201_rep0.tsv")
    if not path.exists():
        pytest.skip(f"Missing regression fixture: {path}")

    full_data = load_tumor_tsv(path)
    ambiguous_mutations = np.flatnonzero(
        np.any(full_data.multiplicity_estimation_mask, axis=1)
    )
    if ambiguous_mutations.size == 0:
        pytest.skip(
            f"Regression fixture has no ambiguous scalar mutation_regions: {path}"
        )
    data = _subset_tumor_mutations(full_data, ambiguous_mutations[:40])
    runtime = resolve_runtime("cpu", dtype="float64")
    torch_data = to_torch_tumor_data(data, runtime)

    legacy_primary, legacy_secondary, legacy_valid = (
        compute_scalar_mutation_region_wells(
            data,
            major_prior=0.5,
            eps=1e-6,
            tol=1e-4,
            max_iter=30,
        )
    )
    tensor_primary, tensor_secondary, tensor_valid = (
        compute_scalar_mutation_region_wells_torch(
            torch_data,
            phi_init=data.phi_init,
            major_prior=0.5,
            eps=1e-6,
            tol=1e-4,
            max_iter=30,
        )
    )

    assert np.allclose(tensor_primary.detach().cpu().numpy(), legacy_primary, atol=1e-8)
    assert np.array_equal(tensor_valid.detach().cpu().numpy(), legacy_valid)
    assert np.allclose(
        tensor_secondary.detach().cpu().numpy(),
        legacy_secondary,
        atol=1e-8,
        equal_nan=True,
    )

    legacy_bank = compute_scalar_well_start_bank(
        data,
        major_prior=0.5,
        eps=1e-6,
        tol=1e-4,
        max_iter=30,
        exact_pilot=legacy_primary,
        secondary_wells=legacy_secondary,
        valid_secondary=legacy_valid,
    )
    context = prepare_torch_problem(
        data,
        major_prior=0.5,
        eps=1e-6,
        tol=1e-4,
        inner_max_iter=30,
        exact_pilot=legacy_primary,
        device="cpu",
        dtype="float64",
        objective_shape="generic_nonconvex",
    )
    assert len(context.scalar_well_starts) == len(legacy_bank)
    for tensor_start, legacy_start in zip(context.scalar_well_starts, legacy_bank):
        assert np.allclose(tensor_start.detach().cpu().numpy(), legacy_start, atol=1e-8)


def test_torch_scalar_mutation_region_wells_rank_on_torch() -> None:
    path = Path("/data/CliPP2/CliPP2Sim1K_TSV/300_5_0.3_0.2_S20_Lm4000_M4201_rep0.tsv")
    if not path.exists():
        pytest.skip(f"Missing regression fixture: {path}")

    full_data = load_tumor_tsv(path)
    ambiguous_mutations = np.flatnonzero(
        np.any(full_data.multiplicity_estimation_mask, axis=1)
    )
    if ambiguous_mutations.size == 0:
        pytest.skip(
            f"Regression fixture has no ambiguous scalar mutation_regions: {path}"
        )
    data = _subset_tumor_mutations(full_data, ambiguous_mutations[:8])
    runtime = resolve_runtime("cpu", dtype="float64")
    torch_data = to_torch_tumor_data(data, runtime)

    primary, secondary, valid = compute_scalar_mutation_region_wells_torch(
        torch_data,
        phi_init=data.phi_init,
        major_prior=0.5,
        eps=1e-6,
        tol=1e-4,
        max_iter=30,
    )

    assert primary.device.type == "cpu"
    assert secondary.device.type == "cpu"
    assert valid.device.type == "cpu"
    assert torch.all(torch.isfinite(primary))


def test_torch_scalar_well_start_bank_matches_legacy_bank() -> None:
    path = Path("/data/CliPP2/CliPP2Sim1K_TSV/50_6_0.6_0.0_S2_Lm300_M273_rep0.tsv")
    if not path.exists():
        pytest.skip(f"Missing regression fixture: {path}")

    data = load_tumor_tsv(path)
    runtime = resolve_runtime("cpu", dtype="float64")
    torch_data = to_torch_tumor_data(data, runtime)
    pilot, secondary, valid_secondary = compute_scalar_mutation_region_wells(
        data,
        major_prior=0.5,
        eps=1e-6,
        tol=1e-4,
        max_iter=30,
    )
    legacy_bank = compute_scalar_well_start_bank(
        data,
        major_prior=0.5,
        eps=1e-6,
        tol=1e-4,
        max_iter=30,
        exact_pilot=pilot,
        secondary_wells=secondary,
        valid_secondary=valid_secondary,
    )
    tensor_bank = compute_scalar_well_start_bank_torch(
        torch_data,
        eps=1e-6,
        exact_pilot=torch_data.phi_upper.new_tensor(pilot),
        secondary_wells=secondary,
        valid_secondary=valid_secondary,
    )
    context = prepare_torch_problem(
        data,
        major_prior=0.5,
        eps=1e-6,
        tol=1e-4,
        inner_max_iter=30,
        device="cpu",
        dtype="float64",
        objective_shape="generic_nonconvex",
    )

    assert len(tensor_bank) == len(legacy_bank)
    assert len(context.scalar_well_starts) == len(legacy_bank)
    for tensor_start, legacy_start in zip(tensor_bank, legacy_bank):
        assert tensor_start.device.type == "cpu"
        assert np.allclose(
            tensor_start.detach().cpu().numpy(), legacy_start, atol=1e-10
        )
    for tensor_start, legacy_start in zip(context.scalar_well_starts, legacy_bank):
        assert tensor_start.device.type == "cpu"
        assert np.allclose(
            tensor_start.detach().cpu().numpy(), legacy_start, atol=1e-10
        )
