"""Fixed-primal dual updates and typed numerical diagnostics, CPU only."""
from dataclasses import asdict, replace
import os

import numpy as np
import pytest
import torch

from CliPP2.core.fusion import torch_backend as backend
from CliPP2.core.fusion import certificates
from CliPP2.core.fusion.graph_ops import tensorize_graph
from CliPP2.core.fusion.solver import _backward_error_kkt_within_gate
from CliPP2.core.fusion.types import (
    CertificateOptions, CompressedEdgeCertificate, DenseEdgeCertificate,
    KKTDiagnostics, PairwiseFusionGraph,
)
from CliPP2.core.objective import compile_observed_model, model_to_torch, observed_terms_torch
from test_integer_likelihood import EPS, integer_data


def _binomial_counterexample():
    data = replace(
        integer_data(((1,),) * 4, purity=1.),
        alt_counts=np.array([[0.], [500.], [0.], [1.]]),
        total_counts=np.array([[1500.], [500.], [3.], [1.]]),
    )
    model = model_to_torch(
        compile_observed_model(data, eps=EPS),
        backend.resolve_runtime("cpu", dtype="float64"), eps=EPS,
    )
    phi = torch.full((4, 1), .5, dtype=torch.float64)
    gradient = observed_terms_torch(model, phi, eps=EPS).gradient
    torch.testing.assert_close(
        gradient, torch.tensor([[1000.], [-1000.], [2.], [-2.]], dtype=phi.dtype),
        rtol=0, atol=0,
    )
    edge_u, edge_v = torch.triu_indices(4, 4, 1)
    dual = torch.zeros((6, 1), dtype=phi.dtype)
    dual[0] = -1000.
    return dict(
        phi=phi, grad_smooth=gradient, dual_kkt=dual,
        lower=model.lower, upper=model.upper, edge_u=edge_u, edge_v=edge_v,
        edge_w=torch.full((6,), 1 / 3, dtype=phi.dtype),
        lambda_value=3000., atol=8e-4,
    )


def test_supported_binomial_witness_requires_authoritative_refinement():
    kwargs = _binomial_counterexample()
    frozen = {key: value.clone() for key, value in kwargs.items() if torch.is_tensor(value)}
    before = backend.graph_fusion_kkt_residual_from_grad_torch(**kwargs)
    assert before.kkt_residual == pytest.approx(.000249911, abs=1e-9)
    assert before.kkt_residual < .004
    assert before.backward_error_kkt_residual == 1.
    exact = kwargs["dual_kkt"].clone()
    exact[-1] = -2.
    certified = backend.graph_fusion_kkt_residual_from_grad_torch(
        **dict(kwargs, dual_kkt=exact),
    )
    assert certified.backward_error_kkt_residual == 0.
    result = backend.refine_graph_fusion_dual_certificate_torch(**kwargs)
    assert result["refinement_iterations"] > 0
    assert result["status"] == "refined_fused_edge_dual"
    assert result["diag"].backward_error_kkt_residual <= .004
    for key, value in frozen.items():
        assert torch.equal(kwargs[key], value), key


def _counterexample_certificate_problem(*, compressed):
    kwargs = _binomial_counterexample()
    graph_spec = PairwiseFusionGraph(
        edge_u=kwargs["edge_u"].numpy(), edge_v=kwargs["edge_v"].numpy(),
        edge_w=kwargs["edge_w"].numpy(), degree_bound=3,
    )
    graph = tensorize_graph(
        graph_spec, backend.resolve_runtime("cpu", dtype="float64"), num_nodes=4,
    )
    problem = certificates.CertificateProblem(
        graph=graph, graph_hash=graph_spec.fingerprint,
        lower=kwargs["lower"], upper=kwargs["upper"],
        lambda_value=kwargs["lambda_value"], atol=kwargs["atol"],
    )
    gradient = certificates.CertificateGradient(
        value=kwargs["grad_smooth"], scope="observed_objective",
        directional_admissible=True, at_breakpoint=torch.zeros_like(kwargs["phi"], dtype=torch.bool),
    )
    witness_args = dict(graph_hash=problem.graph_hash, gradient_scope=gradient.scope)
    witness = (
        CompressedEdgeCertificate(
            labels=torch.zeros(4, dtype=torch.long), centers=kwargs["phi"][:1].clone(),
            internal_edge_ids=torch.arange(6), internal_dual=kwargs["dual_kkt"], **witness_args,
        ) if compressed else DenseEdgeCertificate(dual=kwargs["dual_kkt"], **witness_args)
    )
    return dict(problem=problem, phi=kwargs["phi"], gradient=gradient, witness=witness)


@pytest.mark.parametrize("compressed", [False, True])
def test_certificate_wrapper_refines_legacy_passing_binomial_witness(compressed):
    kwargs = _counterexample_certificate_problem(compressed=compressed)
    before = certificates.certify(**kwargs, refine=False)
    assert before.diagnostics.kkt_residual < .004
    assert before.diagnostics.backward_error_kkt_residual == 1.
    result = certificates.certify(**kwargs, refine=True)
    assert result.diagnostics.backward_error_kkt_residual <= .004
    assert result.certificate.graph_hash == kwargs["problem"].graph_hash
    assert result.certificate.gradient_scope == "observed_objective"
    after = certificates.certify(**dict(kwargs, witness=result.certificate), refine=False)
    assert after.diagnostics == result.diagnostics


def test_compressed_final_audit_rejects_legacy_only_passing_witness(monkeypatch):
    kwargs = _counterexample_certificate_problem(compressed=True)

    def unchanged_workset(**values):
        return values["dual_start"], torch.zeros_like(values["phi"]), 0., 1

    monkeypatch.setattr(certificates, "_optimize_internal_workset", unchanged_workset)
    result = certificates.certify(
        **kwargs, refine=True, options=CertificateOptions(refinement_rounds=0),
    )
    assert result.work_counters.full_certificate_audit_passes == 2
    assert result.status == "not_certified"
    assert result.diagnostics.kkt_residual < .004
    assert result.diagnostics.backward_error_kkt_residual == 1.


def test_dense_retention_and_plateau_receive_authoritative_residuals(monkeypatch):
    kwargs = _problem(torch.float64, case="incoming_after_analytic")
    audit_values = iter(zip([.01, .02, .003, .002, .001], [.95, .9, .8, .7, .75]))
    traces, duals, plateaus = [], [], []

    def audit(**values):
        legacy, authoritative = next(audit_values)
        diag = KKTDiagnostics(legacy, 0., 0., 0., authoritative, 0., 0.)
        traces.append(diag)
        duals.append(values["dual_kkt"].clone())
        return diag

    def plateau(**values):
        plateaus.append(values)
        return values["anchor_residual"], 0, False

    monkeypatch.setattr(backend, "graph_fusion_kkt_residual_from_grad_torch", audit)
    monkeypatch.setattr(backend, "_update_certificate_refinement_plateau", plateau)
    result = backend.refine_graph_fusion_dual_certificate_torch(**kwargs, max_iter=3)
    assert result["refinement_iterations"] == 3
    assert result["diag"] == traces[3]
    assert torch.equal(result["dual"], duals[3])
    assert result["diag"].kkt_residual > traces[4].kkt_residual
    assert [item["anchor_residual"] for item in plateaus] == [.9] * 3
    assert [item["best_residual"] for item in plateaus] == [.8, .7, .7]


def test_dense_target_stopping_uses_authoritative_residual(monkeypatch):
    kwargs = _problem(torch.float64)
    kwargs["atol"] = 8e-4
    audits = []

    def audit(**_values):
        authoritative = [1., .9, .5, .003][len(audits)]
        diag = KKTDiagnostics(.0001, 0., 0., 0., authoritative, 0., 0.)
        audits.append(diag)
        return diag

    monkeypatch.setattr(backend, "graph_fusion_kkt_residual_from_grad_torch", audit)
    result = backend.refine_graph_fusion_dual_certificate_torch(**kwargs)
    assert result["refinement_iterations"] == 2
    assert result["diag"].backward_error_kkt_residual == .003


def _problem(dtype, *, case="fused"):
    phi = torch.full((4, 2), .5, dtype=dtype)
    grad = torch.tensor([[2., -1.], [-2., 1.], [1., -.5], [-1., .5]], dtype=dtype)
    edge_u, edge_v = torch.triu_indices(4, 4, 1)
    weight = torch.full((6,), .25, dtype=dtype)
    dual, lam = None, 4.
    if case == "zero_graph":
        edge_u, edge_v, weight = edge_u[:0], edge_v[:0], weight[:0]
    elif case == "zero_penalty":
        lam = 0.
    elif case == "incoming":
        dual = torch.zeros((6, 2), dtype=dtype)
        grad.zero_()
    elif case == "analytic":
        phi = torch.tensor([[.1, .2], [.3, .4], [.6, .7], [.8, .9]], dtype=dtype)
    elif case in ("mixed", "incoming_after_analytic"):
        phi[2:] = .8
        if case == "incoming_after_analytic":
            dual = torch.full((6, 2), .1, dtype=dtype)
            grad = -backend.graph_adjoint_edges(dual, edge_u=edge_u, edge_v=edge_v, num_nodes=4)
    elif case == "plateau":
        grad = torch.full_like(phi, 2.)
    return dict(
        phi=phi, grad_smooth=grad, dual_kkt=dual,
        lower=torch.zeros_like(phi), upper=torch.ones_like(phi),
        edge_u=edge_u, edge_v=edge_v, edge_w=weight,
        lambda_value=lam, atol=1e-6,
    )


def _capture(monkeypatch, kwargs, *, chunk_edges, max_iter=32):
    traces, duals = [], []
    original = backend.graph_fusion_kkt_residual_from_grad_torch

    def audit(**values):
        result = original(**values)
        assert isinstance(result, KKTDiagnostics)
        traces.append(result.kkt_residual)
        duals.append(None if values["dual_kkt"] is None else values["dual_kkt"].clone())
        return result

    with monkeypatch.context() as patch:
        patch.setattr(backend, "graph_fusion_kkt_residual_from_grad_torch", audit)
        result = backend.refine_graph_fusion_dual_certificate_torch(
            **kwargs, max_iter=max_iter,
            edge_work_bytes=chunk_edges * 2 * kwargs["phi"].element_size(),
        )
    return result, traces, duals


# Pinned 72f76a8 float64 CPU values, independently captured before deletion.
# These exercise the nonmonotone trial residuals as well as best-witness retention.
_PINNED = {
    "fused": (32, [.31180751631538306, .31180751631538306,
                    .26210321234714, .2323615986714042, .21212729076097853],
              .0008441646074105967),
    "plateau": (8, [.21244472379441628] * 5, .21244472379441628),
    "incoming_after_analytic": (8, [.21213203435596428] + [.3343934897139791] * 4,
                                 .21213203435596428),
}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("case", [
    "zero_graph", "zero_penalty", "incoming", "analytic", "fused",
    "mixed", "plateau", "incoming_after_analytic",
])
def test_chunking_preserves_sequences_status_and_best_witness(monkeypatch, dtype, case):
    kwargs = _problem(dtype, case=case)
    phi_before = kwargs["phi"].clone()
    incoming_before = None if kwargs["dual_kkt"] is None else kwargs["dual_kkt"].clone()
    single, sequence, _ = _capture(monkeypatch, kwargs, chunk_edges=100)
    streamed, streamed_sequence, _ = _capture(monkeypatch, kwargs, chunk_edges=2)
    tolerance = 2e-7 if dtype == torch.float32 else 1e-14
    assert streamed["status"] == single["status"]
    assert streamed["refinement_iterations"] == single["refinement_iterations"]
    assert streamed_sequence == pytest.approx(sequence, abs=tolerance)
    torch.testing.assert_close(streamed["dual"], single["dual"], rtol=0, atol=tolerance)
    assert asdict(streamed["diag"]) == pytest.approx(asdict(single["diag"]), abs=tolerance)
    assert torch.equal(kwargs["phi"], phi_before)
    if incoming_before is not None:
        assert torch.equal(kwargs["dual_kkt"], incoming_before)
    assert single["fused_edges"] + single["nonzero_edges"] == (
        0 if kwargs["lambda_value"] == 0 else kwargs["edge_u"].numel()
    )
    if case in _PINNED:
        iterations, prefix, final = _PINNED[case]
        assert single["refinement_iterations"] == iterations
        assert sequence[:len(prefix)] == pytest.approx(prefix, abs=tolerance)
        assert single["diag"].kkt_residual == pytest.approx(final, abs=tolerance)
        assert len(sequence) == iterations + 2
        assert single["diag"].kkt_residual == min(sequence)
    if case == "mixed":
        # The old legacy-residual policy ran 22 iterations and retained a
        # lower legacy residual. All witnesses here instead have saturated
        # authoritative error 1, so retain the analytic tie and stop at the
        # componentwise moving-plateau limit. Edge-update arithmetic is unchanged.
        assert single["refinement_iterations"] == 16
        assert single["diag"].backward_error_kkt_residual == 1.
        assert single["diag"].kkt_residual == pytest.approx(sequence[1], abs=tolerance)
        assert sequence[:5] == pytest.approx([
            .3094601239615637, .2033664829756155, .20445049661434517,
            .20290922775396536, .20179518178152384,
        ], abs=tolerance)
    if case in ("incoming", "incoming_after_analytic"):
        assert single["status"] == "input_dual_retained"
        assert not single["dual_refined"]
        assert torch.equal(single["dual"], incoming_before)
    elif case == "analytic":
        diff = kwargs["phi"][kwargs["edge_u"]] - kwargs["phi"][kwargs["edge_v"]]
        expected = diff / torch.linalg.vector_norm(diff, dim=1)[:, None]
        torch.testing.assert_close(single["dual"], expected, rtol=0, atol=tolerance)
        assert single["status"] == "analytic_nonfused_dual"
    elif case.startswith("zero_"):
        assert single["status"] == "zero_penalty_no_dual_needed"
        assert torch.count_nonzero(single["dual"]) == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("chunk_edges", [1, 2, 6])
@pytest.mark.parametrize("lambda_value", [.04, 4.])
def test_each_iteration_uses_one_frozen_full_residual(monkeypatch, dtype, chunk_edges, lambda_value):
    kwargs = _problem(dtype)
    kwargs["lambda_value"] = lambda_value
    _, _, audited_duals = _capture(monkeypatch, kwargs, chunk_edges=chunk_edges, max_iter=3)
    # Independent dense-incidence NumPy calculation. Sequential chunk updates
    # using a partly updated adjoint disagree from the first iteration onward.
    incidence = np.zeros((6, 4))
    incidence[np.arange(6), kwargs["edge_u"].numpy()] = 1
    incidence[np.arange(6), kwargs["edge_v"].numpy()] = -1
    phi = kwargs["phi"].numpy().astype(float)
    gradient = kwargs["grad_smooth"].numpy().astype(float)
    dual = np.zeros((6, 2))
    radius = lambda_value / 4
    assert len(audited_duals) == 5  # incoming + analytic + three updates
    for actual in audited_duals[2:]:
        residual = phi - np.clip(phi - (gradient + incidence.T @ dual), 0, 1)
        proposal = dual - (.25 / 3) * incidence @ residual
        dual = proposal / np.maximum(1, np.linalg.norm(proposal, axis=1)[:, None] / radius)
        np.testing.assert_allclose(actual.numpy(), dual, rtol=0,
                                   atol=4e-8 if dtype == torch.float32 else 2e-16)


def test_one_chunk_refinement_keeps_complete_graph_adjoint(monkeypatch):
    calls = []
    original = backend.graph_adjoint_edges

    def adjoint(dual, **kwargs):
        calls.append((tuple(dual.shape), kwargs["num_nodes"]))
        return original(dual, **kwargs)

    monkeypatch.setattr(backend, "graph_adjoint_edges", adjoint)
    result, _, _ = _capture(monkeypatch, _problem(torch.float64), chunk_edges=100, max_iter=3)
    assert result["refinement_iterations"] == 3
    # Three update adjoints plus analytic and each trial's full-graph audit.
    assert calls == [((6, 2), 4)] * 7


@pytest.mark.skipif(
    os.environ.get("CLIPP2_TEST_CUDA") != "1",
    reason="Enable CLIPP2_TEST_CUDA=1 only for explicit LSF CUDA qualification.",
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_chunking_preserves_reference_adjoint_routes(monkeypatch, dtype):
    assert torch.cuda.is_available(), "Explicit CUDA qualification requires a CUDA runtime."
    kwargs = {
        key: value.to("cuda") if torch.is_tensor(value) else value
        for key, value in _problem(dtype).items()
    }
    calls = []
    original = backend.graph_adjoint_edges

    def adjoint(dual, **options):
        calls.append((tuple(dual.shape), options["num_nodes"]))
        return original(dual, **options)

    monkeypatch.setattr(backend, "graph_adjoint_edges", adjoint)
    single, sequence, _ = _capture(monkeypatch, kwargs, chunk_edges=100, max_iter=3)
    # One-chunk updates keep the deterministic complete-graph adjoint, as do
    # the analytic/trial audits. Streamed updates retain their original scatter
    # reduction order; they must not silently adopt a different trajectory.
    assert calls == [((6, 2), 4)] * 7
    calls.clear()
    streamed, streamed_sequence, _ = _capture(monkeypatch, kwargs, chunk_edges=2, max_iter=3)
    assert calls == [((6, 2), 4)] * 4
    assert single["status"] == streamed["status"]
    assert single["refinement_iterations"] == streamed["refinement_iterations"] == 3
    tolerance = 2e-7 if dtype == torch.float32 else 1e-14
    assert streamed_sequence == pytest.approx(sequence, abs=tolerance)
    torch.testing.assert_close(streamed["dual"], single["dual"], rtol=0, atol=tolerance)
    assert asdict(streamed["diag"]) == pytest.approx(asdict(single["diag"]), abs=tolerance)


def test_kkt_record_preserves_distinct_progress_and_admission_quantities():
    diag = backend.graph_fusion_kkt_diagnostics_from_components_torch(
        phi=torch.tensor([[.5]], dtype=torch.float64),
        grad_smooth=torch.tensor([[2.]], dtype=torch.float64),
        adj=torch.tensor([[-1.]], dtype=torch.float64),
        lower=torch.zeros((1, 1)), upper=torch.ones((1, 1)), atol=1e-6,
        max_edge_residual=.6, max_ball_residual=.2, max_radius=2.,
        max_scaled_edge_residual=.4, max_scaled_ball_residual=.1,
    )
    assert isinstance(diag, KKTDiagnostics)
    assert diag.stationarity_residual == .125
    assert diag.backward_error_stationarity_residual == 1 / 3
    assert diag.kkt_residual == pytest.approx(.2)
    assert diag.backward_error_kkt_residual == .4
    assert set(asdict(diag)) == {
        "stationarity_residual", "edge_subgradient_residual", "dual_ball_residual",
        "box_residual", "backward_error_stationarity_residual",
        "backward_error_edge_subgradient_residual", "backward_error_dual_ball_residual",
    }
    assert not _backward_error_kkt_within_gate(diag, certification_tol=8e-4)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_audit_and_missing_admission_components_fail_closed(bad_value):
    kwargs = _problem(torch.float64)
    kwargs["grad_smooth"][0, 0] = bad_value
    diag = backend.graph_fusion_kkt_residual_from_grad_torch(**kwargs)
    assert isinstance(diag, KKTDiagnostics)
    assert not np.isfinite(diag.backward_error_kkt_residual)
    assert not _backward_error_kkt_within_gate(diag, certification_tol=8e-4)
    incomplete = KKTDiagnostics(0., 0., 0., 0.)
    assert not _backward_error_kkt_within_gate(incomplete, certification_tol=8e-4)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
@pytest.mark.parametrize("backend_name", ["alm_dense", "alm_streamed", "pdhg"])
@pytest.mark.parametrize("backward", [False, True])
def test_inner_results_return_typed_diagnostics_without_mapping_copies(dtype_name, backend_name, backward):
    runtime = backend.resolve_runtime("cpu", dtype=dtype_name)
    U = torch.tensor([[.1, .9], [.2, .8], [.7, .3], [.9, .1]], dtype=runtime.dtype)
    edge_u, edge_v = torch.triu_indices(4, 4, 1)
    counters = {}
    kwargs = dict(
        runtime=runtime, num_mutations=4, U=U,
        h=torch.tensor([[3., 5.], [6., 4.], [2., 7.], [8., 3.]], dtype=runtime.dtype),
        lower=torch.zeros_like(U), upper=torch.ones_like(U),
        lambda_value=1.7, edge_u=edge_u, edge_v=edge_v,
        edge_w=torch.ones(6, dtype=runtime.dtype) / 3,
        tol=1e-6, max_iter=48, phi_start=U, dual_start=None,
        use_backward_error_stopping=backward,
    )
    if backend_name == "pdhg":
        solve = backend.solve_majorized_subproblem_pdhg_torch
        kwargs["degree_bound"] = 3
    else:
        solve = backend.solve_majorized_subproblem_alm_torch
        kwargs["edge_work_bytes"] = (2 if backend_name.endswith("streamed") else 100) * 2 * U.element_size()
        kwargs["diagnostics_out"] = counters
    phi, dual, iterations, converged, diag = solve(**kwargs)
    assert isinstance(diag, KKTDiagnostics)
    assert iterations == 48 and not converged
    assert np.isfinite(diag.backward_error_kkt_residual if backward else diag.kkt_residual)
    recomputed = backend.graph_fusion_kkt_residual_from_grad_torch(
        phi=phi, grad_smooth=kwargs["h"] * (phi - U), dual_kkt=dual,
        lower=kwargs["lower"], upper=kwargs["upper"],
        edge_u=edge_u, edge_v=edge_v, edge_w=kwargs["edge_w"], lambda_value=1.7, atol=1e-6,
    )
    assert asdict(diag) == pytest.approx(asdict(recomputed), abs=2e-7 if dtype_name == "float32" else 1e-14)
    if backend_name != "pdhg":
        expected_audits = 5 if dtype_name == "float32" and backward else 6
        assert counters == {"inner_kkt_audits": expected_audits, "inner_stationarity_checks": 6}
    kwargs["lambda_value"] = 0.
    closed = solve(**kwargs)
    assert closed[2:4] == (0, True)
    if backend_name == "pdhg":
        assert closed[4] is None
    else:
        assert isinstance(closed[4], KKTDiagnostics)
        assert counters == {"inner_kkt_audits": 0, "inner_stationarity_checks": 0}
