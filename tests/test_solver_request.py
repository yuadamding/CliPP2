"""Prepared fusion contexts cannot silently replace the requested objective."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from CliPP2.config import resolve_fit_config
from CliPP2.core.fusion import solver
from CliPP2.core.fusion.graph import build_complete_uniform_graph
from test_integer_likelihood import EPS, integer_data


def _prepared(*, adaptive=False):
    data = integer_data(((2,), (3,)))
    graph = None if adaptive else build_complete_uniform_graph(2)
    context = solver.prepare_torch_problem(
        data, eps=EPS, tol=8e-4, inner_max_iter=16,
        graph=graph, exact_pilot=data.phi_init, pooled_start=data.phi_init,
        scalar_well_starts=(), device="cpu", dtype="float64",
    )
    return data, context


def _fit(data, context, **kwargs):
    options = resolve_fit_config(device="cpu", dtype="float64",
                                 outer_max_iter=2, inner_max_iter=16).solver
    return solver.fit_prepared(
        context, .1, options, phi_start=data.phi_init, **kwargs,
    )


def test_same_topology_different_graph_weights_rejected_before_solver(monkeypatch):
    data, context = _prepared()
    graph_b = replace(context.graph_spec, edge_w=context.graph_spec.edge_w * 2)
    monkeypatch.setattr(solver, "_fit_from_start", lambda *a, **k: pytest.fail("optimizer entered"))
    with pytest.raises(TypeError, match="graph"):
        _fit(data, context, graph=graph_b)
    with pytest.raises(ValueError, match="graph identity"):
        _fit(data, replace(context, graph_spec=graph_b))


@pytest.mark.parametrize("option", [
    {"adaptive_weight_gamma": 2}, {"adaptive_weight_floor": .01},
    {"adaptive_weight_baseline": 2}, {"data": None}, {"eps": None}, {"runtime": None},
])
def test_changed_adaptive_graph_recipe_rejected(monkeypatch, option):
    data, context = _prepared(adaptive=True)
    monkeypatch.setattr(solver, "_fit_from_start", lambda *a, **k: pytest.fail("optimizer entered"))
    with pytest.raises(TypeError):
        _fit(data, context, **option)


def test_equivalent_graph_can_reuse_context(monkeypatch):
    data, context = _prepared()
    def reached_optimizer(*args, **kwargs):
        raise RuntimeError("optimizer boundary reached")
    monkeypatch.setattr(solver, "_fit_from_start", reached_optimizer)
    with pytest.raises(RuntimeError, match="optimizer boundary"):
        _fit(data, context)


def test_corrupted_cached_graph_identity_is_rejected():
    data, context = _prepared()
    with pytest.raises(ValueError, match="identity is inconsistent"):
        _fit(data, replace(context, base_objective_key=replace(
            context.base_objective_key, graph_hash="stale",
        )))


def test_changed_source_data_cannot_reuse_prepared_identity():
    data, context = _prepared()
    changed = replace(data, alt_counts=np.zeros_like(data.alt_counts))
    with pytest.raises(ValueError, match="data fingerprint"):
        _fit(data, replace(context, source_data=changed))


def test_changed_epsilon_cannot_reuse_prepared_identity():
    data, context = _prepared()
    with pytest.raises(TypeError, match="eps"):
        replace(context, eps=.01)
    with pytest.raises(ValueError, match="likelihood or epsilon"):
        _fit(data, replace(context, base_objective_key=replace(
            context.base_objective_key, eps_hex=float(.01).hex(),
        )))


def test_changed_base_objective_is_rejected():
    data, context = _prepared()
    with pytest.raises(ValueError, match="objective identity"):
        _fit(data, replace(context, base_objective_key=replace(
            context.base_objective_key, box_hash="stale",
        )))


def test_prepared_bounds_and_hashes_have_one_authoritative_owner():
    from dataclasses import fields
    _, context = _prepared()
    stored = {field.name for field in fields(context)}
    assert not stored.intersection({"lower", "upper", "eps", "graph_hash", "objective_spec_hash",
                                    "base_fusion_objective_hash"})
    assert context.eps.hex() == context.base_objective_key.eps_hex
    assert context.lower is context.model.lower
    assert context.upper is context.model.upper
    assert context.graph_hash == context.base_objective_key.graph_hash
    assert context.objective_spec_hash == context.base_objective_key.fingerprint
    assert context.base_fusion_objective_hash == context.objective_spec_hash


def test_certificate_options_are_reused_without_changing_dtype_or_profile_gate():
    options = resolve_fit_config(device="cpu", dtype="float64").solver
    certificate = solver._certificate_options(options, torch.float64)
    assert solver._certificate_options(options, torch.float64) is certificate
    assert solver._certificate_options(replace(options), torch.float64) is certificate
    assert certificate.mapping_tolerance == .1 * options.tolerance
    assert certificate.column_tolerance == options.tolerance
    float32 = solver._certificate_options(options, torch.float32)
    assert float32 is not certificate
    assert float32.column_tolerance == certificate.column_tolerance
    deeper = replace(options, tolerance=1e-6, certification_tolerance=options.tolerance)
    assert solver._certificate_options(deeper, torch.float64) == certificate


def test_preparation_resolves_config_once_and_preserves_explicit_frozen_sources(monkeypatch):
    data, context = _prepared()
    options = resolve_fit_config(device="cpu", dtype="float64", inner_max_iter=3,
                                 graph=context.graph_spec, eps=.02)
    requests = []
    def prepare(source, **kwargs):
        assert source is data
        requests.append(kwargs)
        return context
    monkeypatch.setattr(solver, "prepare_torch_problem", prepare)
    result = solver.prepare_torch_problem_with_resource_policy(data, options)
    assert result.fallback_policy == options.runtime.fallback
    initial = requests[-1]
    assert initial["eps"] == .02
    assert initial["inner_max_iter"] == 16
    assert initial["graph"] is context.graph_spec
    assert initial["tol"] == options.solver.tolerance
    solver.prepare_torch_problem_with_resource_policy(
        data, options, graph=None, defer_graph=True, runtime=context.runtime,
        exact_pilot=context.exact_pilot,
    )
    overridden = requests[-1]
    assert overridden["graph"] is None and overridden["defer_graph"]
    assert overridden["runtime"] is context.runtime
    assert "torch_data" not in overridden
    assert overridden["exact_pilot"] is context.exact_pilot


def test_frozen_graph_arrays_cannot_be_made_writable():
    _, context = _prepared()
    for values in (context.graph_spec.edge_u, context.graph_spec.edge_v,
                   context.graph_spec.edge_w):
        with pytest.raises(ValueError):
            values.setflags(write=True)


def test_preparation_preserves_nondefault_epsilon_and_float64_source():
    import torch
    from CliPP2.core.objective import compile_observed_model
    data = integer_data(((2,), (3,)))
    context = solver.prepare_torch_problem(
        data, eps=.02, tol=8e-4, inner_max_iter=16,
        graph=build_complete_uniform_graph(2), device="cpu", dtype="float32",
    )
    assert context.eps == .02
    assert context.source_model is compile_observed_model(data, eps=.02)
    promoted = solver.promote_solver_context_dtype(context, dtype=torch.float64)
    assert promoted.eps == .02
    assert promoted.source_model is context.source_model
    promoted.validate()


@pytest.mark.parametrize("target", ["weights", "counts", "lower", "pilot", "preconditioner"])
def test_prepared_runtime_tensor_mutation_is_rejected_before_optimization(monkeypatch, target):
    data, context = _prepared()
    tensors = {
        "weights": context.graph.weight,
        "counts": context.model.alt,
        "lower": context.lower,
        "pilot": context.exact_pilot,
        "preconditioner": context.graph.pdhg_tau_node,
    }
    tensors[target].add_(1)
    monkeypatch.setattr(solver, "_fit_from_start", lambda *a, **k: pytest.fail("optimizer entered"))
    with pytest.raises(ValueError, match="Prepared runtime tensor.*changed"):
        _fit(data, context)


def test_replacing_runtime_views_cannot_rebaseline_stale_identity():
    _, context = _prepared()
    with pytest.raises(ValueError, match="Prepared runtime tensor graph.weight changed"):
        replace(context, graph=replace(context.graph, weight=context.graph.weight * 10))
    model = context.model
    with pytest.raises(ValueError, match="Prepared runtime tensor model.alt changed"):
        replace(context, model=replace(model, alt=model.alt + 1))


def test_runtime_precision_rebuild_is_bound_to_unchanged_host_sources():
    _, context = _prepared()
    promoted = solver.promote_solver_context_dtype(context, dtype=torch.float32)
    promoted.validate()
    restored = solver.promote_solver_context_dtype(promoted, dtype=torch.float64)
    restored.validate()
    assert restored.source_model is context.source_model
    assert restored.graph_hash == context.graph_hash
    torch.testing.assert_close(restored.graph.weight, context.graph.weight, rtol=0, atol=0)
    for field in ("alt", "nonalt", "lower", "upper", "slope", "log_prior", "valid", "observed"):
        torch.testing.assert_close(getattr(restored.model, field), getattr(context.model, field),
                                   rtol=0, atol=0)
    assert not torch.equal(promoted.model.slope.double(), context.model.slope)
    context.graph.weight.add_(1)
    with pytest.raises(ValueError, match="Prepared runtime tensor graph.weight changed"):
        solver.promote_solver_context_dtype(context, dtype=torch.float32)


def test_audit_tensors_are_rebuilt_from_source_and_cached_per_frozen_context():
    _, context64 = _prepared()
    context32 = solver.promote_solver_context_dtype(context64, dtype=torch.float32)
    audit = solver._float64_audit_context(context32)
    assert solver._float64_audit_context(context32) is audit
    assert audit.model.source_fingerprint == context64.source_model.fingerprint
    torch.testing.assert_close(audit.model.slope, context64.model.slope, rtol=0, atol=0)
    torch.testing.assert_close(audit.graph.weight, context64.graph.weight, rtol=0, atol=0)
    context32.model.alt.add_(1)
    with pytest.raises(ValueError, match="Prepared runtime tensor model.alt changed"):
        solver._float64_audit_context(context32)
