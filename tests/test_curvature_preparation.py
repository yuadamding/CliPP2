"""Curvature reuses one source-checked runtime without rebuilding its model."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from CliPP2.config import resolve_fit_config
from CliPP2.core.fusion import partition_starts as partitions, solver
from CliPP2.model_selection.proposals import build_partition_guided_graph_with_resource_policy
from test_integer_likelihood import EPS, integer_data


def _context(data, *, dtype="float64", eps=EPS, derive_pilot=False):
    starts = {} if derive_pilot else dict(
        exact_pilot=data.phi_init, pooled_start=data.phi_init, scalar_well_starts=(),
    )
    return solver.prepare_torch_problem_with_resource_policy(
        data, resolve_fit_config(device="cpu", dtype=dtype, eps=eps),
        defer_graph=True, graph=None, **starts,
    )


# Independent aa799344 capture, before replacing the runtime reconstruction.
_PINNED = {
    "float32": {
        "pilot": [[.342848539352417, .5402541160583496], [.4790276288986206, .6036857962608337],
                  [.8583443760871887, .3136019706726074], [.12461940199136734, .9380341172218323],
                  [.9642945528030396, .5]],
        "curvature": [[122.93931579589844, 130.3901824951172], [55.881507873535156, 44.70520782470703],
                      [163.91909790039062, 171.36996459960938], [238.96022033691406, 171.36996459960938],
                      [119.21388244628906, 9.999999974752427e-7]],
        "weights": [.6417339002659994, .17119914931608018, .21248395515454804, .10969484459065343,
                    .20188720644586516, .1978653558785519, .14047819974718534, .10006172104313406,
                    .6434101950558482, .0811854725021351],
        "graph_hash": "dede0ad7d994fe9a3ee7e7c720bb3c70d599ac273b86f76e033401435d6bbfb7",
        "tau": .041469332601100466,
    },
    "float64": {
        "pilot": [[.3428485516828756, .5402540987505273], [.4790276266754733, .6036857682124577],
                  [.8583444003051608, .31360196011495195], [.1246194056549072, .9380341195831298],
                  [.9642945713152193, .5]],
        "curvature": [[116.00924421789584, 132.0211905284082], [59.338750670045854, 53.131208169188],
                      [151.51787968736798, 166.67261666658504], [240.0329614587499, 195.9279539859299],
                      [107.54393535428146, 1e-6]],
        "weights": [.6417339565574631, .17119914164845745, .2124839386905561, .10969484017706482,
                    .20188719594351354, .19786534315010118, .14047818948980925, .10006171551723217,
                    .6434102102182663, .0811854686075362],
        "graph_hash": "2fe56227ef5a8512d700b8579ee78f26387b66df9e52b460f5e594118cd487c1",
        "tau": .0419794141165389,
    },
}
_PARTITIONS = {
    2: ([0, 0, 1, 0, 1], 508.41661250594365, 1030.3837996555592,
        [[.41198789270019526, .9566040473022459], [.9020997072753906, .31274482788085933]]),
    3: ([0, 0, 1, 2, 1], 506.73583326082553, 1033.1705584758888,
        [[.41784726184082027, .5650639114990235], [.9022217774658203, .3132331086425781],
         [.12414638366699217, .9375000625]]),
    5: ([0, 1, 2, 3, 4], 505.8060038498855, 1040.3550481428654,
        [[.34375065625, .541016083984375], [.4789433921508789, .6038211969604494],
         [.8583985791015626, .3131110384521484], [.12414638366699217, .9379883432617186],
         [.9648437851562499, .5000005]]),
    1: ([0, 0, 0, 0, 0], 519.9868079443526, 1044.3680650433776,
        [[.7913210094604493, .9041138654174805]]),
}


def _prepared_case(dtype):
    data = replace(
        integer_data(((1, 2), (4, 6), (3, 2), (6, 4), (2, 5)),
                     observed=np.array([[1, 1]] * 4 + [[1, 0]], dtype=bool)),
        alt_counts=np.array([[12., 28.], [30., 42.], [53., 16.], [7., 64.], [50., 36.]]),
    )
    return _context(data, dtype=dtype, derive_pilot=True)


@pytest.fixture(params=["float32", "float64"])
def prepared_case(request):
    return _prepared_case(request.param), _PINNED[request.param]


def _curvature(context):
    return partitions.observed_curvature_at_pilot_torch(
        context.model, context.exact_pilot, eps=context.eps,
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_curvature_and_proposals_construct_and_use_only_one_runtime(monkeypatch, dtype):
    models, evaluated = [], []
    construct, evaluate = solver.model_to_torch, partitions.observed_loss_grid_torch

    def construct_once(*args, **kwargs):
        model = construct(*args, **kwargs)
        models.append(model)
        return model

    def evaluate_supplied(model, *args, **kwargs):
        evaluated.append(model)
        return evaluate(model, *args, **kwargs)

    monkeypatch.setattr(solver, "model_to_torch", construct_once)
    monkeypatch.setattr(partitions, "observed_loss_grid_torch", evaluate_supplied)
    context = _prepared_case(dtype)
    options = resolve_fit_config(device="cpu", dtype=dtype)
    curvature = _curvature(context)
    for phi, supplied_curvature in (
        (context.exact_pilot, curvature), (context.exact_pilot.cpu().numpy() * .9, None),
    ):
        partitions.generate_partition_initializer_pool(
            context=context, pilot_phi=phi, fit_options=options,
            curvature=supplied_curvature, declared_k_grid=(1, 2, 3, 5),
        )
    assert models == [context.model]
    assert len(evaluated) == 6 and all(model is context.model for model in evaluated)


def test_curvature_reuse_matches_reconstruction_exactly_in_same_environment(monkeypatch, prepared_case):
    context, _ = prepared_case
    rebuilt = solver.model_to_torch(context.source_model, context.runtime, eps=context.eps)
    assert rebuilt is not context.model
    assert rebuilt.source_fingerprint == context.model.source_fingerprint
    captures = []
    evaluate, quantile = partitions.observed_loss_grid_torch, torch.quantile
    for model in (context.model, rebuilt):
        capture = []

        def record_loss(supplied, phi, **kwargs):
            assert supplied is model
            loss = evaluate(supplied, phi, **kwargs)
            capture.extend((phi.clone(), loss.clone()))
            return loss

        def record_quantile(values, *args, **kwargs):
            cap = quantile(values, *args, **kwargs)
            capture.extend((values.clone(), cap.clone()))
            return cap

        with monkeypatch.context() as patch:
            patch.setattr(partitions, "observed_loss_grid_torch", record_loss)
            patch.setattr(torch, "quantile", record_quantile)
            capture.append(partitions.observed_curvature_at_pilot_torch(
                model, context.exact_pilot, eps=context.eps,
            ))
        # Left/center/right arguments and losses, pre-cap values, cap, result.
        assert len(capture) == 9
        captures.append(capture)
    for reused, reconstructed in zip(*captures, strict=True):
        torch.testing.assert_close(reused, reconstructed, rtol=0, atol=0)


def test_preparation_pilot_and_source_identity_match_reference(prepared_case):
    context, expected = prepared_case
    assert context.exact_pilot.tolist() == expected["pilot"]
    assert context.source_model.fingerprint == "5367ac6527734b7c7e01a629ebff3237fbfddd353cc9fdddbafe4baba3c71354"


def test_portable_curvature_reference_with_one_working_dtype_ulp(prepared_case):
    context, expected = prepared_case
    curvature = _curvature(context).cpu().numpy()
    # The measured torch 2.9.1/2.14.0 CPU matrix differs only at the float32
    # quantile cap (one ULP). Keep the old golden and bound only this numerical
    # comparison; labels, scores and identities have independent exact tests.
    np.testing.assert_array_max_ulp(
        curvature, np.asarray(expected["curvature"], dtype=curvature.dtype), maxulp=1,
    )


def test_ward_labels_match_reference_independently_of_curvature_golden(prepared_case):
    context, _ = prepared_case
    ward = partitions.hessian_weighted_ward_label_sets_torch(
        context.exact_pilot, _curvature(context), K_grid=(1, 2, 3, 5),
    )
    assert {k: labels.tolist() for k, labels in ward.items()} == {
        k: values[0] for k, values in _PARTITIONS.items()
    }


def test_proposal_labels_order_families_refits_and_scores_match_reference(prepared_case):
    context, _ = prepared_case
    options = resolve_fit_config(device="cpu", dtype=str(context.runtime.dtype).removeprefix("torch."))
    for phi, supplied_curvature in (
        (context.exact_pilot, _curvature(context)), (context.exact_pilot.cpu().numpy() * .9, None),
    ):
        pool = partitions.generate_partition_initializer_pool(
            context=context, pilot_phi=phi, fit_options=options,
            curvature=supplied_curvature, declared_k_grid=(1, 2, 3, 5),
        )
        assert [candidate.K for candidate in pool] == [2, 3, 5, 1]
        for candidate in pool:
            labels, loss, bic, centers = _PARTITIONS[candidate.K]
            assert candidate.labels.tolist() == labels
            np.testing.assert_array_equal(candidate.phi_start, np.asarray(centers)[labels])
            assert candidate.fit_loss == loss and candidate.bic == bic
            assert candidate.source == f"hessian_ward_K{candidate.K}"
            assert candidate.requested_k == candidate.K and candidate.component_death_count == 0
            assert candidate.finite_candidate_found


def test_graph_weights_and_identity_match_qualified_reference_environments(prepared_case):
    context, expected = prepared_case
    options = resolve_fit_config(device="cpu", dtype=str(context.runtime.dtype).removeprefix("torch."))
    graph, tensor_graph, tau = build_partition_guided_graph_with_resource_policy(
        guide_phi=context.exact_pilot, guide_curvature=_curvature(context),
        solver_context=context, fit_options=options, noise_divisor=4**1.05,
    )
    assert tensor_graph is None and tau == expected["tau"]
    assert graph.edge_w.tolist() == expected["weights"]
    assert graph.fingerprint == expected["graph_hash"]
    replay, _, replay_tau = build_partition_guided_graph_with_resource_policy(
        guide_phi=context.exact_pilot, guide_curvature=_curvature(context),
        solver_context=context, fit_options=options, noise_divisor=4**1.05,
    )
    assert replay.fingerprint == graph.fingerprint and replay_tau == tau
    np.testing.assert_array_equal(replay.edge_w, graph.edge_w)


@pytest.mark.parametrize("change", ["numpy", "shape", "dtype", "device"])
def test_curvature_rejects_competing_or_mismatched_pilot_before_evaluation(monkeypatch, change):
    context = _context(integer_data())
    phi = context.exact_pilot
    if change == "numpy":
        phi = phi.numpy()
    elif change == "shape":
        phi = phi[..., None]
    elif change == "dtype":
        phi = phi.float()
    else:
        phi = torch.empty_like(phi, device="meta")
    monkeypatch.setattr(partitions, "observed_loss_grid_torch", lambda *a, **k: pytest.fail("evaluated"))
    with pytest.raises((TypeError, ValueError), match="Tensor|shape|dtype and device"):
        partitions.observed_curvature_at_pilot_torch(context.model, phi, eps=context.eps)


@pytest.mark.parametrize("field", [
    "alt", "nonalt", "observed", "lower", "upper", "slope", "log_prior", "valid",
    "exact_pilot", "pooled_start",
])
def test_proposal_boundary_rejects_edited_prepared_tensors(monkeypatch, field):
    context = _context(integer_data())
    tensor = getattr(context, field) if field.endswith("pilot") or field == "pooled_start" else getattr(context.model, field)
    if tensor.dtype == torch.bool:
        tensor.logical_not_()
    else:
        tensor.add_(1)
    monkeypatch.setattr(partitions, "observed_curvature_at_pilot_torch", lambda *a, **k: pytest.fail("evaluated"))
    with pytest.raises(ValueError, match="Prepared runtime tensor.*changed"):
        partitions.generate_partition_initializer_pool(
            context=context, pilot_phi=context.exact_pilot,
            fit_options=resolve_fit_config(device="cpu", dtype="float64"),
        )


def test_deferred_context_is_proposal_only_and_fit_still_fails_closed(monkeypatch):
    context = _context(integer_data())
    context.validate(allow_deferred_graph=True)
    monkeypatch.setattr(solver, "_fit_from_start", lambda *a, **k: pytest.fail("optimizer entered"))
    with pytest.raises(ValueError, match="deferred likelihood pilot"):
        solver.fit_prepared(context, .1, resolve_fit_config(device="cpu").solver)
