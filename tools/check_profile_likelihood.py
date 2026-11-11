"""Profile fitting / frozen marginal selection qualification; CPU by default."""
import itertools
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom

from CliPP2.config import FitConfig, WorkflowConfig, fitting_identity
from CliPP2.io.data import MultiplicityBounds
from CliPP2.io.reporting import independent_log_columns, _verify_fitted_numerics
from CliPP2.kernel.model import RegionalModel
from CliPP2.kernel.profile import switch_grid
from CliPP2.kernel.solver import ContinuationPolicy, generate_candidates
from CliPP2.kernel.topology import build_tree
from CliPP2.tree.candidates import Candidate, CandidateBank, score_key
from check_support import write_catalog


def rejects(operation, exception=ValueError):
    try:
        operation()
    except exception:
        return
    raise AssertionError("Incompatible fitting/scoring request was accepted")


def run_checks(root, *, device="cpu"):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    catalog = write_catalog(root/"two.tsv", [[4], [2]], 20, [[2], [1]],
                            minor=[[0], [1]], purity=[.5], segments=[["a"], ["b"]])
    data = catalog.fit_data()
    labels = np.zeros(2, dtype=int)
    report = {}
    with RegionalModel(data, device, cpu_threads=1, fit_multiplicity="profile") as model, \
            RegionalModel(data, device, cpu_threads=1) as marginal:
        assert model.identity != marginal.identity
        assert model.fit_backend is not model.selection_backend
        assert marginal.fit_backend is marginal.selection_backend
        for mode, expected in (("profile", [-2.776981342107, -2.830014656748, -3.175636915754]),
                               ("marginal", [-3.125508756242, -3.087040250270, -3.328718159493])):
            backend = model.fit_backend if mode == "profile" else model.selection_backend
            actual = []
            for center, expected_ll in zip((.4, .45, .6), expected):
                x = np.full((2, 1), center)
                ll = -backend.node_terms(x, False).sum()
                np.testing.assert_allclose(ll, expected_ll, atol=6e-13, rtol=0)
                np.testing.assert_allclose(backend.regional_grid(0, np.arange(2), [center]).sum(), ll, atol=1e-13)
                np.testing.assert_allclose(backend.conditional_values([(np.arange(2), 0)], [center]), [ll], atol=1e-13)
                np.testing.assert_allclose(backend.joint_columns([[center]]).sum(), ll, atol=1e-13)
                np.testing.assert_allclose(independent_log_columns(model, [[center]], aggregation=mode).sum(), ll, atol=1e-12)
                actual.append(float(ll))
            report[mode+"_at_04_045_06"] = actual
        assert model.fit_backend.aggregation == "profile"
        diag = model.fit_backend.component_diagnostics(np.full((2, 1), .4))
        np.testing.assert_allclose(diag["log_emission"].sum(), report["marginal_at_04_045_06"][0], atol=1e-13)
        np.testing.assert_allclose(diag["max_component_log_mass"].sum(), report["profile_at_04_045_06"][0], atol=1e-13)
        fits = [model.refit(labels, [[start]]) for start in (.39, .61)]
        for fit in fits:
            np.testing.assert_allclose(fit.centers, [[.4]], rtol=0, atol=2e-12)
            np.testing.assert_allclose(fit.fitting_log_likelihood, -2.776981342107, rtol=0, atol=6e-13)
            assert not fit.centers.flags.writeable
            _verify_fitted_numerics(model, fit)
        np.testing.assert_array_equal(fits[0].centers, fits[1].centers)
        with patch.object(model, "_fit_coordinates", side_effect=AssertionError("scoring refitted centers")):
            at45 = model.score_frozen_centers([[.45]], labels)
            at40 = model.score_frozen_centers(fits[0].centers, labels)
        assert at40["score"] > at45["score"]
        np.testing.assert_allclose(at40["score"]-at45["score"], .076937011944, atol=2e-12, rtol=0)
        # Ranking must prefer the marginal BIC, even against a better profile fit.
        other = replace(fits[0], centers=np.array([[.45]]), score=at45["score"],
                        fitting_log_likelihood=report["profile_at_04_045_06"][1])
        candidates = [Candidate(fit, state_index=i) for i, fit in enumerate((fits[0], other))]
        assert min(candidates, key=score_key).fit is other
        for c in (.4, .45, .6):
            first = model.score_frozen_centers([[c]], labels)
            second = marginal.score_frozen_centers([[c]], labels)
            for key in ("score", "weights", "conditional_log_likelihood", "mixture_log_likelihood"):
                np.testing.assert_array_equal(first[key], second[key])
            selected = model.selection_column_set([[c]])
            profiled = model.fit_column_set([[c]])
            rejects(lambda: model.selection_column_set([[c]], retained_columns=profiled))
            rejects(lambda: model._weight_result(profiled, [2]))
            np.testing.assert_array_equal(selected.materialize(), model.selection_column_set([[c]], retained_columns=selected).materialize())
        x = np.full((2, 1), .4)
        loss, gradient, curvature = model.node_terms(x)
        h = 1e-5
        finite_difference = (model.loss(x+h)-model.loss(x-h))/(2*h)
        np.testing.assert_allclose(gradient, finite_difference, atol=5e-8, rtol=1e-7)
        second_difference = (model.loss(x+h)-2*loss+model.loss(x-h))/(h*h)
        np.testing.assert_allclose(curvature, second_difference, atol=1e-4, rtol=1e-5)
        pilots, _ = model.pilots()
        np.testing.assert_array_equal(pilots, marginal.pilots()[0])
        tree = build_tree(pilots, model.observed, model.mutation_ids)
        bank = CandidateBank(model, tree)
        parent_fit = model.refit(np.array([0, 1]))
        parent = bank.prepare(None, np.array([0, 1]), fitted=parent_fit,
                              columns=model.selection_column_set(parent_fit.centers))
        child = bank.prepare(None, labels, parent=parent)
        refitted = model.refit_prepared(child)
        np.testing.assert_array_equal(refitted.centers, fits[0].centers)
        rejects(lambda: bank.prepare(None, labels, columns=model.fit_column_set([[.4]])))
        policy = ContinuationPolicy(levels=2, iterations_per_level=12)
        native = generate_candidates(model, tree, pilots, [1, 2], policy=policy)
        host = generate_candidates(model, tree, pilots, [1, 2], policy=policy, native=False)
        for a, b in zip(native, host):
            np.testing.assert_array_equal(a["labels"], b["labels"])
            np.testing.assert_allclose(a["raw_phi"], b["raw_phi"], atol=2e-10, rtol=1e-9)
        from CliPP2.kernel.continuation_requests import continue_requests
        lo, hi = np.full_like(pilots, 1e-8), np.full_like(pilots, 1-1e-8)
        requested = continue_requests(model, tree, np.clip(pilots, lo, hi), [1, 2], lo, hi, policy)
        for a, b in zip(native, requested):
            np.testing.assert_array_equal(a["labels"], b["labels"])
            np.testing.assert_allclose(a["raw_phi"], b["raw_phi"], atol=2e-10, rtol=1e-9)
        report["native"] = model.execution_diagnostics()

    support = MultiplicityBounds(catalog.mutation_ids, catalog.region_ids, [[1], [1]], [[1], [1]], "singleton_test")
    with RegionalModel(data, device, cpu_threads=1, multiplicity_support=support) as a, \
            RegionalModel(data, device, cpu_threads=1, multiplicity_support=support, fit_multiplicity="profile") as b:
        np.testing.assert_allclose(a.refit(labels).centers, [[.6]], atol=1e-8)
        np.testing.assert_allclose(b.refit(labels).centers, [[.6]], atol=2e-12)
        np.testing.assert_array_equal(a.loss([[.6], [.6]]), b.loss([[.6], [.6]]))
    switches = switch_grid(4, 20, .25, 1, 2, 0, 1)
    assert np.min(abs(switches-.5490573468672738)) < 1e-15
    # Exhaustive branch-assignment oracle on small, heterogeneous supports.
    rng = np.random.default_rng(193)
    for index in range(12):
        n = 3
        y = rng.integers(0, 21, size=(n, 1))
        major = rng.integers(1, 4, size=(n, 1))
        cat = write_catalog(root/f"random{index}.tsv", y, 20, major, minor=0, purity=[.7],
                            segments=np.arange(n)[:, None].astype(str))
        with RegionalModel(cat.fit_data(), device, cpu_threads=1, fit_multiplicity="profile") as model:
            fit = model.refit(np.zeros(n, dtype=int))
            oracle = -np.inf
            for ms in itertools.product(*(range(1, int(m)+1) for m in major[:, 0])):
                scales = model.scaling[:, 0]*ms
                def derivative(c):
                    return np.sum(y[:, 0]/c-(20-y[:, 0])*scales/(1-scales*c))
                points = [0., 1.]
                if derivative(1e-14) > 0 and derivative(1-1e-14) < 0:
                    points.append(brentq(derivative, 1e-14, 1-1e-14, xtol=1e-14))
                oracle = max(oracle, *(float(binom.logpmf(y[:, 0], 20, scales*c).sum()) for c in points))
            np.testing.assert_allclose(fit.fitting_log_likelihood, oracle, atol=5e-11, rtol=0)

    for name, y, phi in (("zero", 0, 0.), ("all", 20, 1.), ("impossible", 1, 0.)):
        cat = write_catalog(root/f"{name}.tsv", [[y], [2]], 20, 2, minor=0, purity=[1.])
        with RegionalModel(cat.fit_data(), device, cpu_threads=1, fit_multiplicity="profile") as model:
            loss = model.loss([[phi], [.4]])
            assert np.isposinf(loss[0, 0]) if name == "impossible" else np.isfinite(loss[0, 0])
            if name != "impossible":
                np.testing.assert_allclose(loss[0, 0], -binom.logpmf(y, 20, phi), atol=1e-12)
    cat = write_catalog(root/"mask.tsv", [[4, 0], [2, 1]], 20, 2, minor=0,
                        observed=[[True, False], [True, True]], purity=[.5, .7])
    with RegionalModel(cat.fit_data(), device, cpu_threads=1, fit_multiplicity="profile") as model:
        assert model.loss(np.ones((2, 2))*.4)[0, 1] == 0
    # Retained single-move paths keep marginal selection columns and reuse only
    # unchanged profile coordinates, even after fitting columns interleave.
    cat = write_catalog(root/"parent.tsv", [[3], [5], [7], [9], [11], [13]], 40, 3)
    with RegionalModel(cat.fit_data(), device, cpu_threads=1, fit_multiplicity="profile") as model:
        pilots, _ = model.pilots()
        tree = build_tree(pilots, model.observed, model.mutation_ids)
        bank = CandidateBank(model, tree)
        parent_fit = model.refit(np.repeat(np.arange(3), 2))
        parent = bank.prepare(None, parent_fit.labels, fitted=parent_fit,
                              columns=model.selection_column_set(parent_fit.centers))
        model.fit_column_set(parent_fit.centers)
        moved, initial, reason = bank.prepare_single_move(parent, 1, 1, parent_fit.centers)
        assert reason is None and moved.selection_columns.aggregation == "marginal"
        reused = model.telemetry["unchanged_coordinate_reuses"]
        answer = model.refit_prepared(moved, initial)
        assert model.telemetry["unchanged_coordinate_reuses"] > reused
        fresh = model.refit(moved.labels)
        np.testing.assert_array_equal(answer.centers, fresh.centers)
        np.testing.assert_array_equal(answer.weights, fresh.weights)
        assert answer.score == fresh.score and answer.fitting_log_likelihood == fresh.fitting_log_likelihood
    # At exact endpoint ties, use feasible one-sided slopes, not an averaged
    # derivative that could hide descent. Flat maxima choose the smallest CCF.
    from CliPP2.kernel.solver import _stationarity
    cat = write_catalog(root/"ties.tsv", [[0], [20]], 20, 2, minor=0)
    with RegionalModel(cat.fit_data(), device, cpu_threads=1, fit_multiplicity="profile") as model:
        x = np.array([[0.], [1.]])
        right, left = model.fit_backend.profile_slopes(x)
        np.testing.assert_array_equal(right, [[10.], [0.]])
        np.testing.assert_array_equal(left, [[20.], [-20.]])
        _, gradient, _ = model.node_terms(x)
        assert _stationarity(model, x, gradient, np.zeros_like(x), model.lower, model.upper) == 0
        np.testing.assert_array_equal(model.refit(np.array([0, 1])).centers, x)
    assert WorkflowConfig(fit=FitConfig(fit_multiplicity="profile")).fit.fit_multiplicity == "profile"
    rejects(lambda: FitConfig(fit_multiplicity="invalid"))
    rejects(lambda: FitConfig(selection_criterion="profile_bic"))
    assert fitting_identity("profile") != fitting_identity("marginal")
    from CliPP2.simulation.evaluation import aggregate
    rejects(lambda: aggregate([{"fitting_method_identity": fitting_identity("profile")},
                                {"fitting_method_identity": fitting_identity("marginal")}]))
    from compare_profile_fitting import compare_data, compare_workflow
    comparison_config = FitConfig(device=device, max_clusters=2, cpu_threads=1)
    fixed = compare_data(data, root/"fixed_centers", comparison_config, scope="fixed-centers",
                         states=[dict(labels=[0, 0], centers=[[.4]]), dict(labels=[0, 0], centers=[[.45]])])
    assert len(fixed["comparisons"]) == 2
    compared = compare_data(data, root/"complete_search", comparison_config)
    assert compared["searches"]["profile"]["tree_identity"] == compared["searches"]["marginal"]["tree_identity"]
    separate = compare_data(data, root/"fixed_partitions", comparison_config, scope="fixed-partitions",
                            partitions=[[0, 0], [0, 1]])
    assert len(separate["comparisons"]) == 2
    # Actual two-stage workflow, same source/remaining population for isolated
    # Stage 2 first, then end-to-end (Stage 1 may change the caps).
    workflow_input = root/"workflow.tsv"
    write_catalog(workflow_input, [[50, 50]]*3+[[20, 15]]*3, 100, 1, minor=1)
    workflow_config = WorkflowConfig(fit=FitConfig(device=device, max_clusters=2, cpu_threads=1),
                                     stage_two_max_clusters=2)
    workflows = compare_workflow(workflow_input, root/"workflows", workflow_config)
    assert all(w["status"] == "complete" for w in workflows["workflows"].values())
    from CliPP2.io.assignment import load_accepted_fit
    from CliPP2.workflow import AcceptedFit, run_workflow
    from CliPP2.provenance import validate_replay_policy
    accepted = load_accepted_fit(root/"workflows/profile/stage_one_accepted.json")
    accepted_two = load_accepted_fit(root/"workflows/profile/stage_two_accepted.json")
    assert AcceptedFit.from_dict(accepted.to_dict()).state_fingerprint == accepted.state_fingerprint
    rejects(lambda: validate_replay_policy(accepted, workflow_config.fit, normal_cn_policy=None))
    from CliPP2.provenance import validate_fit_execution_receipt
    producer = accepted.to_dict()["producer_receipt"]
    producer["execution"]["selection_native"]["build_id"] = "different-build"
    rejects(lambda: validate_fit_execution_receipt(producer["fit_config"], producer["execution"]))
    replay_config = replace(workflow_config, fit=replace(workflow_config.fit, fit_multiplicity="profile"))
    with patch("CliPP2.workflow._fit_accepted", side_effect=AssertionError("Replay started a fit")):
        replay = run_workflow(workflow_input, root/"profile_replay", replay_config,
                             accepted_stage_one=accepted, accepted_stage_two=accepted_two)
    assert replay["status"] == "complete"
    original = load_accepted_fit(root/"workflows/marginal/stage_one_accepted.json")
    rejects(lambda: validate_replay_policy(original, replay_config.fit, normal_cn_policy=None))
    report["checks"] = "normalization_derivatives_global_branches_frozen_selection_cache_replay_continuation_masks"
    report["device"] = device
    (root/"checks.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    run_checks(args.outdir, device=args.device)
    print("Profile likelihood checks passed")
