"""Joint-pilot, group-refit and search integration checks against native kernels."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.stats import binom

from CliPP2.io.data import MultiplicityBounds
from CliPP2.io.reporting import independent_log_columns, _verify_fitted_numerics
from CliPP2.kernel.initialization import pooled_ccf_initialization
from CliPP2.kernel.joint import joint_components
from CliPP2.kernel.model import RegionalModel
from CliPP2.kernel.solver import ContinuationPolicy, generate_candidates
from CliPP2.kernel.topology import build_tree
from CliPP2.tree.candidates import CandidateBank
from CliPP2.tree.selection import fit_regions, _likelihood_ranked_edges

from check_support import write_catalog


def _pilot_oracle(model):
    """Explicit raw component/grid integration, independent of native pilot DP."""
    masses = [[None] * model.r for _ in range(model.n)]
    means = [[None] * model.r for _ in range(model.n)]
    pooled = np.full((model.n, model.r), np.nan)
    for r in range(model.r):
        rows = np.flatnonzero(model.observed[:, r])
        pilot, _, grid, weights = pooled_ccf_initialization(model.pilot_view(r, rows), return_pool=True)
        pooled[rows, r] = pilot
        active = weights > 0
        for i in rows:
            m = np.arange(model.multiplicity_min[i, r], model.multiplicity_max[i, r] + 1)
            probability = np.minimum(1., model.scaling[i, r] * m[:, None] * grid[None, active])
            values = binom.logpmf(model.alt[i, r], model.depth[i, r], probability) + np.log(weights[active])
            log_mass = logsumexp(values, axis=1)
            masses[i][r] = log_mass
            finite = np.isfinite(log_mass)
            mean = np.zeros(len(m))
            mean[finite] = np.exp(values[finite] - log_mass[finite, None]) @ grid[active]
            means[i][r] = mean
    expected = np.full_like(pooled, np.nan)
    for i in range(model.n):
        for r in np.flatnonzero(~model.observed[i]):
            masses[i][r], means[i][r] = np.zeros(1), np.zeros(1)
        state = joint_components(masses[i], model.multiplicity_min[i], observed=model.observed[i])
        if np.isfinite(state["log_marginal"]):
            for r in np.flatnonzero(model.observed[i]):
                expected[i, r] = state["probabilities"][r] @ means[i][r]
        else:
            expected[i] = (pooled[i] + .5) / 2
    return expected, pooled


def _verify_group_stationarity(model, fitted):
    """Check fitted vectors through the separate public derivative interface."""
    coordinates = fitted.centers[fitted.labels]
    if model.fit_multiplicity == "profile":
        right, left = model.backend.profile_slopes(coordinates)
    else:
        _, gradient, _ = model.node_terms(coordinates)
    for k, center in enumerate(fitted.centers):
        rows = fitted.labels == k
        lo, hi = model.lower[rows].max(axis=0), model.upper[rows].min(axis=0)
        if model.fit_multiplicity == "profile":
            inward_right = np.where(center < hi, -right[rows].sum(axis=0), 0.)
            inward_left = np.where(center > lo, left[rows].sum(axis=0), 0.)
            residual = max(0., inward_right.max(), inward_left.max())
        else:
            g = gradient[rows].sum(axis=0)
            g[((center <= lo) & (g >= 0)) | ((center >= hi) & (g <= 0))] = 0.
            residual = np.max(np.abs(g))
        residual /= 1 + int(rows.sum())
        assert residual <= 1.01e-5, (model.fit_multiplicity, center, residual)
        np.testing.assert_allclose(fitted.group_fits[k].projected_gradient, residual,
                                   atol=2e-9, rtol=2e-7)


def _conditional_profile_oracle(model, fixed):
    """Enumerate every branch pair, independently of the native ordered stack."""
    components, breaks = [], [0., 1.]
    for row in range(model.n):
        m = np.arange(model.multiplicity_min[row, 0], model.multiplicity_max[row, 0] + 1)
        other = np.arange(model.multiplicity_min[row, 1], model.multiplicity_max[row, 1] + 1)
        other_scores = binom.logpmf(model.alt[row, 1], model.depth[row, 1],
                                   model.scaling[row, 1] * other * fixed)
        offsets = np.array([other_scores[np.abs(other - value) <= 1].max(initial=-np.inf)
                            for value in m])
        active = np.isfinite(offsets)
        m, offsets = m[active], offsets[active]
        alt, depth, scale = model.alt[row, 0], model.depth[row, 0], model.scaling[row, 0]
        components.append((alt, depth, scale, m, offsets))
        if depth > alt:
            for a in range(len(m)):
                for b in range(a + 1, len(m)):
                    small, large = m[a], m[b]
                    z = (alt * np.log(large / small) + offsets[b] - offsets[a]) / (depth - alt)
                    if z > 0:
                        t = -np.expm1(-z)
                        crossing = t / (scale * ((large - small) + small * t))
                        if 0 < crossing < 1:
                            breaks.append(crossing)
    breaks = np.unique(breaks)
    def score(phi):
        return sum(np.max(binom.logpmf(alt, depth, scale * m * phi) + offsets)
                   for alt, depth, scale, m, offsets in components)
    candidates = list(breaks)
    for left, right in zip(breaks[:-1], breaks[1:]):
        middle = (left + right) / 2
        branches = [(alt, depth, scale * m[np.argmax(
            binom.logpmf(alt, depth, scale * m * middle) + offsets)])
            for alt, depth, scale, m, offsets in components]
        def derivative(phi):
            return sum((alt / phi if alt else 0.) -
                       ((depth - alt) * slope / (1 - slope * phi) if depth > alt else 0.)
                       for alt, depth, slope in branches)
        low, high = np.nextafter(left, right), np.nextafter(right, left)
        with np.errstate(over="ignore", divide="ignore"):
            if low < high and derivative(low) > 0 and derivative(high) < 0:
                candidates.append(brentq(derivative, low, high, xtol=1e-14, rtol=1e-14))
    values = [score(phi) for phi in candidates]
    return max(values), len(breaks)


def _profile_envelope_checks(directory, device):
    rng = np.random.default_rng(9159)
    largest, switches = 0., 0
    for case in range(32):
        major = rng.integers(2, 6, size=(3, 2))
        alt = rng.integers(0, 41, size=(3, 2))
        if case == 0:
            alt[:, 0] = 0
        elif case == 1:
            alt[:, 0] = 40
        fixed = .23 + .06 * (case % 7)
        catalog = write_catalog(directory / f"profile_envelope_{case}.tsv", alt, 40, major,
                                segments=np.array([[f"s{i}"] for i in range(3)]))
        with RegionalModel(catalog.fit_data(), device, cpu_threads=1, fit_multiplicity="profile") as model:
            expected, count = _conditional_profile_oracle(model, fixed)
            fitted = model.backend.refit_groups([np.arange(3)], [[.73, fixed]],
                lower=[[0., fixed]], upper=[[1., fixed]])
            assert fitted["status"].item() == 0, (case, fitted)
            actual = fitted["log_likelihood"].item()
            largest = max(largest, abs(actual - expected))
            switches += count - 2
            np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-9)
            np.testing.assert_allclose(fitted["centers"][0, 1], fixed, atol=0, rtol=0)
    assert switches > 0
    return dict(random_cases=32, zero_and_all_alt=True, all_pair_crossings=switches,
                maximum_score_error=largest)


def run_checks(directory, *, device="cpu"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    catalog = write_catalog(directory / "pilots.tsv",
        [[6, 45], [8, 50], [25, 10], [30, 12]], 100, [2, 4])
    report = {}
    report["conditional_profile_envelopes"] = _profile_envelope_checks(directory, device)
    with RegionalModel(catalog.fit_data(), device, cpu_threads=1) as model:
        expected, independent = _pilot_oracle(model)
        pilot, diagnostic = model.pilots()
        np.testing.assert_allclose(pilot, expected, atol=2e-10, rtol=2e-10)
        assert np.max(np.abs(pilot-independent)) > 1e-6
        assert diagnostic["joint"]["method"] == "joint_pooled_posterior_pilot_v1"
        seen = []
        def capture_tree(values, *args, **kwargs):
            seen.append(np.array(values))
            return build_tree(values, *args, **kwargs)
        with patch("CliPP2.tree.selection.build_tree", side_effect=capture_tree):
            result = fit_regions(model, max_clusters=2)
        assert len(seen) == 1
        np.testing.assert_array_equal(seen[0], pilot)
        np.testing.assert_array_equal(result.pilot, pilot)
        _verify_fitted_numerics(model, result.selected.fit)
        tree = result.tree
        edges, ranks = _likelihood_ranked_edges(model, tree, list(range(len(tree.edges))),
                                                candidate_centers=result.centers)
        assert sorted(edges) == list(range(len(tree.edges)))
        assert ranks["method"] == "joint_center_vector_gain_v1"
        policy = ContinuationPolicy(levels=2, iterations_per_level=8)
        native = generate_candidates(model, tree, pilot, [1, 2], policy=policy)
        python = generate_candidates(model, tree, pilot, [1, 2], policy=policy, native=False)
        for a, b in zip(native, python):
            np.testing.assert_array_equal(a["labels"], b["labels"])
            np.testing.assert_allclose(a["raw_phi"], b["raw_phi"], atol=3e-9, rtol=3e-9)
        from CliPP2.kernel.continuation_requests import continue_requests
        lo, hi = np.full_like(pilot, 1e-8), np.full_like(pilot, 1-1e-8)
        requested = continue_requests(model, tree, np.clip(pilot, lo, hi), [1, 2], lo, hi, policy)
        for a, b in zip(native, requested):
            np.testing.assert_array_equal(a["labels"], b["labels"])
            np.testing.assert_allclose(a["raw_phi"], b["raw_phi"], atol=3e-9, rtol=3e-9)
        report["pilots_and_search"] = dict(posterior_matches_oracle=True, tree_receives_joint_pilots=True,
            joint_split_ranking=True, native_python_requested_continuation_agree=True,
            maximum_pilot_change=float(np.max(np.abs(pilot-independent))))

    with RegionalModel(catalog.fit_data(), device, cpu_threads=1, fit_multiplicity="profile") as model:
        result = fit_regions(model, max_clusters=2)
        _verify_fitted_numerics(model, result.selected.fit)
        _verify_group_stationarity(model, result.selected.fit)
        pilot, tree = result.pilot, result.tree
        native = generate_candidates(model, tree, pilot, [1, 2], policy=policy)
        python = generate_candidates(model, tree, pilot, [1, 2], policy=policy, native=False)
        requested = continue_requests(model, tree, np.clip(pilot, lo, hi), [1, 2], lo, hi, policy)
        for batch in (python, requested):
            for a, b in zip(native, batch):
                np.testing.assert_array_equal(a["labels"], b["labels"])
                np.testing.assert_allclose(a["raw_phi"], b["raw_phi"], atol=3e-9, rtol=3e-9)
        report["profile_search"] = dict(joint_fitting_marginal_selection_verified=True,
            native_python_requested_continuation_agree=True)

    endpoint = write_catalog(directory / "boundary.tsv", [[1]], 2, 2)
    with RegionalModel(endpoint.fit_data(), device, cpu_threads=1) as model:
        fitted = model.refit([0], [[1.]])
        np.testing.assert_allclose(fitted.centers, [[.6]], atol=2e-7, rtol=0)
        assert fitted.coordinate_fits is None and len(fitted.group_fits) == 1
        assert fitted.diagnostics["group_vector_stationarity_checked"]
        loss, gradient, working = model.node_terms([[1.]])
        np.testing.assert_allclose(gradient, [[4.]], atol=1e-10)
        assert np.isfinite(loss).all() and np.isfinite(working).all() and (working > 0).all()
        _verify_fitted_numerics(model, fitted)
        report["boundary_rescue"] = dict(center=float(fitted.centers.item()), correct_endpoint_derivative=True)

    near_boundary = write_catalog(directory / "stationary_incumbent.tsv",
        [[50, 50]] * 4 + [[20, 15]] * 4, 100, 1, minor=1)
    with RegionalModel(near_boundary.fit_data(), device, cpu_threads=1) as model:
        start = np.array([[np.nextafter(1., 0.)] * 2, [.3984375, .30078125]])
        labels = np.repeat(np.arange(2), 4)
        baseline = model.fit_column_set(start).assigned_values(labels).sum()
        fitted = model.refit(labels, start)
        assert fitted.eligible
        assert fitted.fitting_log_likelihood >= baseline
        _verify_group_stationarity(model, fitted)
        report["stationary_incumbent_roundoff"] = dict(success=True, supplied_objective_preserved=True)

    fallback = write_catalog(directory / "fallback.tsv", [[0, 3]], 10, [4, 6])
    bounds = MultiplicityBounds(fallback.mutation_ids, fallback.region_ids, [[3, 5]], [[4, 6]], "hard_fixture")
    with RegionalModel(fallback.fit_data(), device, cpu_threads=1, multiplicity_support=bounds) as model:
        def sparse_pool(view, *, return_pool=False):
            c = 1. if view.region == 0 else .5
            return np.full(len(view), c), {}, np.array([c]), np.ones(1)
        with patch("CliPP2.kernel.model.pooled_ccf_initialization", side_effect=sparse_pool):
            pilot, diagnostics = model.pilots()
        np.testing.assert_array_equal(pilot, [[.75, .5]])
        assert diagnostics["joint"]["interior_fallback_rows"] == 1
        assert np.isfinite(model.loss(pilot)).all()
        report["sparse_pool"] = dict(interior_fallback=True, finite_joint_start=True)

    parent_catalog = write_catalog(directory / "groups.tsv",
        [[2, 3], [3, 4], [6, 5], [7, 6], [11, 12], [12, 13]], 40, 3)
    report["group_refits"] = {}
    for method in ("marginal", "profile"):
        with RegionalModel(parent_catalog.fit_data(), device, cpu_threads=1, fit_multiplicity=method) as model:
            pilot, _ = model.pilots()
            tree = build_tree(pilot, model.observed, model.mutation_ids)
            bank = CandidateBank(model, tree)
            fitted = model.refit(np.repeat(np.arange(3), 2))
            _verify_fitted_numerics(model, fitted, eligible=fitted.eligible)
            _verify_group_stationarity(model, fitted)
            parent = bank.prepare(None, fitted.labels, fitted=fitted,
                                  columns=model.selection_column_set(fitted.centers))
            moved, initial, reason = bank.prepare_single_move(parent, 1, 1, fitted.centers)
            assert reason is None
            before = model.telemetry["unchanged_group_reuses"]
            answer = model.refit_prepared(moved, initial)
            baseline = model.fit_column_set(initial).assigned_values(moved.labels).sum()
            assert answer.fitting_log_likelihood >= baseline - 2e-9 * (1 + abs(baseline))
            assert model.telemetry["unchanged_group_reuses"] == before + 1
            assert answer.group_fits[2] is fitted.group_fits[2]
            _verify_fitted_numerics(model, answer, eligible=answer.eligible)
            _verify_group_stationarity(model, answer)
            columns = independent_log_columns(model, answer.centers,
                                             aggregation=method)
            np.testing.assert_allclose(answer.fitting_log_likelihood,
                columns[np.arange(model.n), answer.labels].sum(), atol=2e-8, rtol=1e-10)
            with patch.object(model, "_fit_group_vectors", side_effect=AssertionError("selection refitted groups")):
                scored = model.score_frozen_centers(answer.centers, answer.labels)
            np.testing.assert_allclose(scored["score"], answer.score, atol=1e-8)
            report["group_refits"][method] = dict(complete_group_cache=True, changed_groups_refitted=True,
                fitting_objective_verified=True, frozen_marginal_selection=True,
                incumbent_retained_or_improved=True, projected_stationarity_verified=True)

    for name, alt, major in (("one_region", [[4], [8], [12]], 3),
                             ("already_allowed", [[4, 6], [8, 7], [12, 10]], 2)):
        cat = write_catalog(directory / (name + ".tsv"), alt, 40, major)
        with RegionalModel(cat.fit_data(), device, cpu_threads=1) as joint, \
                RegionalModel(cat.fit_data(), device, cpu_threads=1, max_multiplicity_difference=None) as independent:
            centers = np.full((2, joint.r), .3)
            centers[1] = .7
            np.testing.assert_allclose(joint.selection_column_set(centers).materialize(),
                independent.selection_column_set(centers).materialize(), atol=2e-12, rtol=2e-12)
            np.testing.assert_allclose(joint.pilots()[0], independent.pilots()[0], atol=2e-10, rtol=2e-10)
        report[name] = dict(independent_likelihood_and_pilot_recovered=True)
    return report


if __name__ == "__main__":
    import argparse
    import tempfile
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    print(json.dumps(run_checks(args.outdir or tempfile.mkdtemp(prefix="clipp2-joint-model-"), device=args.device), indent=2))
