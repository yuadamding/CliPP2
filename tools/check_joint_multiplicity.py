#!/usr/bin/env python3
"""Exhaustive and native checks of the fixed regional range-one model."""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import tempfile

import numpy as np
from scipy.special import logsumexp

from CliPP2.kernel.joint import joint_components, joint_row


def _enumerate(masses, lower, observed, difference):
    active = np.flatnonzero(observed)
    words = [word for word in itertools.product(*(
        range(int(lower[r]), int(lower[r])+len(masses[r])) for r in active))
        if difference is None or not word or max(word)-min(word) <= difference]
    scores = np.array([sum(masses[r][m-lower[r]] for r, m in zip(active, word)) for word in words])
    return active, words, scores


def reference_checks():
    rng = np.random.default_rng(173)
    errors = []
    for trial in range(300):
        regions = int(rng.integers(1, 6))
        lower = rng.integers(1, 5, size=regions)
        lengths = rng.integers(1, 4, size=regions)
        observed = rng.random(regions) > .2
        # Integral log masses deliberately create exact joint MAP ties.
        masses = [rng.integers(-4, 1, size=int(n)).astype(float) if trial % 3 == 0
                  else rng.normal(size=int(n)) for n in lengths]
        for difference in (None, 1):
            active, words, scores = _enumerate(masses, lower, observed, difference)
            if not words:
                try:
                    joint_components(masses, lower, observed=observed,
                                     max_multiplicity_difference=difference)
                except ValueError:
                    continue
                raise AssertionError("Empty structural support was accepted")
            result = joint_components(masses, lower, observed=observed,
                                      max_multiplicity_difference=difference)
            normalizer = float(logsumexp(scores))
            for actual, expected in ((result["log_normalizer"], math.log(len(words))),
                    (result["log_marginal"], normalizer-math.log(len(words))),
                    (result["profile"], float(scores.max()))):
                errors.append(abs(actual-expected))
                np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)
            expected_map = min(word for word, score in zip(words, scores) if score == scores.max())
            np.testing.assert_array_equal(result["map"][active], expected_map)
            for position, r in enumerate(active):
                probability = np.zeros(lengths[r])
                for word, mass in zip(words, np.exp(scores-normalizer)):
                    probability[word[position]-lower[r]] += mass
                np.testing.assert_allclose(result["probabilities"][r], probability, atol=2e-13, rtol=2e-13)
                if trial % 3 == 0:
                    expected = tuple(sorted({word[position] for word, score in zip(words, scores)
                                             if score == scores.max()}))
                    assert result["map_sets"][r] == expected
    # {1,2}^R already obeys the rule: no preference for equal vectors.
    values = [[-1., -2.], [-3., -.4], [-.2, -4.]]
    constrained = joint_components(values, [1, 1, 1])
    independent = joint_components(values, [1, 1, 1], max_multiplicity_difference=None)
    assert np.isclose(constrained["log_normalizer"], math.log(8))
    np.testing.assert_allclose(constrained["log_marginal"], independent["log_marginal"], atol=2e-14)
    equal = joint_components([[0.]*3]*3, [1, 1, 1])
    assert np.isclose(np.exp(equal["log_normalizer"]), 15)
    assert equal["map_sets"] == ((1, 2, 3),)*3
    empty = joint_components([[0.]*3]*3, [1, 1, 1], observed=[False]*3)
    assert empty["log_marginal"] == empty["profile"] == empty["log_normalizer"] == 0
    # At phi0, the observed coordinate stays in the support even with zero ALT.
    absent = joint_row([0, 10], [20, 20], [.25, .25], [1, 1], [4, 4], [True, True], [0, 1])
    assert absent["map"][0] >= 1 and np.ptp(absent["map"]) <= 1
    assert len(absent["probabilities"][0]) == 4
    return dict(random_cases=300, settings=[None, 1], maximum_error=max(errors),
                equal_vectors_counted_once=True, raw_normalization=True,
                fixed_zero_ccf_scope=True, empty_prediction_product=True)


def native_checks(directory, device="cpu"):
    from CliPP2.io.data import MultiplicityBounds
    from CliPP2.kernel.backend import NativeBackend
    from CliPP2.kernel.observations import prepare_prediction
    from check_support import write_catalog

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    catalog = write_catalog(directory / "native.tsv",
        [[5, 11, 16], [0, 3, 15], [7, 0, 0], [0, 0, 0]], 20, 4,
        observed=[[True]*3, [True]*3, [True, False, False], [False]*3])
    bounds = MultiplicityBounds(catalog.mutation_ids, catalog.region_ids,
        [[1, 2, 1], [1, 1, 1], [2, 1, 1], [4, 1, 1]], [[4]*3]*4)
    phi = np.array([[.43, .71, .92], [0., .56, .87], [.63, .4, .5], [.2, .3, .4]])
    results = {}
    for difference in (None, 1):
        data = prepare_prediction(catalog, catalog.mutation_ids, bounds, device=device,
                                  max_multiplicity_difference=difference)
        for aggregation in ("marginal", "profile"):
            with NativeBackend(data, cpu_threads=1, aggregation=aggregation) as backend:
                terms = backend.node_terms(phi)
                diagnostics = backend.joint_diagnostics(phi)
                loss, gradient, curvature = terms
                assert loss.shape == (4,) and gradient.shape == curvature.shape == (4, 3)
                for i in range(4):
                    def oracle(center):
                        return joint_row(data.alt[i], data.depth[i], data.scaling[i],
                            data.multiplicity_min[i], data.multiplicity_max[i], data.observed[i], center,
                            max_multiplicity_difference=difference)
                    expected = oracle(phi[i])
                    for name in ("log_marginal", "profile", "log_normalizer", "coupling_profile_loss"):
                        np.testing.assert_allclose(diagnostics[name][i], expected[name], rtol=1e-11, atol=2e-11)
                    np.testing.assert_allclose(diagnostics["map_multiplicity"][i], expected["map"], equal_nan=True)
                    for r in np.flatnonzero(data.observed[i]):
                        np.testing.assert_allclose(diagnostics["probabilities"][i][r], expected["probabilities"][r],
                                                   rtol=1e-11, atol=2e-12)
                        assert int(expected["map"][r]) in diagnostics["map_sets"][i][r]
                    field = "log_marginal" if aggregation == "marginal" else "profile"
                    np.testing.assert_allclose(loss[i], -expected[field], rtol=1e-11, atol=2e-11)
                    for r in np.flatnonzero(data.observed[i] & (phi[i] > 0) & (phi[i] < 1)):
                        plus, minus = phi[i].copy(), phi[i].copy()
                        plus[r] += 1e-6
                        minus[r] -= 1e-6
                        derivative = -(oracle(plus)[field]-oracle(minus)[field])/2e-6
                        np.testing.assert_allclose(gradient[i, r], derivative, rtol=2e-6, atol=2e-6)
                assert np.all(np.isfinite(curvature)) and np.all(curvature >= 0)
                results[f"{difference}_{aggregation}"] = True
    # Zero component probability has a nonzero endpoint derivative.
    endpoint = write_catalog(directory / "endpoint.tsv", [[1]], 2, 2)
    data = prepare_prediction(endpoint, endpoint.mutation_ids, device=device)
    with NativeBackend(data, cpu_threads=1) as backend:
        loss, gradient, curvature = backend.node_terms([[1.]])
        np.testing.assert_allclose(loss, [-math.log(.25)], atol=1e-13)
        np.testing.assert_allclose(gradient, [[4.]], atol=1e-12)
        assert np.isfinite(curvature).all() and (curvature > 0).all()
    results["endpoint_derivative"] = True
    # Symmetric exact joint maxima must preserve both projected donor-cap endpoints.
    symmetric = write_catalog(directory / "symmetric.tsv", [[25, 75], [0, 75], [0, 0]], 100, 4)
    centers = np.array([[1., 1.], [0., 1.], [0., 0.]])
    for difference in (None, 1):
        data = prepare_prediction(symmetric, symmetric.mutation_ids, device=device,
                                  max_multiplicity_difference=difference)
        with NativeBackend(data, cpu_threads=1) as backend:
            diagnostics = backend.joint_diagnostics(centers)
        for i in range(3):
            expected = joint_row(data.alt[i], data.depth[i], data.scaling[i], data.multiplicity_min[i],
                data.multiplicity_max[i], data.observed[i], centers[i], max_multiplicity_difference=difference)
            assert diagnostics["map_sets"][i] == expected["map_sets"]
            np.testing.assert_array_equal(diagnostics["map_multiplicity"][i], expected["map"])
        if difference == 1:
            assert diagnostics["map_sets"][0] == ((1, 2), (2, 3))
    results["joint_diagnostics_and_exact_projected_ties"] = True
    rng = np.random.default_rng(385)
    alt = rng.integers(0, 101, (100, 5))
    centers = rng.choice([0., .2, .5, .8, 1.], (100, 5))
    alt[centers == 0] = 0
    observed = rng.random((100, 5)) > .2
    catalog = write_catalog(directory / "random_diagnostics.tsv", alt, 100, 4, observed=observed)
    bounds = MultiplicityBounds(catalog.mutation_ids, catalog.region_ids,
        rng.integers(1, 5, (100, 5)), np.full((100, 5), 4))
    data = prepare_prediction(catalog, catalog.mutation_ids, bounds, device=device)
    with NativeBackend(data, cpu_threads=1) as backend:
        diagnostics = backend.joint_diagnostics(centers)
    for i in range(100):
        expected = joint_row(data.alt[i], data.depth[i], data.scaling[i], data.multiplicity_min[i],
            data.multiplicity_max[i], data.observed[i], centers[i])
        for name in ("log_marginal", "profile", "log_normalizer"):
            np.testing.assert_allclose(diagnostics[name][i], expected[name], rtol=2e-12, atol=2e-11)
        np.testing.assert_array_equal(diagnostics["map_multiplicity"][i], expected["map"])
        if not np.isfinite(diagnostics["profile"][i]):
            assert diagnostics["status"][i] == 1
            assert all(np.isnan(values).all() for values in diagnostics["probabilities"][i])
            continue
        assert diagnostics["map_sets"][i] == expected["map_sets"]
        for r in np.flatnonzero(data.observed[i]):
            np.testing.assert_allclose(diagnostics["probabilities"][i][r], expected["probabilities"][r],
                                       rtol=2e-11, atol=2e-12)
    results["randomized_five_region_diagnostics"] = 100
    return results


def support_checks(directory):
    from CliPP2.io.data import MultiplicityBounds
    from CliPP2.io.multiplicity import reconcile_workflow_bounds
    from CliPP2.kernel.observations import prepare_prediction
    from check_support import write_catalog, expect_value_error

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    catalog = write_catalog(directory / "support.tsv", [[90, 10, 20]], 100, [[5, 1, 4]])
    requested = MultiplicityBounds(catalog.mutation_ids, catalog.region_ids, [[5, 1, 3]], [[5, 1, 4]])
    first, audit = reconcile_workflow_bounds(catalog, requested)
    np.testing.assert_array_equal(first.lower, [[2, 1, 2]])
    np.testing.assert_array_equal(first.upper, [[5, 1, 4]])
    assert audit[0]["raw_read_floor"] == 5 and audit[0]["effective_lower"] == 2
    second, audit = reconcile_workflow_bounds(catalog, requested, [[1, 1, 1]], stage="stage_two")
    np.testing.assert_array_equal(second.lower, [[2, 1, 2]])
    np.testing.assert_array_equal(second.upper, [[2, 1, 2]])
    assert audit[0]["raw_upper_cap"] == 1 and audit[0]["raw_read_floor"] == 5
    expect_value_error(lambda: prepare_prediction(catalog, catalog.mutation_ids, requested))
    prepare_prediction(catalog, catalog.mutation_ids, requested, max_multiplicity_difference=None)
    independent, _ = reconcile_workflow_bounds(catalog, requested, [[1, 1, 1]],
                                                max_multiplicity_difference=None)
    np.testing.assert_array_equal(independent.lower, requested.lower)
    np.testing.assert_array_equal(independent.upper, requested.lower)
    rng = np.random.default_rng(910)
    from types import SimpleNamespace
    for case in range(300):
        major = rng.integers(1, 6, size=(1, 4))
        floor = np.array([[rng.integers(1, m+1) for m in major[0]]])
        caps = np.array([[rng.integers(1, m+1) for m in major[0]]])
        mask = rng.random((1, 4)) > .25
        source = SimpleNamespace(mutation_ids=("m",), region_ids=("a", "b", "c", "d"),
            major_cn=major, count_observed=mask, eligible=np.ones_like(mask),
            total_counts=np.ones_like(major), segment_ids=np.full((1, 4), "s"))
        raw = MultiplicityBounds(source.mutation_ids, source.region_ids, floor, major)
        resolved, _ = reconcile_workflow_bounds(source, raw, caps)
        assert np.all(resolved.lower >= 1) and np.all(resolved.lower <= resolved.upper)
        assert np.all(resolved.upper <= major)
        active = np.flatnonzero(mask[0])
        words = [w for w in itertools.product(*(range(int(resolved.lower[0, r]),
            int(resolved.upper[0, r])+1) for r in active)) if not w or max(w)-min(w) <= 1]
        assert words
        # The retained floor is the largest possible under the original major CNs.
        if len(active):
            feasible = [w for w in itertools.product(*(range(1, int(major[0, r])+1)
                        for r in active)) if max(w)-min(w) <= 1]
            for position, r in enumerate(active):
                assert resolved.lower[0, r] == min(floor[0, r], max(w[position] for w in feasible))
    return dict(random_cases=300, original_major_preserved=True, raw_requests_retained=True,
                stage_two_reconciles_original_floor=True, explicit_support_remains_hard=True)


def run_checks(directory, *, device="cpu", reference_only=False):
    result = dict(reference=reference_checks(), support=support_checks(Path(directory) / "support"))
    if not reference_only:
        result["native"] = native_checks(directory, device)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--reference-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_checks(args.outdir or tempfile.mkdtemp(prefix="clipp2-joint-checks-"),
        device=args.device, reference_only=args.reference_only), indent=2))
