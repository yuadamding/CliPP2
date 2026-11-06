"""Native conditional-confidence regressions, including rounded certainty."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from CliPP2.assignment import confidence_passes
from CliPP2.io.data import MultiplicityBounds
from CliPP2.kernel.backend import NativeBackend
from CliPP2.kernel.observations import prepare_prediction

from check_workflow import write_catalog


def _diagnostic(catalog, center, call, bounds=None):
    observations = prepare_prediction(catalog, catalog.mutation_ids, bounds)
    with NativeBackend(observations, cpu_threads=1) as backend:
        result = backend.component_diagnostics([[center]], [[call]])
    assert result["status"][0, 0] == 0
    return {key: value[0, 0] for key, value in result.items()}


def _passes(row, threshold):
    finite_rival = bool(row["has_second"] and np.isfinite(row["top_two_gap"]))
    return confidence_passes(row["accepted_log_probability"], threshold, finite_rival)


def run_native_diagnostic_checks(directory):
    """Exercise the shared C++ diagnostic path through the public CPU wrapper."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for depth in (100, 2000):
        catalog = write_catalog(directory / f"finite_{depth}.tsv", [[depth]], depth, 2)
        best = _diagnostic(catalog, 1., 2.)
        expected = -np.log1p(np.exp(-depth*np.log(2.)))
        np.testing.assert_allclose(best["map_log_probability"], expected, rtol=2e-14, atol=0)
        assert best["accepted_log_probability"] == best["map_log_probability"]
        assert np.exp(best["map_log_probability"]) == 1. and best["map_multiplicity"] == 2
        assert np.isfinite(best["top_two_gap"]) and not _passes(best, 1.)
        assert _passes(best, np.nextafter(1., 0.))
        assert _passes(best, .95)
        if depth == 100:
            assert best["map_log_probability"] < 0
        else:
            # Even log1p cannot retain a ratio below the floating-point range.
            # The finite gap still proves that certainty is not exact.
            assert best["map_log_probability"] == 0.
        other = _diagnostic(catalog, 1., 1.)
        np.testing.assert_allclose(other["accepted_log_probability"],
                                   -depth*np.log(2.)+expected, rtol=2e-14)
        assert not _passes(other, 1.)

    catalog = write_catalog(directory / "impossible_alternative.tsv", [[0]], 100, 2)
    sole_finite = _diagnostic(catalog, 1., 1.)
    assert sole_finite["has_second"] and np.isposinf(sole_finite["top_two_gap"])
    assert sole_finite["map_log_probability"] == 0. and _passes(sole_finite, 1.)
    impossible = _diagnostic(catalog, 1., 2.)
    assert np.isneginf(impossible["accepted_log_probability"]) and not _passes(impossible, 1.)
    tied = _diagnostic(catalog, 0., 2.)
    np.testing.assert_allclose(tied["map_log_probability"], -np.log(2.), rtol=1e-15)
    assert tied["top_two_gap"] == 0 and not _passes(tied, 1.)

    catalog = write_catalog(directory / "singleton.tsv", [[100]], 100, 2)
    bounds = MultiplicityBounds(catalog.mutation_ids, catalog.region_ids, [[2]], [[2]])
    singleton = _diagnostic(catalog, 1., 2., bounds)
    assert not singleton["has_second"] and singleton["map_multiplicity"] == 2
    assert np.isposinf(singleton["top_two_gap"]) and _passes(singleton, 1.)
    outside = _diagnostic(catalog, 1., 1., bounds)
    assert np.isneginf(outside["accepted_log_probability"])

    catalog = write_catalog(directory / "normalization.tsv", [[2]], 3, 4)
    full = _diagnostic(catalog, 1., 2.)
    np.testing.assert_allclose(np.exp(full["log_emission"]), 15/64, rtol=1e-14)
    np.testing.assert_allclose(full["map_log_probability"], np.log(.45), rtol=1e-14)
    np.testing.assert_allclose(full["accepted_log_probability"], np.log(.4), rtol=1e-14)
    return dict(stable_finite_gap_log_probability=True, underflow_cannot_imply_certainty=True,
                impossible_alternative_and_singleton_certainty=True,
                tied_and_nonmap_sources_rejected=True, absolute_emission_unchanged=True)
