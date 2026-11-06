"""Native conditional-confidence regressions, including rounded certainty."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from CliPP2.assignment import confidence_passes
from CliPP2.io.data import MultiplicityBounds
from CliPP2.kernel.backend import NativeBackend
from CliPP2.kernel.observations import prepare_prediction

from check_workflow import write_catalog


def _single_state_contract_checks(directory):
    """Independent binomial oracle across normal CN and the full integer support."""
    from dataclasses import replace
    from scipy.special import logsumexp
    from scipy.stats import binom
    from CliPP2.io.data import TumorData
    from CliPP2.kernel.model import RegionalModel
    from CliPP2.kernel.observations import prepare_observations

    data = TumorData("single_state", ("m0", "m1", "m2"), ("normal0", "normal1", "normal2"),
        np.array([[0, 0, 0], [20, 20, 20], [4, 7, 11]]), np.full((3, 3), 20),
        np.broadcast_to([.3, .5, .8], (3, 3)), np.full((3, 3), 6), np.zeros((3, 3)),
        np.broadcast_to([0, 1, 2], (3, 3)),
        count_observed=np.array([[True, False, True], [True, True, True], [True, True, True]]),
        input_excluded=np.array([[False, False, False], [False, False, False], [False, True, False]]))
    full = prepare_observations(data)
    assert np.all(full["support"] == 6) and not full["observed"][0, 1] and not full["observed"][2, 1]
    bounds = MultiplicityBounds(data.mutation_ids, data.region_ids, np.full((3, 3), 2), np.full((3, 3), 5))
    limited = prepare_observations(data, bounds)
    np.testing.assert_array_equal(limited["scaling"], full["scaling"])
    np.testing.assert_array_equal(limited["total"], full["total"])
    for support in (None, bounds):
        model = RegionalModel(data, device="cpu", cpu_threads=1, multiplicity_support=support)
        try:
            assert not model.view(0).standard and not model.view(1).standard and model.view(2).standard
            if support is None:
                import ctypes as ct
                from CliPP2.kernel.backend import _Observations, _array, _ptr
                for lower, upper, flags in ((0., 1., 3), (np.nextafter(0., 1.), 1., 1),
                                            (0., np.nextafter(1., 0.), 1)):
                    buffers = [_array([value]) for value in (0., 20., 1/6, -np.log(6.), lower, upper)]
                    buffers += [_array([1], np.int64), _array([6], np.int64), _array([flags], np.uint8)]
                    native = _Observations(1, 1, *map(_ptr, buffers))
                    context = ct.c_void_p()
                    status = model.backend.lib.CliPP2BackendCreate(ct.byref(native), -1, ct.byref(context))
                    if context.value:
                        model.backend.lib.CliPP2BackendDestroy(context)
                    assert status != 0 and not context.value, "Legacy mixed observation crossed the native ABI"
            for center in (0., np.nextafter(0., 1.), .37, np.nextafter(1., 0.), 1.):
                phi = np.full((model.n, model.r), center)
                loss = model.backend.node_terms(phi, derivatives=False)
                diagnostic = model.backend.component_diagnostics(phi)
                for i, r in np.ndindex(loss.shape):
                    if not model.observed[i, r]:
                        assert loss[i, r] == 0 and diagnostic["status"][i, r] == 4
                        assert np.isnan(diagnostic["log_emission"][i, r])
                        continue
                    multiplicities = np.arange(model.multiplicity_min[i, r], model.multiplicity_max[i, r]+1)
                    probability = np.minimum(1., (model.scaling[i, r]*multiplicities)*center)
                    masses = binom.logpmf(model.alt[i, r], model.depth[i, r], probability)
                    expected = logsumexp(masses)-np.log(len(multiplicities))
                    np.testing.assert_allclose(-loss[i, r], expected, atol=2e-11, rtol=2e-13)
                    np.testing.assert_allclose(diagnostic["log_emission"][i, r], expected, atol=2e-11, rtol=2e-13)
                    if np.isfinite(expected):
                        assert diagnostic["map_multiplicity"][i, r] == multiplicities[np.argmax(masses)]
            # Match the established single-state tie encoding, including its
            # historical constant mixed flag, without retaining that field.
            import hashlib
            keys = []
            for i in range(model.n):
                if support is None:
                    arrays = (model.observed[i], model.alt[i], model.depth[i],
                        model.scaling[i, :, None]*np.arange(1, 7), model.support[i],
                        np.zeros(model.r, dtype=bool), np.zeros(model.r), np.ones(model.r))
                    prefix = b""
                else:
                    arrays = (model.observed[i], model.alt[i], model.depth[i], model.scaling[i],
                        model.multiplicity_min[i], model.multiplicity_max[i], model.support[i],
                        np.zeros(model.r, dtype=bool), np.zeros(model.r), np.ones(model.r))
                    prefix = b"clipp2_observation_multiplicity_intervals_v1\0"
                keys.append(int.from_bytes(hashlib.sha256(prefix+b"".join(value.tobytes() for value in arrays)).digest(), "big"))
            assert model.observation_keys() == keys
        finally:
            model.backend.close()
    for bad in ({"minor_cn": np.full((3, 3), 6.5)}, {"normal_cn": np.full((3, 3), -.1)},
                {"mapping_genuine": np.zeros((3, 3), dtype=bool)},
                {"normal_cn_supplied": np.zeros((3, 3), dtype=bool)}):
        try:
            replace(data, **bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid programmatic single-state CN was accepted")
    from types import SimpleNamespace
    try:
        prepare_observations(SimpleNamespace(**vars(data)))
    except TypeError:
        pass
    else:
        raise AssertionError("An unvalidated observation object bypassed the input contract")
    catalog = write_catalog(directory / "offset_interval.tsv", [[2]], 3, 4)
    row = _diagnostic(catalog, 1., 3., MultiplicityBounds(catalog.mutation_ids, catalog.region_ids, [[2]], [[4]]))
    np.testing.assert_allclose(np.exp(row["log_emission"]), .265625, atol=1e-15, rtol=1e-14)
    assert row["map_multiplicity"] == 3
    return dict(normal_cn_0_1_2=True, support_above_four=True, endpoint_and_mask_oracle=True,
                restricted_support_keeps_original_denominator=True, single_state_tie_encoding_preserved=True,
                legacy_native_flags_and_boxes_rejected=True, validated_programmatic_input_required=True)


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
    return dict(single_state_contract=_single_state_contract_checks(directory),
                stable_finite_gap_log_probability=True, underflow_cannot_imply_certainty=True,
                impossible_alternative_and_singleton_certainty=True,
                tied_and_nonmap_sources_rejected=True, absolute_emission_unchanged=True)
