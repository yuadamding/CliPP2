"""Native emissions, CPU weights, and mocked resident-boundary regressions."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from CliPP2.assignment import confidence_passes
from CliPP2.io.data import MultiplicityBounds
from CliPP2.kernel.backend import NativeBackend
from CliPP2.kernel.observations import prepare_prediction

from check_support import write_catalog


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


def _cpu_weight_checks():
    """Independent objectives plus exact boundary/underflow solver regressions."""
    from scipy.special import logsumexp
    from CliPP2.kernel.scoring import fit_mixture_weights

    cases = {
        "singleton": (np.array([[-1000.], [-100.], [0.]]), [1e300]),
        "duplicates": (np.log([[.9, .9, .1], [.1, .1, .9]]), [8., 1., 1.]),
        "boundary": (np.log([[.9, .1], [.8, .2], [.7, .3]]), [0., 1.]),
        "underflow": (np.array([[0., -1000.], [-1000., 0.]]), [1., 0.]),
        "extreme_finite": (np.array([[-1e6, -1e6-1000.], [-1e6-1200., -1e6],
                                      [-1e6, -1e6-900.]]), [1., 0.]),
        "repair": (-np.random.default_rng(0).exponential(5., (12, 4)), [1., 0., 0., 0.]),
    }
    records = {}
    for name, (columns, start) in cases.items():
        before, initial = columns.copy(), np.array(start, dtype=float)
        events, telemetry = [], {}
        result = fit_mixture_weights(columns, initial, telemetry=telemetry, progress_callback=events.append)
        weights = result["cluster_weights"]
        with np.errstate(divide="ignore"):
            masses = logsumexp(columns + np.log(weights), axis=1)
        np.testing.assert_allclose(result["log_likelihood"], masses.sum(), rtol=1e-14, atol=1e-12)
        scores = np.exp(columns-masses[:, None]).mean(axis=0)
        assert max(0., float(scores.max()-1.)) <= 1.01e-8
        assert float(scores.max()-scores[weights > 0].min()) <= 1.01e-8
        assert result["weight_optimality_gap"] <= len(columns)*1e-8
        assert result["weight_active_score_gap"] <= 1e-8
        np.testing.assert_array_equal(columns, before)
        np.testing.assert_array_equal(initial, start)
        assert events == [dict(operation="weight_fit", occupied_q=len(weights), boundary="solve_complete")]
        records[name] = dict(result={key: value.tolist() if isinstance(value, np.ndarray) else value
                                   for key, value in result.items()}, telemetry=telemetry, events=events)
    np.testing.assert_allclose(records["duplicates"]["result"]["cluster_weights"][2], .5, atol=1e-8, rtol=0)
    np.testing.assert_array_equal(records["boundary"]["result"]["cluster_weights"], [1., 0.])
    np.testing.assert_allclose(records["underflow"]["result"]["cluster_weights"], [.5, .5], atol=1e-8, rtol=0)
    np.testing.assert_allclose(records["extreme_finite"]["result"]["cluster_weights"], [2/3, 1/3], atol=1e-8, rtol=0)
    assert records["underflow"]["telemetry"]["weight_log_domain_fallback_rows"] > 0
    assert records["repair"]["telemetry"]["weight_transfer_steps"] > 0
    for columns, start in (([[-np.inf, -np.inf]], None), ([[0., np.nan]], None),
                           ([[0., np.inf]], None), ([[0., 0.]], [0., 0.])):
        events = []
        try:
            fit_mixture_weights(columns, start, progress_callback=events.append)
        except ValueError:
            assert not events
        else:
            raise AssertionError("Invalid CPU weight problem was accepted")
    return records


def _resident_lease_checks():
    """Mock the native call only: qualify Python ownership/routing, not CUDA math."""
    import ctypes as ct
    from contextlib import contextmanager
    from types import SimpleNamespace
    from unittest.mock import Mock
    from CliPP2.kernel.backend import DeviceMixtureProblem, NativeExecutionError, WeightBatchCancelled
    from CliPP2.kernel.resident_weights import fit_resident_weights
    from CliPP2.kernel.scoring import WeightConvergenceError

    class Owner:
        kind, n = "cuda", 4
        def __init__(self):
            self.context, self.locked, self.mode, self.calls = object(), False, "success", 0
            self.sizes, self.released = {}, []
            self.lib = SimpleNamespace(CliPP2CreateMixtureProblem=Mock(side_effect=self.create),
                CliPP2ReleaseMixtureProblem=Mock(side_effect=self.release),
                CliPP2ResidentWeights=Mock(side_effect=self.solve))
        def _require_open(self):
            if self.context is None:
                raise NativeExecutionError("closed owner")
        def _column_handles(self, columns):
            return np.asarray(columns, dtype=np.int64)
        def _check(self, code):
            if code:
                raise ValueError("native rejected input")
        @contextmanager
        def weight_submission(self):
            assert not self.locked
            self.locked = True
            try:
                yield
            finally:
                self.locked = False
        def create(self, context, q, columns, output):
            handle = len(self.sizes)+1
            self.sizes[handle] = q
            ct.cast(output, ct.POINTER(ct.c_int64))[0] = handle
            return 0
        def release(self, context, handle):
            self.released.append(handle)
            return 0
        def solve(self, context, count, handles, starts, weights, likelihood, optimality, active, statuses, counters):
            assert self.locked
            self.calls += 1
            if self.mode == "native_error":
                return 2
            cursor = 0
            for i in range(count):
                q = self.sizes[handles[i]]
                values = np.asarray([starts[cursor+j] for j in range(q)])
                values /= values.sum()
                for j, value in enumerate(values):
                    weights[cursor+j] = value
                likelihood[i], optimality[i], active[i] = -1., 0., 0.
                statuses[i] = 3 if self.mode == "nonconverged" else 1 if self.mode == "unknown" else 0
                if self.mode == "false_success":
                    active[i] = 1.
                for j in range(6):
                    counters[6*i+j] = 0
                cursor += q
            return 0

    owner, events = Owner(), []
    def progress(index, record):
        assert not owner.locked
        events.append((index, record["boundary"]))
    def fails(error_type, operation):
        calls = len(events)
        try:
            operation()
        except error_type:
            assert not owner.locked and len(events) == calls
        else:
            raise AssertionError("Invalid resident operation was accepted")
    with DeviceMixtureProblem(owner, [11, 12]) as first, DeviceMixtureProblem(owner, [13]) as second:
        for _ in range(2):
            outcomes = fit_resident_weights(owner, [first, second], [[2., 1.], [1.]], progress_callback=progress)
            assert [outcome.operation for outcome in outcomes] == ["complete", "complete"]
            np.testing.assert_array_equal(outcomes[0].result["cluster_weights"], [2/3, 1/3])
            assert not outcomes[0].result["cluster_weights"].flags.writeable
        assert events == [(0, "solve_complete"), (1, "solve_complete")]*2 and not owner.released
        before = owner.calls
        fails(ValueError, lambda: fit_resident_weights(owner, [first, first]))
        fails(ValueError, lambda: fit_resident_weights(owner, [first], [[0., 0.]]))
        fails(WeightBatchCancelled, lambda: fit_resident_weights(owner, [first], cancelled=lambda: True))
        assert owner.calls == before
        cancellation = iter([False, True])
        fails(WeightBatchCancelled, lambda: fit_resident_weights(owner, [first],
            cancelled=lambda: next(cancellation), progress_callback=progress))
        assert owner.calls == before+1
        for mode in ("native_error", "unknown", "false_success"):
            owner.mode = mode
            fails(NativeExecutionError, lambda: fit_resident_weights(owner, [first], progress_callback=progress))
        owner.mode = "nonconverged"
        outcome, = fit_resident_weights(owner, [first], progress_callback=progress)
        assert outcome.operation == "error" and isinstance(outcome.error, WeightConvergenceError)
        assert events[-1] == (0, "solve_failed") and not owner.released
    assert owner.released == [2, 1]
    first.close()
    second.close()
    assert owner.released == [2, 1]
    fails(NativeExecutionError, first.backend)
    assert not owner.locked
    return dict(repeated_owned_leases=True, ordered_detached_callbacks=True,
        validation_before_submission=True, cancellation_boundaries=True,
        execution_errors_and_scientific_nonconvergence_separate=True, gpu_arithmetic_qualified=False)


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
    return dict(cpu_weight_edges={name: True for name in _cpu_weight_checks()},
                resident_python_lifecycle=_resident_lease_checks(),
                single_state_contract=_single_state_contract_checks(directory),
                stable_finite_gap_log_probability=True, underflow_cannot_imply_certainty=True,
                impossible_alternative_and_singleton_certainty=True,
                tied_and_nonmap_sources_rejected=True, absolute_emission_unchanged=True)


def _cuda_weight_fixtures():
    """Controlled columns independent of biological fitting or GPU availability."""
    fixtures = []
    for n, q in ((128, 20), (129, 20), (128, 21), (129, 21), (8, 1)):
        columns = -.01*np.arange(n)[:, None] - 4.*(np.arange(q)[None, :] != (np.arange(n) % q)[:, None])
        fixtures.append(dict(name=f"boundary_{n}_{q}", columns=columns, start=np.eye(1, q)[0]))
    columns = fixtures[0]["columns"][:, :10]
    fixtures.append(dict(name="duplicate_columns", columns=np.repeat(columns, 2, axis=1), start=np.arange(1., 21.)))
    fixtures.append(dict(name="boundary_optimum", columns=-.01*np.arange(129)[:, None]-np.arange(21)[None, :],
                         start=np.eye(21)[-1]))
    columns = np.full((129, 21), -11000.)
    columns[np.arange(129), np.arange(129) % 3] = -10000.
    fixtures.append(dict(name="finite_underflow", columns=columns, start=np.eye(21)[-1]))
    columns = np.full((129, 3), -np.inf)
    columns[np.arange(129), np.arange(129) % 3] = 0.
    fixtures.append(dict(name="structural_support_zero_start", columns=columns, start=np.array([0., 1., 0.])))
    from CliPP2.io.data import content_hash
    for item in fixtures:
        item["fixture_id"] = content_hash(dict(schema="clipp2.cuda-weight-fixture.v1", name=item["name"],
            columns=item["columns"], initial_weights=item["start"]))
    return fixtures


def _cuda_weight_oracle(columns, weights):
    """Independent logsumexp objective and simplex derivatives, without clipping."""
    from scipy.special import logsumexp
    weights = np.asarray(weights, dtype=float)
    assert weights.shape == (columns.shape[1],) and np.isfinite(weights).all() and np.all(weights >= 0)
    np.testing.assert_allclose(weights.sum(), 1., rtol=0, atol=1e-12)
    offsets = columns.max(axis=1)
    assert np.isfinite(offsets).all()
    shifted = columns-offsets[:, None]
    with np.errstate(divide="ignore"):
        mass = logsumexp(shifted+np.log(weights), axis=1)
    assert np.isfinite(mass).all()
    scores = np.exp(logsumexp(shifted-mass[:, None], axis=0)-np.log(len(columns)))
    np.testing.assert_allclose(weights@scores, 1., rtol=0, atol=2e-12)
    return dict(log_likelihood=float(np.sum(offsets+mass)), centered_log_likelihood=float(mass.sum()),
        weight_optimality_gap=len(columns)*max(0., float(scores.max()-1.)),
        weight_active_score_gap=float(scores.max()-scores[weights > 0].min()),
        weighted_score_sum=float(weights@scores))


def _cuda_check_result(fixture, result, reference):
    """Compare objectives and unchanged gates, allowing nonunique optimal weights."""
    n = len(fixture["columns"])
    oracle = _cuda_weight_oracle(fixture["columns"], result["cluster_weights"])
    for checked in (result, oracle):
        assert 0 <= checked["weight_optimality_gap"] <= n*1e-8
        assert 0 <= checked["weight_active_score_gap"] <= 1e-8
    np.testing.assert_allclose(result["log_likelihood"], oracle["log_likelihood"], rtol=0, atol=1e-7)
    np.testing.assert_allclose(oracle["centered_log_likelihood"], reference["centered_log_likelihood"],
                               rtol=0, atol=max(1e-7, 2*n*1e-8))
    return oracle


def _cuda_graph_weights(backend, problems, starts):
    """Private hook retained solely for same-input small/graph qualification."""
    import ctypes as ct
    from CliPP2.kernel.backend import _ptr
    function = backend.lib.CliPP2QualifyResidentGraphWeights
    function.argtypes = ([ct.c_void_p, ct.c_int64, ct.POINTER(ct.c_int64)]
        + [ct.POINTER(ct.c_double)]*5 + [ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int64)])
    function.restype = ct.c_int
    count, total = len(problems), sum(problem.shape[1] for problem in problems)
    handles = np.asarray([problem.handle for problem in problems], dtype=np.int64)
    initial = np.concatenate(starts)
    weights = np.full(total, np.nan)
    likelihood, optimality, active = (np.full(count, np.nan) for _ in range(3))
    statuses, counters = np.full(count, -1, np.int32), np.full((count, 6), -1, np.int64)
    with backend.weight_submission():
        backend._check(function(backend.context, count, _ptr(handles), _ptr(initial), _ptr(weights),
            _ptr(likelihood), _ptr(optimality), _ptr(active), statuses.ctypes.data_as(ct.POINTER(ct.c_int32)), _ptr(counters)))
    records, cursor = [], 0
    for i, problem in enumerate(problems):
        q = problem.shape[1]
        records.append(dict(native_status=int(statuses[i]), cluster_weights=weights[cursor:cursor+q].copy(),
            log_likelihood=float(likelihood[i]), weight_optimality_gap=float(optimality[i]),
            weight_active_score_gap=float(active[i]), counters=counters[i].tolist()))
        cursor += q
    return records


def _cuda_backend(n):
    from types import SimpleNamespace
    from CliPP2.io.data import TumorData
    from CliPP2.kernel.observations import prepare_observations
    shape = n, 1
    data = TumorData("cuda_qualification", tuple(f"m{i}" for i in range(n)), ("r",),
        np.zeros(shape), np.ones(shape), np.ones(shape), np.ones(shape), np.ones(shape), np.full(shape, 2))
    return NativeBackend(SimpleNamespace(n=n, r=1, device="cuda", **prepare_observations(data)), cpu_threads=1)


def _cuda_import(backend, columns):
    import ctypes as ct
    from CliPP2.kernel.backend import DeviceColumn, _array, _ptr
    handles = np.zeros(columns.shape[1], dtype=np.int64)
    values = _array(columns.T)
    function = backend.lib.CliPP2ImportJointColumns
    function.argtypes = [ct.c_void_p, ct.c_int64, ct.POINTER(ct.c_double), ct.POINTER(ct.c_int64)]
    function.restype = ct.c_int
    backend._check(function(backend.context, len(handles), _ptr(values), _ptr(handles)))
    return tuple(DeviceColumn(backend, handle) for handle in handles)


def _cuda_boundary_preflight():
    """Check qualification adapter payloads without loading a CUDA binary."""
    import ctypes as ct
    import sys
    from types import SimpleNamespace
    from unittest.mock import Mock, patch
    with patch.object(sys.modules[__name__], "NativeBackend") as constructor:
        _cuda_backend(3)
        model = constructor.call_args.args[0]
        assert (model.n, model.r, model.device) == (3, 1, "cuda")
        assert constructor.call_args.kwargs == {"cpu_threads": 1}
        for field, expected in (("alt", 0), ("depth", 1), ("scaling", .5),
                                ("support", 1), ("multiplicity_min", 1), ("observed", True)):
            np.testing.assert_array_equal(getattr(model, field), np.full((3, 1), expected))
    class Owner:
        n, context, status = 3, ct.c_void_p(77), 0
        def _check(self, code):
            if code:
                raise ValueError("controlled import rejection")
    owner = Owner()
    expected = np.array([[0., -np.inf], [-1., -2.], [-3., -4.]])
    def imported(context, count, values, output):
        assert context is owner.context and count == 2
        np.testing.assert_array_equal(np.ctypeslib.as_array(values, shape=(6,)).reshape(2, 3).T, expected)
        output[0], output[1] = 101, 102
        return owner.status
    owner.lib = SimpleNamespace(CliPP2ImportJointColumns=Mock(side_effect=imported),
                                CliPP2ReleaseJointColumns=Mock(return_value=0))
    columns = _cuda_import(owner, expected)
    assert [column.handle for column in columns] == [101, 102]
    assert owner.lib.CliPP2ImportJointColumns.restype is ct.c_int
    for column in columns:
        column.close()
    assert owner.lib.CliPP2ReleaseJointColumns.call_count == 2
    owner.status = 2
    try:
        _cuda_import(owner, expected)
    except ValueError:
        pass
    else:
        raise AssertionError("Import error was ignored")
    return dict(validated_observation_payload=True, explicit_ctypes_import=True,
                column_major_transfer=True, native_status_propagated=True, gpu_execution=False)


def _cuda_execute(fixtures, references, report):
    from contextlib import ExitStack
    from CliPP2.kernel.backend import NativeExecutionError, WeightBatchCancelled
    from CliPP2.kernel.resident_weights import fit_resident_weights
    by_name = {item["name"]: item for item in fixtures}
    for n in sorted({len(item["columns"]) for item in fixtures}):
        report["active_request"] = dict(stage="backend_create", mutations=n)
        with _cuda_backend(n) as backend, ExitStack() as leases:
            report["loaded_backends"].append(dict(mutations=n, kind=backend.kind,
                build_id=backend.identity, numerical_policy=backend.policy))
            selected = [item for item in fixtures if len(item["columns"]) == n]
            problems = {}
            for item in selected:
                report["active_request"] = dict(stage="column_import", name=item["name"], fixture_id=item["fixture_id"])
                columns = _cuda_import(backend, item["columns"])
                try:
                    np.testing.assert_array_equal(backend.export_joint_columns(columns), item["columns"])
                    problems[item["name"]] = leases.enter_context(backend.mixture_problem(columns))
                finally:
                    for column in columns:
                        column.close()
            assert backend.scoring_diagnostics()["live_column_handles"] == 0

            def execute(names, label, *, graph=False, repeat=False, replace_geometry=False):
                report["active_request"] = dict(stage="resident_weights", names=names, batch=label, force_graph=graph)
                before = backend.resident_execution_diagnostics()
                request = [problems[name] for name in names]
                starts = [by_name[name]["start"] for name in names]
                callbacks = []
                if graph:
                    results = _cuda_graph_weights(backend, request, starts)
                else:
                    def completed(index, record):
                        with backend.weight_submission():
                            callbacks.append((index, record["boundary"]))
                    outcomes = fit_resident_weights(backend, request, starts, progress_callback=completed)
                    results = [dict(native_status=0, **outcome.result) if outcome.operation == "complete" else
                        dict(native_status=3, error=str(outcome.error)) for outcome in outcomes]
                    assert callbacks == [(i, "solve_complete" if result["native_status"] == 0 else "solve_failed")
                                         for i, result in enumerate(results)]
                after = backend.resident_execution_diagnostics()
                expected_graph = graph or n > 128 or max(problem.shape[1] for problem in request) > 20
                assert (after["graph_launches"] > before["graph_launches"]) == expected_graph
                assert after["workspace_resets"] == before["workspace_resets"]+1 and after["live_owner"] == 0
                if repeat:
                    assert after["arena_reuses"] > before["arena_reuses"]
                    if expected_graph:
                        assert after["graph_reuses"] > before["graph_reuses"]
                if replace_geometry:
                    assert after["arena_allocations"] > before["arena_allocations"]
                    assert after["workspace_releases"] > before["workspace_releases"]
                for name, result in zip(names, results):
                    item = by_name[name]
                    record = dict(name=name, fixture_id=item["fixture_id"], batch=label, force_graph=graph,
                        schedule="graph" if expected_graph else "small", dimensions=list(item["columns"].shape),
                        initial_weights=item["start"], result=result, diagnostics_before=before, diagnostics_after=after,
                        execution_status={0: "complete", 3: "scientific_nonconvergence"}.get(
                            result["native_status"], "execution_error"),
                        qualification_status="pending")
                    report["cases"].append(record)
                    try:
                        assert result["native_status"] == 0, f"{name}: unexpected resident nonconvergence"
                        record["oracle"] = _cuda_check_result(item, result, references[name])
                        record["qualification_status"] = "passed"
                    except Exception:
                        record["qualification_status"] = "failed"
                        raise
            for item in selected:
                execute([item["name"]], "single")
                execute([item["name"]], "repeat", repeat=True)
                if item["name"] == "boundary_128_20":
                    execute([item["name"]], "forced_graph", graph=True)
                    execute([item["name"]], "forced_graph_repeat", graph=True, repeat=True)
            names = [item["name"] for item in selected]
            if len(names) > 1:
                execute(names, "mixed_q", replace_geometry=True)
                execute(names, "mixed_q_repeat", repeat=True)
                sizes = [problems[name].shape[1] for name in names]
                execute(names[::-1], "reordered_mixed_q", repeat=sizes == sizes[::-1],
                        replace_geometry=sizes != sizes[::-1])
                execute(names[::-1], "reordered_mixed_q_repeat", repeat=True)
            first = problems[names[0]]
            report["active_request"] = dict(stage="ownership_and_error_boundaries", mutations=n)
            before = backend.resident_execution_diagnostics()["workspace_resets"]
            for request, starts, cancelled, error_type in (
                    ([first, first], None, None, ValueError), ([first], [np.zeros(first.shape[1])], None, ValueError),
                    ([first], None, lambda: True, WeightBatchCancelled)):
                try:
                    fit_resident_weights(backend, request, starts, cancelled=cancelled)
                except error_type:
                    pass
                else:
                    raise AssertionError("Invalid/cancelled resident request was accepted")
            assert backend.resident_execution_diagnostics()["workspace_resets"] == before
            cancellation = iter([False, True])
            try:
                fit_resident_weights(backend, [first], cancelled=lambda: next(cancellation))
            except WeightBatchCancelled:
                pass
            else:
                raise AssertionError("Post-call cancellation was ignored")
            execute([names[0]], "after_cancellation")
            report["active_request"] = dict(stage="ownership_and_error_boundaries", mutations=n)
            with _cuda_backend(n) as other:
                try:
                    fit_resident_weights(other, [first])
                except ValueError:
                    pass
                else:
                    raise AssertionError("A cross-context lease was accepted")
            impossible = np.zeros((n, 2))
            impossible[0] = -np.inf
            columns = _cuda_import(backend, impossible)
            try:
                try:
                    with backend.mixture_problem(columns):
                        raise AssertionError("A mutation without finite support was accepted")
                except ValueError:
                    pass
            finally:
                for column in columns:
                    column.close()
            execute([names[0]], "after_rejected_input")
            first.close()
            try:
                fit_resident_weights(backend, [first])
            except NativeExecutionError:
                pass
            else:
                raise AssertionError("A closed lease was accepted")
            leases.close()
            final = backend.scoring_diagnostics()
            assert final["live_column_handles"] == final["live_mixture_problems"] == 0
            assert backend.resident_execution_diagnostics()["live_owner"] == 0
            report["ownership"].append(dict(mutations=n, status="passed", columns_released_before_solve=True,
                invalid_and_cancelled_requests=True, no_support_rejected=True, diagnostics=backend.diagnostics()))
    report.pop("active_request", None)


def run_cuda_qualification(directory, *, execute=False):
    """Opt-in real GPU qualification; CPU preflight cannot qualify GPU arithmetic."""
    import json
    import os
    import subprocess
    import sys
    from CliPP2.io.data import json_value
    from CliPP2.kernel.build_identity import backend_identity, native_identity
    from CliPP2.kernel.scoring import fit_mixture_weights

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "cuda_qualification.json"
    if output.exists():
        raise FileExistsError(output)
    def command(args):
        try:
            result = subprocess.run(args, capture_output=True, text=True, timeout=10, check=False)
            return dict(returncode=result.returncode, stdout=result.stdout.strip(), stderr=result.stderr.strip())
        except (OSError, subprocess.TimeoutExpired) as error:
            return dict(status="unavailable", error=str(error))
    repository = Path(__file__).resolve().parents[1]
    report = dict(schema="clipp2.cuda-weight-qualification.v1", mode="cuda" if execute else "cpu_preflight",
        status="running", gpu_arithmetic_qualified=False, source_files_modified_by_qualification=False,
        source_revision=command(["git", "-C", str(repository), "rev-parse", "HEAD"]),
        source_worktree=command(["git", "-C", str(repository), "status", "--porcelain"]),
        native_builds=dict(controller=native_identity(), cpu=backend_identity("cpu"), cuda=backend_identity("cuda")),
        environment=dict(python=sys.version, numpy=np.__version__, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES")),
        numerical_gates=dict(mean_optimality_gap=1e-8, active_score_gap=1e-8,
            weight_sum_atol=1e-12, native_log_likelihood_atol=1e-7,
            centered_objective_atol="max(1e-7, 2*N*1e-8), bounds combined solver optimality gaps"),
        cpu_cases=[], cases=[], ownership=[], loaded_backends=[])
    error = None
    try:
        if not __debug__:
            raise RuntimeError("CUDA qualification requires Python assertions enabled")
        report["boundary_preflight"] = _cuda_boundary_preflight()
        fixtures, references = _cuda_weight_fixtures(), {}
        for item in fixtures:
            result = fit_mixture_weights(item["columns"], item["start"])
            oracle = _cuda_weight_oracle(item["columns"], result["cluster_weights"])
            _cuda_check_result(item, result, oracle)
            references[item["name"]] = oracle
            report["cpu_cases"].append(dict(name=item["name"], fixture_id=item["fixture_id"],
                dimensions=list(item["columns"].shape), initial_weights=item["start"], result=result, oracle=oracle))
        if execute:
            report["environment"].update(nvcc=command(["nvcc", "--version"]),
                gpu=command(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader"]))
            _cuda_execute(fixtures, references, report)
            report.update(status="passed", gpu_arithmetic_qualified=True)
        else:
            report["status"] = "passed_cpu_preflight"
    except Exception as caught:
        error = caught
        report.update(status="failed", error=dict(type=type(caught).__name__, message=str(caught)))
    with output.open("x", encoding="utf-8") as stream:
        json.dump(json_value(report), stream, indent=2, allow_nan=False)
        stream.write("\n")
    if error is not None:
        raise RuntimeError(f"CUDA qualification {report['status']}; evidence: {output}") from error
    return json_value(report)
