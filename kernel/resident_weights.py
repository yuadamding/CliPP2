"""Explicit experimental resident optimizer boundary; never default dispatch.

Requires the source-bound experimental resident ABI. It has no host numerical
fallback and cannot reproduce per-request progress callbacks from inside a
resident GPU solve. Cancellation is observed only before/after its synchronous
native call; it never promises mid-kernel cancellation.
"""
import ctypes as ct
from types import MappingProxyType

import numpy as np

from .backend import DeviceMixtureProblem, NativeExecutionError, _ptr
from .execution import preflight_host_export
from .scoring import WeightConvergenceError
from .weight_batch import WeightBatchCancelled, _OWNER_LOCK, _OWNERS
from .weight_controller import WeightRequest
from ..io.data import readonly_array


def _check_dimensions(n, q):
    """Pinned SLSQP uses LP64 offsets across its complete quadratic arena."""
    if (type(n) is not int or type(q) is not int or not 0 < n <= np.iinfo(np.int32).max
            or q <= 0 or q*(q+1)//2+10*q*q+49*q+33 > np.iinfo(np.int32).max):
        raise ValueError('Resident dimensions exceed the qualified LP64 state range')


def fit_resident_weights(backend, problems, initial_weights=None, *, force_graph=False,
                         progress_callback=None, cancelled=None, telemetry=None):
    """Return ordered COMPLETE/scientific-ERROR outcomes for caller-owned leases.

    Status 3 alone denotes genuine final weight nonconvergence. Backend,
    nonterminal/unknown status and invalid claimed-success outputs raise an
    execution error before any partial outcome is returned. Input validation
    precedes mutation. Once the native call is attempted, every problem's
    cached transfer endpoint is invalidated on both success and failure.

    Existing fixed numerical gates remain 1e-8 (scaled by N for optimality).
    Resident arithmetic is separately qualified, not presumed bit-identical to
    SciPy/OpenBLAS. The q20/q64 performance regression keeps this route opt-in.
    """
    if backend.kind != 'cuda':
        raise ValueError('Resident weight fitting requires an explicit CUDA backend')
    backend._require_open()
    if type(force_graph) is not bool:
        raise TypeError('force_graph must be a boolean')
    if progress_callback is not None:
        raise NotImplementedError('Resident fitting does not expose per-request progress callbacks')
    if cancelled is not None and not callable(cancelled):
        raise TypeError('cancelled must be callable or None')
    problems = tuple(problems)
    starts = [None]*len(problems) if initial_weights is None else list(initial_weights)
    if len(starts) != len(problems):
        raise ValueError('Provide one initial-weight vector per resident problem')
    handles, sizes, rows = [], [], []
    for index, problem in enumerate(problems):
        if not isinstance(problem, DeviceMixtureProblem) or problem.backend() is not backend:
            raise ValueError('Resident weight problems must be live leases of the submission owner')
        if problem.handle in handles:
            raise ValueError('Resident optimizers require independent mixture states')
        n, q = problem.shape
        _check_dimensions(n, q)
        start = np.ones(q) if starts[index] is None else np.array(starts[index], dtype=float, copy=True)
        if (start.shape != (q,) or not np.isfinite(start).all()
                or np.any(start < 0) or not start.max() > 0):
            raise ValueError('Initial cluster weights must be finite, nonnegative and nonzero')
        starts[index] = start
        handles.append(problem.handle)
        sizes.append(q)
        rows.append(n)
    if not problems:
        return ()
    name = 'CliPP2ExperimentalResident'+('Graph' if force_graph else '')+'Weights'
    try:
        function = getattr(backend.lib, name)
    except AttributeError as error:
        raise NativeExecutionError('The loaded backend has no qualified experimental resident ABI') from error
    function.argtypes = ([ct.c_void_p, ct.c_int64, ct.POINTER(ct.c_int64)]
                         + [ct.POINTER(ct.c_double)]*5
                         + [ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int64)])
    function.restype = ct.c_int
    count, total_q = len(problems), sum(sizes)
    preflight_host_export(64*total_q+256*count)
    handles = np.asarray(handles, dtype=np.int64)
    starts = np.concatenate(starts)
    weights = np.full(total_q, np.nan)
    likelihood, optimality, active = (np.full(count, np.nan) for _ in range(3))
    statuses, counters = np.full(count, -1, dtype=np.int32), np.full((count, 6), -1, dtype=np.int64)

    def check_cancelled():
        if cancelled is not None and cancelled():
            raise WeightBatchCancelled('Resident fitting cancelled at a synchronous call boundary')

    with _OWNER_LOCK:
        if backend in _OWNERS:
            raise RuntimeError('A weight-batch submission owner is already active')
        _OWNERS.add(backend)
    attempted = False
    try:
        check_cancelled()
        for problem in problems:
            problem._transfer_endpoint = None
        attempted = True
        code = function(backend.context, count, _ptr(handles), _ptr(starts), _ptr(weights),
                        _ptr(likelihood), _ptr(optimality), _ptr(active),
                        statuses.ctypes.data_as(ct.POINTER(ct.c_int32)), _ptr(counters))
        try:
            backend._check(code)
        except ValueError as error:
            raise NativeExecutionError('Resident backend rejected its validated execution request') from error
        check_cancelled()
        if np.any((statuses != 0) & (statuses != 3)):
            raise NativeExecutionError(f'Resident optimizer returned nonterminal/unknown status: {statuses.tolist()}')
        if (not np.isfinite(weights).all() or np.any(weights < 0)
                or not np.isfinite([likelihood, optimality, active]).all()
                or np.any(optimality < 0) or np.any(active < 0) or np.any(counters < 0)):
            raise NativeExecutionError('Resident optimizer returned invalid numerical outputs')
        cursor = 0
        for index, q in enumerate(sizes):
            if not np.isclose(weights[cursor:cursor+q].sum(), 1., atol=1e-12, rtol=0):
                raise NativeExecutionError('Resident optimizer returned non-normalized weights')
            if statuses[index] == 0 and (optimality[index] > rows[index]*1e-8 or active[index] > 1e-8):
                raise NativeExecutionError('Resident success does not satisfy the unchanged numerical gates')
            cursor += q
        result, cursor = [], 0
        for index, q in enumerate(sizes):
            if statuses[index] == 3:
                result.append(WeightRequest('error', error=WeightConvergenceError(
                    'Cluster-weight likelihood optimization did not converge')))
            else:
                result.append(WeightRequest('complete', result=MappingProxyType({
                    'cluster_weights': readonly_array(weights[cursor:cursor+q]),
                    'log_likelihood': float(likelihood[index]),
                    'weight_optimality_gap': float(optimality[index]),
                    'weight_active_score_gap': float(active[index])})))
            cursor += q
        if telemetry is not None:
            changes = dict(resident_weight_batches=1, resident_weight_cases=count,
                           resident_weight_convergence_failures=int(np.count_nonzero(statuses == 3)))
            changes.update(zip(('resident_weight_evaluations', 'resident_weight_cache_hits',
                'resident_weight_preparations', 'resident_weight_trials', 'resident_weight_slsqp_iterations',
                'resident_weight_repair_steps'), map(int, counters.sum(axis=0))))
            for key, value in changes.items():
                telemetry[key] = telemetry.get(key, 0)+value
        return tuple(result)
    finally:
        if attempted:
            for problem in problems:
                problem._transfer_endpoint = None
        with _OWNER_LOCK:
            _OWNERS.discard(backend)
