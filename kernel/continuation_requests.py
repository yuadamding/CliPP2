"""Native continuation owner for batched CUDA capacity search.

One owner services ordered node-term requests for independent capacity states.
Only the supplied evaluator performs likelihood arithmetic; native workers do
not fall back to CPU emissions. Worker CPU work is serialized. This module
does not import CUDA.
"""
import ctypes as ct

import numpy as np

from .backend import (_ContinuationPolicy, _ContinuationReport, _array, _continuation_diagnostics,
                      NativeExecutionError, native_library)


class ContinuationCancelled(RuntimeError):
    """The caller cancelled between reverse-communication request batches."""


class _Request(ct.Structure):
    _fields_ = [('sequence', ct.c_uint64), ('kind', ct.c_int),
                ('boundary', ct.c_int), ('level', ct.c_int), ('iterations', ct.c_int64),
                ('coordinates', ct.POINTER(ct.c_double))]


_Norm = ct.CFUNCTYPE(ct.c_double, ct.POINTER(ct.c_double), ct.c_longlong)
_Scores = ct.CFUNCTYPE(ct.c_int, ct.POINTER(ct.c_longdouble), ct.c_longlong,
                      ct.c_int, ct.POINTER(ct.c_longdouble))


def _ptr(array):
    return ct.c_void_p(array.ctypes.data)


def _api():
    library, identity = native_library()
    declarations = {
        'Create': ([ct.c_int]*2 + [ct.c_void_p]*9 + [ct.c_int, ct.c_void_p,
                   ct.POINTER(_ContinuationPolicy), _Norm, _Scores,
                   ct.c_int, ct.c_int, ct.POINTER(ct.c_void_p)], ct.c_int),
        'Poll': ([ct.c_void_p, ct.c_int, ct.c_int, ct.POINTER(_Request)], ct.c_int),
        'Reply': ([ct.c_void_p, ct.c_int, ct.c_uint64] + [ct.c_void_p]*3, ct.c_int),
        'Result': ([ct.c_void_p, ct.c_int] + [ct.c_void_p]*3 +
                   [ct.POINTER(_ContinuationReport)], ct.c_int),
        'Error': ([ct.c_void_p, ct.c_int], ct.c_char_p),
        'Destroy': ([ct.c_void_p], None),
    }
    for suffix, (arguments, result) in declarations.items():
        method = getattr(library, 'CliPP2ContinuationOwner'+suffix)
        method.argtypes, method.restype = arguments, result
    library.CliPP2ContinuationLastError.argtypes = []
    library.CliPP2ContinuationLastError.restype = ct.c_char_p
    return library, identity


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 0 < value < 2**63:
        raise ValueError(f'{name} must be a positive int64 integer')
    return int(value)


def continue_requests(model, tree, initial, capacities, lower, upper, policy, missing=0, *,
                      max_active=8,
                      keep_raw_phi=True, evaluate_requests=None, progress_callback=None,
                      cancelled=None, telemetry=None):
    """Run the native controller with caller-owned numerical evaluation.

    ``evaluate_requests([(capacity_index, owned_coordinates), ...])`` returns
    ordered ``(values, gradient, curvature)`` triples. Its default calls this
    model's node_terms sequentially, so a CUDA model still evaluates on CUDA.
    A future CUDA batch owner may implement the same protocol. No dispatcher
    selects this experimental path automatically.

    Only one native worker does CPU work at a time; other workers suspend.
    Boundary handshakes are installed only for cancellation; observers receive
    ordered completed-capacity snapshots without an internal event queue.
    numerical requests and their original per-capacity order are unchanged.
    """
    for value, name in ((evaluate_requests, 'evaluate_requests'),
                        (progress_callback, 'progress_callback'), (cancelled, 'cancelled')):
        if value is not None and not callable(value):
            raise TypeError(f'{name} must be callable or None')
    if type(keep_raw_phi) is not bool:
        raise TypeError('keep_raw_phi must be a boolean')
    max_active = _positive_int(max_active, 'max_active')
    capacities = list(capacities)
    if (not capacities or len(capacities) != len(set(capacities)) or any(
            isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer))
            or not 1 <= k <= model.n for k in capacities)):
        raise ValueError('Distinct nonempty capacities in 1..N are required')
    if tree.n != model.n or max(model.n, model.r, len(capacities), policy.levels,
                                 policy.iterations_per_level) > np.iinfo(np.int32).max:
        raise ValueError('Continuation dimensions must agree and fit int32')
    initial, lower, upper = map(_array, (initial, lower, upper))
    shape = (model.n, model.r)
    if (any(value.shape != shape or not np.isfinite(value).all() for value in (initial, lower, upper))
            or np.any(lower < 0) or np.any(upper > 1) or np.any(lower > upper)
            or np.any(initial < lower) or np.any(initial > upper)):
        raise ValueError('Continuation needs a finite feasible initial box')
    library, identity = _api()
    active = min(max_active, len(capacities))

    def check(code):
        if code:
            raise NativeExecutionError(library.CliPP2ContinuationLastError().decode())

    boundary_requests = cancelled is not None
    if cancelled is not None and cancelled():
        raise ContinuationCancelled('Continuation cancelled before numerical evaluation')
    terms = tuple(_array(value) for value in model.node_terms(initial))
    if len(terms) != 3 or any(value.shape != shape or not np.isfinite(value).all() for value in terms):
        raise ArithmeticError('Initial tree likelihood terms are invalid')
    edges = _array(tree.edges, np.int32)
    canonical = _array(tree.canonical_node_order, np.int32)
    gaps = _array([0.]+[4*np.sin(np.pi/(2*size))**2 for size in range(1, model.n+1)])
    native_policy = _ContinuationPolicy(policy.levels, policy.iterations_per_level,
        policy.stationarity_tolerance, policy.constraint_tolerance, policy.maximum_penalty)
    reduction_errors = []

    @_Norm
    def norm(values, count):
        try:
            return float(np.linalg.norm(np.ctypeslib.as_array(values, shape=(count,)))) if count else 0.
        except BaseException as error:
            reduction_errors.append(error)
            return float('nan')

    @_Scores
    def scores(values, count, regions, output):
        try:
            def extended(pointer, size):
                raw = ct.cast(pointer, ct.POINTER(ct.c_uint8))
                return np.ctypeslib.as_array(raw, shape=(size*np.dtype(np.longdouble).itemsize,)).view(np.longdouble)
            delta = extended(values, count*regions).reshape(count, regions)
            np.copyto(extended(output, count), np.einsum('ij,ij->i', delta, delta))
            return 0
        except BaseException as error:
            reduction_errors.append(error)
            return 1

    stats = {} if telemetry is None else telemetry
    stats.update(native_continuation_build_id=identity, native_continuation_cpu_workers=1,
                 native_continuation_active_limit=active,
                 native_continuation_owner_submissions=0, native_continuation_terms_requests=1,
                 native_continuation_batch_max=0, native_continuation_initial_evaluations=1,
                 native_continuation_boundary_requests=boundary_requests,
                 native_continuation_boundary_handshakes=0)
    outputs = []
    for first in range(0, len(capacities), active):
        chosen = _array(capacities[first:first+active], np.int32)
        owner = ct.c_void_p()
        next_progress = 0
        completed = set()
        results = [None]*len(chosen)
        try:
            check(library.CliPP2ContinuationOwnerCreate(model.n, model.r, _ptr(edges), _ptr(canonical),
                _ptr(initial), *map(_ptr, terms), _ptr(lower), _ptr(upper), _ptr(gaps), len(chosen),
                _ptr(chosen), ct.byref(native_policy), norm, scores,
                int(keep_raw_phi), int(boundary_requests), ct.byref(owner)))
            while len(completed) < len(chosen):
                if reduction_errors:
                    raise reduction_errors[0]
                if cancelled is not None and cancelled():
                    raise ContinuationCancelled('Continuation cancelled between request batches')
                numerical, requests = [], []
                changed = False
                for task in range(len(chosen)):
                    if task in completed:
                        continue
                    request = _Request()
                    check(library.CliPP2ContinuationOwnerPoll(owner, task, 0, ct.byref(request)))
                    if request.kind == 0:
                        continue
                    changed = True
                    if request.kind == 4:
                        if reduction_errors:
                            raise reduction_errors[0]
                        raise NativeExecutionError(library.CliPP2ContinuationOwnerError(owner, task).decode())
                    if request.kind == 1:
                        x = np.ctypeslib.as_array(request.coordinates, shape=(model.n*model.r,)).reshape(shape).copy()
                        numerical.append((first+task, x))
                        requests.append((task, request.sequence))
                    elif request.kind == 2:
                        stats['native_continuation_boundary_handshakes'] += 1
                        check(library.CliPP2ContinuationOwnerReply(owner, task, request.sequence, None, None, None))
                    elif request.kind == 3:
                        results[task] = _result(library, owner, task, int(chosen[task]), tree, model.r,
                                                missing, keep_raw_phi, check)
                        completed.add(task)
                        while next_progress in completed:
                            diagnostics = results[next_progress]['diagnostics']
                            if progress_callback is not None:
                                progress_callback(dict(operation='continuation', boundary='capacity_complete',
                                    requested_k=int(chosen[next_progress]), capacity_index=first+next_progress,
                                    capacity_count=len(capacities), completed_capacities=first+next_progress+1,
                                    level=diagnostics['continuation_levels'], iterations=diagnostics['iterations']))
                            next_progress += 1
                    else:
                        raise NativeExecutionError('Unknown native continuation request kind')
                if numerical:
                    stats['native_continuation_owner_submissions'] += 1
                    stats['native_continuation_terms_requests'] += len(numerical)
                    stats['native_continuation_batch_max'] = max(stats['native_continuation_batch_max'], len(numerical))
                    if evaluate_requests is None:
                        from .model import RegionalModel
                        if (type(model) is RegionalModel and model.backend.kind == "cuda"
                                and "node_terms" not in vars(model)):
                            replies = model.backend.node_terms_batch(np.stack([x for _, x in numerical]))
                        else:
                            replies = [model.node_terms(x) for _, x in numerical]
                    else:
                        replies = list(evaluate_requests(numerical))
                    if len(replies) != len(requests):
                        raise NativeExecutionError('Continuation evaluator returned incomplete ordered replies')
                    for (task, sequence), reply in zip(requests, replies):
                        values = tuple(_array(value) for value in reply)
                        if len(values) != 3 or any(value.shape != shape for value in values):
                            raise NativeExecutionError('Continuation evaluator returned invalid term shapes')
                        check(library.CliPP2ContinuationOwnerReply(owner, task, sequence, *map(_ptr, values)))
                elif not changed:
                    # No full-batch wait: block only until one busy state yields.
                    waiting = next(task for task in range(len(chosen)) if task not in completed)
                    check(library.CliPP2ContinuationOwnerPoll(owner, waiting, 1, ct.byref(_Request())))
            outputs.extend(results)
        finally:
            # Includes callback, evaluator, cancellation and setup failures.
            # Destroy wakes pending states and joins every started worker.
            library.CliPP2ContinuationOwnerDestroy(owner)
    return outputs


def _result(library, owner, task, capacity, tree, regions, missing, keep_raw_phi, check):
    x = np.empty((tree.n, regions)) if keep_raw_phi else None
    labels = np.empty(tree.n, dtype=np.int32)
    mask = np.empty(tree.n-1, dtype=np.uint8)
    report = _ContinuationReport()
    check(library.CliPP2ContinuationOwnerResult(owner, task, None if x is None else _ptr(x),
                                              _ptr(labels), _ptr(mask), ct.byref(report)))
    return {'labels': labels.astype(np.int64), 'cuts': np.flatnonzero(mask), 'requested_k': capacity,
            **({'raw_phi': x} if keep_raw_phi else {}),
            'diagnostics': _continuation_diagnostics(report, tree, missing)}
