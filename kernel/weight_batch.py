"""Experimental single-owner scheduling of independent weight controllers.

Default search dispatch remains serial. The caller owns all live native
problems (normally in an ExitStack) and admits their device memory separately.
This scheduler owns bounded host controller states, never closes caller problems,
and returns terminal outcomes in original input order, not completion order.
"""
from time import perf_counter

import numpy as np

from .weight_controller import WeightController
from .scoring import completed_weight
from .backend import WeightBatchCancelled, WeightOutcome, controller_reservation


def fit_weight_batch(backend, problems, initial_weights=None, *, max_active=8,
                     state_budget_bytes, telemetry=None, progress_callback=None,
                     cancelled=None):
    """Return ordered COMPLETE or genuine solver-gate ERROR outcomes.

    ``backend.evaluate_weight_requests([(problem, request), ...])`` must return
    exactly one reply per request in that order. It may compute in any order,
    but owns validation and dispatch; this function has no numerical fallback.
    ``progress_callback(index, event)`` receives one completed-solve snapshot.
    A callback/evaluator/resource failure aborts the batch and propagates the
    original exception. Genuine final weight-gate failures remain ordered error
    outcomes so the caller can register ordinary scientific rejection.

    All inputs are checked before submitting work. Admission is input-ordered;
    ready partial batches execute immediately, including a single live state.
    No wait-for-full policy, thread pool or elapsed-time ceiling is introduced.
    """
    if type(max_active) is not int or max_active < 1:
        raise ValueError("max_active must be a positive integer")
    if type(state_budget_bytes) is not int or state_budget_bytes < 0:
        raise ValueError("state_budget_bytes must be a nonnegative integer")
    for callback in (progress_callback, cancelled):
        if callback is not None and not callable(callback):
            raise TypeError("Callbacks must be callable or None")
    if not callable(getattr(backend, "evaluate_weight_requests", None)):
        raise TypeError("Backend must implement ordered evaluate_weight_requests")
    problems = tuple(problems)
    starts = [None]*len(problems) if initial_weights is None else list(initial_weights)
    if len(starts) != len(problems):
        raise ValueError("Provide one initial-weight vector per problem")
    identities, sizes = set(), []
    for index, problem in enumerate(problems):
        if problem.backend() is not backend:
            raise ValueError("All weight problems must belong to the submission owner")
        identity = getattr(problem, "handle", id(problem))
        if identity in identities:
            raise ValueError("Independent states cannot share one mutable mixture problem")
        identities.add(identity)
        shape = tuple(problem.shape)
        if len(shape) != 2 or any(type(value) is not int or value < 1 for value in shape):
            raise ValueError("Weight shape must be positive integer (N, q)")
        q = shape[1]
        sizes.append(controller_reservation(q))
        if sizes[-1] > state_budget_bytes:
            raise MemoryError("One weight controller exceeds the live-state budget")
        if starts[index] is not None:
            start = np.array(starts[index], dtype=float, copy=True)
            if (start.shape != (q,) or not np.isfinite(start).all()
                    or np.any(start < 0) or not start.max() > 0):
                raise ValueError("Initial cluster weights must be nonnegative, finite and nonzero")
            starts[index] = start
    stats = {} if telemetry is None else telemetry

    def count(name, amount=1):
        stats[name] = stats.get(name, 0) + amount

    def peak(name, amount):
        stats[name] = max(stats.get(name, 0), amount)

    def check_cancelled():
        if cancelled is not None and cancelled():
            raise WeightBatchCancelled("Weight batch cancelled at a request boundary")

    count("weight_batch_initial_bytes", sum(value.nbytes for value in starts if value is not None))
    peak("weight_batch_state_limit_bytes", state_budget_bytes)
    if not problems:
        return ()
    active, outcomes = {}, [None]*len(problems)
    next_index = live_bytes = 0

    def retire(index):
        nonlocal live_bytes
        controller, event, _ = active[index]
        if event.operation not in ("complete", "error"):
            return
        if event.operation == "error" and event.error is not controller.scientific_error:
            raise event.error
        outcomes[index] = WeightOutcome(event.operation, result=event.result, error=event.error)
        count("weight_batch_completed")
        if event.operation == "error":
            count("weight_batch_scientific_failures")
        controller.close()
        del active[index]
        live_bytes -= sizes[index]
        completed_weight(None if progress_callback is None else
                         lambda record: progress_callback(index, record),
                         problems[index].shape[1], failed=event.operation == "error")

    with backend.weight_submission():
        try:
            while next_index < len(problems) or active:
                check_cancelled()
                while next_index < len(problems) and len(active) < max_active:
                    index, size = next_index, sizes[next_index]
                    if live_bytes + size > state_budget_bytes:
                        break
                    controller = WeightController(tuple(problems[index].shape), starts[index],
                                                  telemetry=stats)
                    # Register before advancing: callbacks or native interface errors
                    # still leave every created controller owned by the finally path.
                    active[index] = [controller, None, perf_counter()]
                    live_bytes += size
                    starts[index] = None
                    next_index += 1
                    peak("weight_batch_live_peak", len(active))
                    peak("weight_batch_state_bytes_peak", live_bytes)
                    active[index][1] = controller.advance()
                    retire(index)
                if not active:
                    continue
                check_cancelled()
                ready = tuple(active)
                submitted_at = perf_counter()
                requests = []
                for index in ready:
                    _, request, ready_at = active[index]
                    requests.append((problems[index], request))
                    count("weight_batch_queue_seconds", submitted_at-ready_at)
                count("weight_batch_submissions")
                count("weight_batch_logical_requests", len(requests))
                if len(requests) < max_active:
                    count("weight_batch_partial_submissions")
                peak("weight_batch_size_max", len(requests))
                replies = tuple(backend.evaluate_weight_requests(requests))
                if len(replies) != len(requests):
                    raise RuntimeError("Batch evaluator returned an incomplete ordered reply set")
                check_cancelled()
                for index, reply in zip(ready, replies):
                    active[index][1] = active[index][0].advance(reply)
                    active[index][2] = perf_counter()
                    retire(index)
            return tuple(outcomes)
        finally:
            for controller, _, _ in active.values():
                controller.close()
