"""Private compact weight completions for ordered coarsening search.

Physical preparation may advance lower caches, never candidate publication or
completed-weight LRU state. Completed records retain no prepared partitions,
generators, device leases or exception frames. Every logical turn revalidates
the ordinary bank, conditional preparation and exact weight/cache identity.
"""
from collections import deque
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass
import sys
from time import perf_counter

import numpy as np

from .model import _freeze
from .scoring import WeightConvergenceError, completed_weight

_KNOWN_INELIGIBLE = object()


def _retained_size(value):
    """Conservatively charge only compact immutable, pointer-free payloads."""
    if isinstance(value, Mapping):
        return 256+64*len(value)+sum(_retained_size(key)+_retained_size(item)
                                    for key, item in value.items())
    if isinstance(value, tuple):
        return sys.getsizeof(value)+sum(map(_retained_size, value))
    if isinstance(value, np.ndarray):
        if value.ndim > 1 or value.dtype.hasobject:
            raise TypeError('Speculative completion arrays must be compact numeric vectors')
        return 256+value.nbytes
    if value is None or isinstance(value, (str, bytes, bool, int, float, np.generic)):
        return 64+sys.getsizeof(value)
    raise TypeError('Speculative completions may not retain live or opaque objects')


@dataclass(frozen=True, slots=True)
class _Completion:
    key: tuple
    result: object
    error_message: str | None
    size: int


@dataclass
class _Ticket:
    partition: object
    steps: object = None
    request: object = None
    key: object = None
    outcome: object = None

    def close_preparation(self):
        if self.steps is not None:
            self.steps.close()
        self.partition = self.steps = self.request = None
        if hasattr(self, 'problem'):
            del self.problem

    def close(self):
        self.close_preparation()
        self.key = self.outcome = None


class CoarseningBatch:
    """Bounded independent work with original-order publication and cache touches.

    Consume a completed batch before refilling it. Eligible/refinement/repair
    barriers retain compact outcomes but never their native resources. Exact
    key/cache/observer revalidation at the logical turn decides whether to adopt.
    """
    def __init__(self, bank, subsets, *, max_active=8):
        if type(max_active) is not int or max_active < 1:
            raise ValueError('max_active must be a positive integer')
        self.bank, self.model = bank, bank.model
        minimum = self.model.refit_batch_reservation(1)
        self.subsets = iter(subsets)
        self.limit = min(max_active, max(1, bank.weight_batch_state_bytes//minimum))
        self.queue, self.current = deque(), None
        self.exhausted = False
        self.private_bytes = 0
        self._peak('private_limit_bytes', bank.weight_batch_state_bytes)

    def _count(self, name, amount=1):
        key = 'coarsening_speculation_'+name
        self.model.telemetry[key] = self.model.telemetry.get(key, 0)+amount

    def _peak(self, name, amount):
        key = 'coarsening_speculation_'+name
        self.model.telemetry[key] = max(self.model.telemetry.get(key, 0), amount)

    def _known(self, cuts):
        membership = ('tree', cuts)
        return (membership in self.bank._structural or
                (membership, None, self.bank.numerical_policy) in self.bank._attempts)

    def _can_cross_known(self, cuts):
        if self.bank.progress is not None or self.bank.request_progress is not None:
            return False
        membership = ('tree', cuts)
        return (membership in self.bank._structural or
                self.bank._attempts.get((membership, None, self.bank.numerical_policy),
                                       _KNOWN_INELIGIBLE) is None)

    def _make_room(self, required):
        """Evict farthest future completions, never their logical cut descriptors."""
        budget = self.bank.weight_batch_state_bytes
        for item in reversed(self.queue):
            if self.private_bytes+required <= budget:
                break
            if isinstance(item[1], _Completion):
                self.private_bytes -= item[1].size
                item[1] = None
                self._count('budget_evictions')
        return self.private_bytes+required <= budget

    def _detach(self, ticket):
        outcome = ticket.outcome
        if outcome.operation == 'error':
            if not isinstance(outcome.error, WeightConvergenceError):
                raise outcome.error
            result, error = None, str(outcome.error)
        elif outcome.operation == 'complete':
            result, error = _freeze(outcome.result), None
        else:
            raise RuntimeError('Speculative weight solver returned a nonterminal outcome')
        key = ticket.key
        size = 512+_retained_size((key, result, error))
        return _Completion(key, result, error, size)

    def _fill(self):
        # Consume every computed completion before refilling. Unprepared cut
        # descriptors behind a bank/observer barrier are not a computed batch:
        # after that turn, they can form the next ready numerical submission.
        if any(isinstance(item[1], _Completion) for item in self.queue):
            raise RuntimeError('Consume the speculative batch before refilling')
        for item in self.queue:
            item[1] = None  # Recheck formerly inert cache hits at this boundary.
        while len(self.queue) < self.limit and not self.exhausted:
            try:
                self.queue.append([tuple(next(self.subsets)), None])
            except StopIteration:
                self.exhausted = True
        pending, reserved = [], 0
        budget = self.bank.weight_batch_state_bytes
        started = perf_counter()
        try:
            with ExitStack() as leases:
                for item in self.queue:
                    cuts = item[0]
                    if self._known(cuts):
                        if self._can_cross_known(cuts):
                            item[1] = _KNOWN_INELIGIBLE
                            self._count('known_ineligible_crossings')
                            continue
                        break
                    q = len(cuts)+1
                    charge = self.model.refit_batch_reservation(q)
                    if charge > budget:
                        self._count('serial_budget_fallbacks')
                        break
                    admission = self.model._weight_admission(q, [len(ticket.request[1]) for _, ticket in pending
                                                                if ticket.request is not None])
                    column_bytes = 8*self.model.n*q+8*q*(self.model.r+1)+4
                    if reserved+charge > budget or (pending and (
                            admission['device_required']+column_bytes > admission['device_available'] or
                            admission['host_required']+64*q > admission['host_available'])):
                        break
                    partition = self.bank.prepare(cuts)
                    ticket = _Ticket(partition)
                    pending.append((item, ticket))
                    reserved += charge
                    ticket.steps = self.model._refit_steps(partition.labels, partition.groups, None,
                        partition.coordinate_fits, partition.columns, partition.membership,
                        record_weight_time=False)
                    try:
                        ticket.request = next(ticket.steps)
                    except StopIteration:
                        # Reconstruct structural/conditional outcomes through
                        # the normal path at their logical turn; retain no fit.
                        ticket.close()
                    else:
                        columns, counts, _, _ = ticket.request
                        ticket.key = self.model._weight_key(columns, counts)
                        ticket.problem = leases.enter_context(self.model.backend.mixture_problem(columns.columns))
                    self._count('prepared')
                active = [ticket for _, ticket in pending if ticket.request is not None]
                if active:
                    physical = {}
                    before = perf_counter()
                    self._count('weight_fits', len(active))
                    self._count('physical_weight_fits', len(active))
                    try:
                        outcomes = self.model._solve_weight_problems([ticket.problem for ticket in active],
                            [ticket.request[1] for ticket in active], max_active=self.limit,
                            state_budget_bytes=budget, telemetry=physical)
                    finally:
                        elapsed = perf_counter()-before
                        self.model.telemetry['weight_seconds'] += elapsed
                        self._count('weight_seconds', elapsed)
                        for key, value in physical.items():
                            self._count('physical_'+key, value)
                    if len(outcomes) != len(active):
                        raise RuntimeError('Speculative weight solver returned incomplete outcomes')
                    for ticket, outcome in zip(active, outcomes):
                        ticket.outcome = outcome
            # Close generators/column owners before retaining any completion.
            # Only compact immutable output survives; opaque handles and error
            # tracebacks are never put in the queue.
            for _, ticket in pending:
                ticket.close_preparation()
            partition = columns = counts = None
            for item, ticket in pending:
                completion = self._detach(ticket) if ticket.outcome is not None else None
                ticket.close()
                if completion is not None:
                    if self.private_bytes+completion.size <= budget:
                        item[1] = completion
                        self.private_bytes += completion.size
                        self._count('retained_completions')
                        self._peak('private_bytes_peak', self.private_bytes)
                    else:
                        self._count('budget_evictions')
        except BaseException:
            self.close()
            raise
        finally:
            for _, ticket in pending:
                ticket.close()
            if pending:
                self.bank.refit_seconds += perf_counter()-started

    def __iter__(self):
        return self

    def __next__(self):
        if self.current is not None:
            raise RuntimeError('Consume the current coarsening before advancing')
        if (not self.queue or (self.queue[0][1] is None and
                not any(isinstance(item[1], _Completion) for item in self.queue))):
            self._fill()
        if not self.queue:
            raise StopIteration
        self.current = self.queue.popleft()
        if isinstance(self.current[1], _Completion):
            self.private_bytes -= self.current[1].size
        return self.current[0]

    def _refit_current(self, cuts, completion, provenance):
        bank, model = self.bank, self.model
        bank.refit_calls += 1
        started = perf_counter()
        partition = bank.prepare(cuts)
        self._count('logical_preparations')
        self._count('repeated_preparations')
        notify = (None if bank.request_progress is None else
                  lambda event: bank.request_progress(bank, provenance, event))
        steps = model._refit_steps(partition.labels, partition.groups, None,
            partition.coordinate_fits, partition.columns, partition.membership,
            progress_callback=notify, record_weight_time=False)
        try:
            try:
                columns, counts, matrix, notify = next(steps)
            except StopIteration as done:
                return bank._commit(partition, None, provenance, done.value)
            key = model._weight_key(columns, counts)
            mixture = model._cached_weight_result(key, columns, notify)
            if mixture is not None:
                self._count('discarded_weight_hit')
                self._count('cache_supersessions')
            elif completion is not None and key == completion.key:
                model.telemetry['weight_solves'] += 1
                completed_weight(notify, len(counts), failed=completion.error_message is not None)
                self._count('adopted_weight_fits')
                if completion.error_message is not None:
                    fitted = model._finish_refit_steps(steps, error=WeightConvergenceError(completion.error_message))
                    return bank._commit(partition, None, provenance, fitted)
                mixture = model._cache_weight_result(key, columns, counts, completion.result)
            else:
                if completion is not None:
                    self._count('stale_key_discards')
                before = perf_counter()
                self._count('physical_weight_fits')
                try:
                    mixture = model._uncached_weight_result(key, columns, counts, matrix, notify)
                except WeightConvergenceError as error:
                    fitted = model._finish_refit_steps(steps, error=error)
                    return bank._commit(partition, None, provenance, fitted)
                finally:
                    model.telemetry['weight_seconds'] += perf_counter()-before
            fitted = model._finish_refit_steps(steps, mixture)
            return bank._commit(partition, None, provenance, fitted)
        finally:
            steps.close()
            bank.refit_seconds += perf_counter()-started

    def consider(self, provenance):
        if self.current is None:
            raise RuntimeError('No current coarsening')
        cuts, completion = self.current
        self.current = None
        bank = self.bank
        if completion is _KNOWN_INELIGIBLE:
            if not self._can_cross_known(cuts):
                self.expire('known_change')
            return bank.consider_cuts(cuts, None, provenance)
        if completion is None:
            before = self.model.telemetry.get('weight_solves', 0)
            try:
                return bank.consider_cuts(cuts, None, provenance)
            finally:
                self._count('physical_weight_fits', self.model.telemetry.get('weight_solves', 0)-before)
        provenance = bank.normalize_provenance(provenance)
        found, candidate = bank._cached(('tree', cuts), None, provenance)
        if found:
            self._count('discarded_bank_hit')
        else:
            charge = self.model.refit_batch_reservation(len(cuts)+1)
            if not self._make_room(charge+completion.size):
                # Drop this reference before entering the ordinary refit; an
                # outer frame must not invisibly keep an evicted record live.
                completion = None
                self._count('budget_evictions')
                self._make_room(charge)
            candidate = self._refit_current(cuts, completion, provenance)
            self._count('published')
        if candidate is not None:
            self.expire('eligible')
        if bank.progress is not None:
            bank.progress(bank, provenance)
        return candidate

    def expire(self, reason='barrier'):
        """Barriers release no resources: only exact compact completions remain."""
        if reason == 'close':
            self.queue.clear()
            self.private_bytes = 0
        else:
            self._count('retained_'+reason, sum(isinstance(item[1], _Completion) for item in self.queue))

    def close(self):
        self.current = None
        self.expire('close')

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
