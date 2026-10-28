"""Private weight speculation for the unchanged ordered coarsening search.

Only independent, no-incumbent coarsenings may enter this owner. Exact lower
preparation/column caches can advance as physical work, but candidate state,
completed-weight caches and progress are published only at the logical turn.
Every eligible/refinement barrier expires unused tickets, not their cut IDs.
"""
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass
from time import perf_counter

from .model import _replay_weight_progress

_KNOWN_INELIGIBLE = object()

@dataclass
class _Ticket:
    partition: object
    steps: object = None
    request: object = None
    key: object = None
    fitted: object = None
    outcome: object = None
    events: object = None

    def close(self):
        if self.steps is not None:
            self.steps.close()
        self.steps = self.request = None


class CoarseningBatch:
    """Bounded lookahead with an ordinary consider/repair/refine publication barrier.

Use ``with CoarseningBatch(bank, subsets, max_active=8) as stream``; each yielded
cut tuple must be consumed by ``stream.consider(provenance)``. Call ``expire``
before refinement or structural repair. CUDA execution failures abort the owner;
scientific weight failures remain private until that ticket's logical turn.
"""
    _names = ('evaluate', 'prepare_transfer', 'transfer_derivative', 'resident_fit')

    def __init__(self, bank, subsets, *, max_active=8):
        if type(max_active) is not int or max_active < 1:
            raise ValueError('max_active must be a positive integer')
        self.bank, self.model = bank, bank.model
        minimum = self.model.refit_batch_reservation(1, progress=bank.request_progress is not None)
        self.subsets = iter(subsets)
        self.limit = min(max_active, max(1, bank.weight_batch_state_bytes//minimum))
        self.queue, self.current = deque(), None
        self.exhausted = False

    def _count(self, name, amount=1):
        key = 'coarsening_speculation_'+name
        self.model.telemetry[key] = self.model.telemetry.get(key, 0)+amount

    def _known(self, cuts):
        membership = ('tree', cuts)
        return (membership in self.bank._structural or
                (membership, None, self.bank.numerical_policy) in self.bank._attempts)

    def _can_cross_known(self, cuts):
        """Only observer-free rejection hits are inert at their logical turn."""
        if self.bank.progress is not None or self.bank.request_progress is not None:
            return False
        membership = ('tree', cuts)
        return (membership in self.bank._structural or
                self.bank._attempts.get((membership, None, self.bank.numerical_policy),
                                       _KNOWN_INELIGIBLE) is None)

    def _fill(self):
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
                    cuts, existing = item
                    if existing is not None:
                        raise RuntimeError('Cannot refill an unconsumed speculative batch')
                    if self._known(cuts):
                        if self._can_cross_known(cuts):
                            item[1] = _KNOWN_INELIGIBLE
                            self._count('known_ineligible_crossings')
                            continue
                        break  # Ordinary cache/provenance callback remains an ordering barrier.
                    q = len(cuts)+1
                    charge = self.model.refit_batch_reservation(q, progress=self.bank.request_progress is not None)
                    if charge > budget:
                        self._count('serial_budget_fallbacks')
                        break
                    admission = self.model._weight_admission(q, [len(ticket.request[1]) for ticket in pending
                                                                if ticket.request is not None])
                    columns = 8*self.model.n*q+8*q*(self.model.r+1)+4
                    if reserved+charge > budget or (pending and (
                            admission['device_required']+columns > admission['device_available'] or
                            admission['host_required']+64*q > admission['host_available'])):
                        break
                    partition = self.bank.prepare(cuts)
                    ticket = _Ticket(partition, events=bytearray())
                    # A private notification sink lets _refit_steps retain its
                    # callback-exception identity guard without observing early.
                    ticket.steps = self.model._refit_steps(partition.labels, partition.groups, None,
                        partition.coordinate_fits, partition.columns, partition.membership,
                        progress_callback=(None if self.bank.request_progress is None else lambda _: None),
                        record_weight_time=False)
                    # Register before any allocation/evaluation can fail.
                    item[1] = ticket
                    pending.append(ticket)
                    reserved += charge
                    try:
                        ticket.request = next(ticket.steps)
                    except StopIteration as done:
                        ticket.fitted = done.value
                        ticket.close()
                    else:
                        columns, counts, _, _ = ticket.request
                        ticket.key = self.model._weight_key(columns, counts)
                        # No completed-weight lookup here: earlier logical fits
                        # may evict, populate or touch that cache before this turn.
                        problem = leases.enter_context(self.model.backend.mixture_problem(columns.columns))
                        ticket.problem = problem
                    self._count('prepared')
                active = [ticket for ticket in pending if ticket.request is not None]
                if active:
                    def record(index, event):
                        ticket = active[index]
                        if len(ticket.events) >= 1200000+len(ticket.request[1]):
                            raise MemoryError('Speculative progress exceeds its reserved capacity')
                        ticket.events.append(self._names.index(event['request']))

                    physical = {}
                    before = perf_counter()
                    self._count('weight_fits', len(active))
                    try:
                        outcomes = self.model._solve_weight_problems([ticket.problem for ticket in active],
                            [ticket.request[1] for ticket in active], max_active=self.limit,
                            state_budget_bytes=budget, telemetry=physical,
                            progress_callback=record if self.bank.request_progress is not None else None)
                    finally:
                        elapsed = perf_counter()-before
                        self.model.telemetry['weight_seconds'] += elapsed
                        self._count('weight_seconds', elapsed)
                        for key, value in physical.items():
                            self._count('physical_'+key, value)
                    for ticket, outcome in zip(active, outcomes):
                        ticket.outcome = outcome
                    # Individual solve times overlap; only this batch total is
                    # a physical elapsed-time measurement, never sum per-state.
            for ticket in pending:
                if hasattr(ticket, 'problem'):
                    del ticket.problem  # Native arenas are closed before callbacks.
        except BaseException:
            self.close()
            raise
        finally:
            if pending:
                self.bank.refit_seconds += perf_counter()-started

    def __iter__(self):
        return self

    def __next__(self):
        if self.current is not None:
            raise RuntimeError('Consume the current coarsening before advancing')
        if not self.queue or self.queue[0][1] is None:
            self._fill()
        if not self.queue:
            raise StopIteration
        self.current = self.queue.popleft()
        return self.current[0]

    def consider(self, provenance):
        if self.current is None:
            raise RuntimeError('No current coarsening')
        cuts, ticket = self.current
        self.current = None
        bank, model = self.bank, self.model
        if ticket is _KNOWN_INELIGIBLE:
            # Cache ownership or observers may change between iterator turns.
            # Expire future private work before a formerly inert turn can act.
            if not self._can_cross_known(cuts):
                self.expire('known_change')
            return bank.consider_cuts(cuts, None, provenance)
        if ticket is None:
            return bank.consider_cuts(cuts, None, provenance)
        try:
            provenance = bank.normalize_provenance(provenance)
            # Recheck BEFORE weight-cache touches or progress: an earlier
            # callback/refinement may have made the speculative attempt redundant.
            found, candidate = bank._cached(('tree', cuts), None, provenance)
            if found:
                self._count('discarded_bank_hit')
            else:
                bank.refit_calls += 1
                started = perf_counter()
                fitted = ticket.fitted
                if ticket.request is not None:
                    columns, counts, _, _ = ticket.request
                    notify = (None if bank.request_progress is None else
                              lambda event: bank.request_progress(bank, provenance, event))
                    mixture = model._cached_weight_result(ticket.key, columns, notify)
                    if mixture is not None:
                        self._count('discarded_weight_hit')
                        fitted = model._finish_refit_steps(ticket.steps, mixture)
                    else:
                        model.telemetry['weight_solves'] += 1
                        if notify is not None:
                            _replay_weight_progress(ticket.events, self._names, counts, notify)
                        if ticket.outcome.operation == 'error':
                            fitted = model._finish_refit_steps(ticket.steps, error=ticket.outcome.error)
                        else:
                            mixture = model._cache_weight_result(ticket.key, columns, counts, ticket.outcome.result)
                            fitted = model._finish_refit_steps(ticket.steps, mixture)
                        self._count('adopted_weight_fits')
                bank.refit_seconds += perf_counter()-started
                candidate = bank._commit(ticket.partition, None, provenance, fitted)
                self._count('published')
            if candidate is not None:
                # Release future columns BEFORE any observer/refinement can
                # allocate, not only after it returns to the outer search.
                self.expire('eligible')
            if bank.progress is not None:
                bank.progress(bank, provenance)
            return candidate
        finally:
            ticket.close()

    def expire(self, reason='barrier'):
        """Drop private outcomes/leases while retaining every unvisited cut tuple."""
        for item in self.queue:
            if item[1] is _KNOWN_INELIGIBLE:
                item[1] = None
            elif item[1] is not None:
                item[1].close()
                item[1] = None
                self._count('discarded_'+reason)

    def close(self):
        if self.current is not None and self.current[1] not in (None, _KNOWN_INELIGIBLE):
            self.current[1].close()
        self.current = None
        self.expire('close')
        self.queue.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
