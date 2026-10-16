"""Ordered independent continuation with one owning likelihood executor.

Host forest work may overlap; each round waits for every active task's next
likelihood request or completion. This avoids timing-dependent batch membership
and never lets worker completion order mutate the candidate search.
"""
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Event

import numpy as np


def ordered_continuations(model, tree, pilot, capacities, policy, workers, solve):
    results = []
    for first in range(0, len(capacities), workers):
        group = capacities[first:first+workers]
        messages = Queue()

        class Proxy:
            def __init__(self, index):
                self.index = index

            def __getattr__(self, name):
                return getattr(model, name)

            def node_terms(self, values):
                ready, reply = Event(), []
                messages.put(('evaluate', self.index, (np.array(values, copy=True), ready, reply)))
                ready.wait()
                if isinstance(reply[0], BaseException):
                    raise reply[0]
                return reply[0]

        def run(index, capacity):
            try:
                return solve(Proxy(index), tree, pilot, [capacity], policy=policy, _batch=False)[0]
            finally:
                messages.put(('done', index, None))

        with ThreadPoolExecutor(max_workers=len(group), thread_name_prefix='clipp2-continuation') as pool:
            futures = [pool.submit(run, index, k) for index, k in enumerate(group)]
            active, waiting, failure = set(range(len(group))), {}, None
            while active:
                try:
                    while len(waiting) < len(active):
                        kind, index, request = messages.get()
                        if kind == 'done':
                            active.remove(index)
                        else:
                            waiting[index] = request
                    if not waiting:
                        continue
                    indices = sorted(waiting)
                    if failure is not None:
                        raise failure
                    points = np.stack([waiting[i][0] for i in indices])
                    values = model.node_terms_batch(points)
                    if len(values) != 3 or any(np.shape(v) != points.shape for v in values):
                        raise ValueError('Batched likelihood must return three matching B by N by R arrays')
                    replies = {index: tuple(v[offset] for v in values)
                               for offset, index in enumerate(indices)}
                except BaseException as error:
                    failure = error
                    # Wake every current waiter, including requests accumulated
                    # before a dispatch/shape error or operator interruption.
                    indices = sorted(waiting)
                    replies = {}
                for index in indices:
                    _, event, reply = waiting[index]
                    reply.append(failure if failure is not None else replies[index])
                    event.set()
                waiting.clear()
            # Exceptions propagate after all request waiters have been released.
            if failure is not None:
                raise failure
            results.extend(future.result() for future in futures)
    return results
