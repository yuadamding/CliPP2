"""Experimental resumable weight controller; not a production dispatch path.

The numerical evaluator is supplied by one external owner. This module never
materializes device columns or substitutes CPU likelihood calculations. SLSQP
uses the qualified SciPy binary; request scheduling is independent of its state.

The SLSQP orchestration and Brent transitions are adapted from SciPy 1.18.0:
scipy/optimize/_slsqp_py.py and scipy/optimize/Zeros/brentq.c (Charles Harris).

Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path
from types import MappingProxyType

import numpy as np
import scipy
from scipy.linalg.lapack import HAS_ILP64
from scipy.optimize import _slsqp_py

from ..io.data import readonly_array
from .scoring import WeightConvergenceError


_WRAPPER_SHA256 = "9e97132bfb5773096675f03b0adb6f5ed67510666f7f2c5da22d0e9669c2e338"
_UNSET = object()
CONTROLLER_ID = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def controller_environment():
    """Fail closed on an unqualified private interface; record the actual ELF."""
    import scipy.optimize._slsqplib as native

    wrapper = hashlib.sha256(Path(_slsqp_py.__file__).read_bytes()).hexdigest()
    if scipy.__version__ != "1.18.0" or wrapper != _WRAPPER_SHA256 or HAS_ILP64:
        raise RuntimeError("Resumable weights require qualified SciPy 1.18.0 LP64 SLSQP")
    return MappingProxyType({
        "scipy": scipy.__version__, "slsqp_wrapper_sha256": wrapper,
        "slsqp_binary_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
        "controller_sha256": CONTROLLER_ID, "indices_dtype": "int32",
    })


@dataclass(frozen=True, slots=True)
class WeightRequest:
    """One logical operation; endpoint derivatives may need no physical launch."""

    operation: str
    weights: np.ndarray | None = None
    grow: int | None = None
    shrink: int | None = None
    fraction: float | None = None
    result: object = None
    error: BaseException | None = None


def slsqp_workspace_size(q):
    """Owned native arrays, excluding Python state and evaluator storage."""
    if type(q) is not int or q < 1:
        raise ValueError("q must be a positive integer")
    buffer = q*(q+1)//2 + 10*q*q + 35*q + 30
    return 8*(buffer + 7*q + 4) + 4*(2*q + 3)


def _slsqp(initial, objective, gradient, trace=None):
    """Generator adapter to the unchanged SciPy reverse-communication core."""
    controller_environment()
    x = np.clip(initial, 0., 1.)
    n = len(x)
    # ScalarFunction caches f and g independently and compares x by array_equal,
    # unlike the outer device oracle's exact-byte evaluation cache.
    sf_x = x.copy()
    sf_f = yield from objective(sf_x)
    sf_g = yield from gradient(sf_x)
    f_updated = g_updated = True

    def value(at, need_gradient=False):
        nonlocal sf_x, sf_f, sf_g, f_updated, g_updated
        if not np.array_equal(at, sf_x):
            sf_x = at.copy()
            f_updated = g_updated = False
        if need_gradient:
            if not g_updated:
                sf_g = yield from gradient(sf_x.copy())
                g_updated = True
            return sf_g
        if not f_updated:
            sf_f = yield from objective(sf_x.copy())
            f_updated = True
        return sf_f

    state = dict.fromkeys(("acc", "alpha", "f0", "gs", "h1", "h2", "h3", "h4", "t", "t0", "tol"), 0.)
    state.update(dict.fromkeys(("exact", "inconsistent", "reset", "iter", "itermax",
                               "line", "m", "meq", "mode", "n"), 0))
    state.update(acc=1e-13, tol=1e-12, itermax=2000, m=1, meq=1, n=n)
    buffer = np.zeros(n*(n+1)//2 + 10*n*n + 35*n + 30)
    indices = np.zeros(2*n+3, dtype=np.int32)
    mult = np.zeros(2*n+3)
    lower, upper = np.zeros(n), np.ones(n)
    normals = np.ones((1, n), order="F")
    constraints = np.array([x.sum()-1.])
    fx, g = sf_f, sf_g
    while True:
        _slsqp_py.slsqp(state, fx, g, normals, constraints, x, mult, lower, upper, buffer, indices)
        if trace is not None:
            trace(dict(state), readonly_array(x))
        if state["mode"] == 1:
            fx = yield from value(x)
            constraints[0] = x.sum()-1.
        if state["mode"] == -1:
            g = yield from value(x, True)
            normals[:] = 1.
        if abs(state["mode"]) != 1:
            return x, state["iter"]


def _brent(derivative):
    """SciPy 1.18.0 brentq transitions, yielding only derivative requests."""
    xpre, xcur, xblk = 0., 1., 0.
    fblk = spre = scur = 0.
    calls = 0

    def evaluate(x):
        nonlocal calls
        result = yield from derivative(x)
        calls += 1
        if np.isnan(result):
            error = ValueError(f"The function value at x={x} is NaN; solver cannot continue.")
            error._x, error._function_calls = x, calls
            raise error
        return result

    fpre = yield from evaluate(xpre)
    fcur = yield from evaluate(xcur)
    if fpre == 0:
        return xpre
    if fcur == 0:
        return xcur
    if np.signbit(fpre) == np.signbit(fcur):
        raise ValueError("f(a) and f(b) must have different signs")
    for _ in range(100):
        if fpre != 0 and fcur != 0 and np.signbit(fpre) != np.signbit(fcur):
            xblk, fblk = xpre, fpre
            spre = scur = xcur-xpre
        if abs(fblk) < abs(fcur):
            # Sequential assignments are intentional (not a three-way swap).
            xpre = xcur
            xcur = xblk
            xblk = xpre
            fpre = fcur
            fcur = fblk
            fblk = fpre
        delta = (1e-14 + (4*np.finfo(float).eps)*abs(xcur))/2
        sbis = (xblk-xcur)/2
        if fcur == 0 or abs(sbis) < delta:
            return xcur
        if abs(spre) > delta and abs(fcur) < abs(fpre):
            if xpre == xblk:
                trial = -fcur*(xcur-xpre)/(fcur-fpre)
            else:
                dpre = (fpre-fcur)/(xpre-xcur)
                dblk = (fblk-fcur)/(xblk-xcur)
                trial = -fcur*(fblk*dblk-fpre*dpre)/(dblk*dpre*(fblk-fpre))
            if 2*abs(trial) < min(abs(spre), 3*abs(sbis)-delta):
                spre, scur = scur, trial
            else:
                spre = scur = sbis
        else:
            spre = scur = sbis
        xpre, fpre = xcur, fcur
        xcur += scur if abs(scur) > delta else (delta if sbis > 0 else -delta)
        fcur = yield from evaluate(xcur)
    raise RuntimeError("Failed to converge after 100 iterations.")


class WeightController:
    """One independent, cancellable state; no shared optimizer or device lease.

    Advance returns an immutable request. Supply exactly one reply for each
    numerical request. COMPLETE and ERROR are terminal; ERROR retains the actual
    exception, including callback exception identity. Close abandons the state.
    """

    def __init__(self, shape, initial_weights=None, *, telemetry=None, progress_callback=None,
                 slsqp_trace=None, memory_bytes=None):
        if len(shape) != 2 or any(type(v) is not int or v < 1 for v in shape):
            raise ValueError("Weight shape must be positive integer (N, q)")
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError("progress_callback must be callable or None")
        if memory_bytes is not None:
            if type(memory_bytes) is not int or memory_bytes < 0:
                raise ValueError("memory_bytes must be a nonnegative integer")
            if shape[1] > 1 and slsqp_workspace_size(shape[1]) > memory_bytes:
                raise MemoryError("SLSQP state exceeds the supplied native-array budget")
        self.shape, self.telemetry = tuple(shape), telemetry
        self.callback, self.trace = progress_callback, slsqp_trace
        self.callback_error = self.scientific_error = None
        self.completed_requests = 0
        self._started = self._terminal = False
        initial = None if initial_weights is None else np.array(initial_weights, copy=True)
        self._generator = self._run(initial)

    def advance(self, reply=_UNSET):
        if self._terminal:
            raise RuntimeError("Weight controller is already terminal")
        if self._started == (reply is _UNSET):
            raise ValueError("Provide a reply only after a numerical request")
        try:
            if not self._started:
                self._started = True
                return next(self._generator)
            return self._generator.send(reply)
        except StopIteration as complete:
            self._terminal = True
            return WeightRequest("complete", result=complete.value)
        except BaseException as error:
            self._terminal = True
            return WeightRequest("error", error=error)

    def close(self):
        self._generator.close()
        self._terminal = True

    def fail(self, error):
        """Terminate after an evaluator failure without turning it into rejection."""
        if self._terminal:
            raise RuntimeError("Weight controller is already terminal")
        if not isinstance(error, BaseException):
            raise TypeError("Evaluator failure must be an exception")
        self.close()
        return WeightRequest("error", error=error)

    def _record(self, name, amount=1):
        if self.telemetry is not None:
            self.telemetry[name] = self.telemetry.get(name, 0) + amount

    def _completed(self, request):
        if self.callback is not None:
            self.completed_requests += 1
            try:
                self.callback({"operation": "weight_fit", "boundary": "request_complete",
                               "request": request, "completed_requests": self.completed_requests,
                               "occupied_q": self.shape[1]})
            except BaseException as error:
                self.callback_error = error
                raise

    def _run(self, initial):
        n, q = self.shape
        self._record("weight_problem_calls")
        weights = np.ones(q)/q if initial is None else np.asarray(initial, dtype=float)
        if (weights.shape != (q,) or not np.isfinite(weights).all()
                or np.any(weights < 0) or not weights.max() > 0):
            raise ValueError("Initial cluster weights must be nonnegative, finite and nonzero")
        weights = weights/weights.max()
        weights /= weights.sum()
        if q == 1:
            result = yield WeightRequest("evaluate", readonly_array(np.ones(1)))
            self._completed("evaluate")
            return self._result(np.ones(1), result[2], 0., 0.)
        evaluation_key, evaluation = None, None

        def evaluate(at):
            nonlocal evaluation_key, evaluation
            key = at.tobytes()
            if key != evaluation_key:
                raw = yield WeightRequest("evaluate", readonly_array(at))
                evaluation = (raw[0], readonly_array(raw[1]), raw[2])
                evaluation_key = key
                self._record("weight_device_evaluations")
                self._completed("evaluate")
            else:
                self._record("weight_device_evaluation_cache_hits")
            return evaluation

        def objective(at):
            result = yield from evaluate(at)
            return result[0]

        def gradient(at):
            result = yield from evaluate(at)
            return result[1]

        if not np.isfinite((yield from objective(weights))):
            weights = np.ones(q)/q
        boundary = weights.copy()
        if np.max(-(yield from gradient(weights))) > 1e100:
            weights = np.ones(q)/q
        self._record("weight_slsqp_calls")
        candidate, iterations = yield from _slsqp(weights, objective, gradient, self.trace)
        self._record("weight_slsqp_iterations", iterations)
        if np.all(np.isfinite(candidate)):
            candidate = np.maximum(candidate, 0.)
            if candidate.sum() > 0:
                candidate /= candidate.sum()
                if (yield from objective(candidate)) <= (yield from objective(weights)):
                    weights = candidate
        if (yield from objective(boundary)) < (yield from objective(weights)):
            weights = boundary
        for _ in range(10000):
            score = -(yield from gradient(weights))
            positive = np.flatnonzero(weights > 0)
            active_gap = float(score.max()-score[positive].min())
            if max(0., float(score.max()-1.)) <= 1e-8 and active_gap <= 1e-8:
                break
            grow, shrink = int(np.argmax(score)), int(positive[np.argmin(score[positive])])
            if grow == shrink:
                break
            self._record("weight_transfer_steps")
            yield WeightRequest("prepare_transfer", readonly_array(weights), grow, shrink)
            self._record("weight_device_transfer_preparations")
            self._completed("prepare_transfer")

            def derivative(fraction):
                self._record("weight_device_transfer_trials")
                value = yield WeightRequest("transfer_derivative", fraction=float(fraction))
                if fraction != 1.:
                    self._completed("transfer_derivative")
                return value

            fraction = 1. if (yield from derivative(1.)) >= 0 else (yield from _brent(derivative))
            transfer = fraction*weights[shrink]
            weights[grow] += transfer
            weights[shrink] = 0. if fraction == 1. else weights[shrink]-transfer
            weights /= weights.sum()
        score = -(yield from gradient(weights))
        optimality_gap = n*max(0., float(score.max()-1.))
        active_gap = float(score.max()-score[weights > 0].min())
        if not np.isfinite([(yield from objective(weights)), optimality_gap, active_gap]).all():
            raise FloatingPointError("Cluster-weight likelihood optimization produced invalid arithmetic")
        if optimality_gap > n*1e-8 or active_gap > 1e-8:
            self._record("weight_convergence_failures")
            self.scientific_error = WeightConvergenceError(
                "Cluster-weight likelihood optimization did not converge")
            raise self.scientific_error
        final = yield from evaluate(weights)
        return self._result(weights, final[2], optimality_gap, active_gap)

    @staticmethod
    def _result(weights, likelihood, optimality_gap, active_gap):
        return MappingProxyType({"cluster_weights": readonly_array(weights), "log_likelihood": likelihood,
                                 "weight_optimality_gap": optimality_gap,
                                 "weight_active_score_gap": active_gap})
