"""CliPP2 observed-mixture weights at fixed conditionally fitted centers."""
# Adapted from CliPP1.5.

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import logsumexp


def fit_mixture_weights(log_columns, initial_weights=None, *, telemetry=None):
    """Maximize the joint observed likelihood of an N-by-q log-emission matrix.

    This is a concave weight problem. The reported Frank-Wolfe gap bounds the
    remaining log-likelihood improvement at these centers, even with duplicate
    centers or boundary weights. Zero weights are returned for explicit repair
    by the caller; this optimizer does not change partition memberships.
    Center fitting itself remains conditional on the supplied hard partition;
    this function does not claim a joint mixture-center maximum.
    """

    def record(name, amount=1):
        if telemetry is not None:
            telemetry[name] = telemetry.get(name, 0) + amount

    record("weight_problem_calls")
    log_kernel = np.asarray(log_columns, dtype=float)
    if (log_kernel.ndim != 2 or not all(log_kernel.shape)
            or np.isnan(log_kernel).any() or np.isposinf(log_kernel).any()):
        raise ValueError("Expected a nonempty N-by-q matrix of finite or -inf log emissions")
    n, k = log_kernel.shape
    offset = np.max(log_kernel, axis=1)
    if np.any(~np.isfinite(offset)):
        raise ValueError("Every mutation needs finite likelihood in some cluster")
    weights = np.ones(k) / k if initial_weights is None else np.asarray(initial_weights, dtype=float)
    if (weights.shape != (k,) or not np.isfinite(weights).all()
            or np.any(weights < 0) or not weights.max() > 0):
        raise ValueError("Initial cluster weights must be nonnegative, finite and nonzero")
    weights = weights / weights.max()
    weights /= weights.sum()
    if k == 1:
        return {
            "cluster_weights": np.ones(1),
            "log_likelihood": float(offset.sum()),
            "weight_optimality_gap": 0.0,
            "weight_active_score_gap": 0.0,
        }
    shifted = log_kernel - offset[:, None]
    kernel = np.exp(shifted)
    mass_key, mass_value = None, None

    def log_mass(w):
        # Keep the inexpensive matrix product on ordinary iterates, but recover
        # underflowed active support from the ORIGINAL finite log emissions.
        nonlocal mass_key, mass_value
        key = w.tobytes()
        if key != mass_key:
            mass = kernel @ w
            with np.errstate(divide="ignore"):
                value = np.log(mass)
                log_weights = np.log(w)
            underflow = mass < np.finfo(float).tiny
            if underflow.any():
                value[underflow] = logsumexp(shifted[underflow] + log_weights, axis=1)
                record("weight_log_domain_fallback_rows", int(underflow.sum()))
            mass_key, mass_value = key, value
            record("weight_matvec_calls")
        else:
            record("weight_matvec_cache_hits")
        return mass_value

    def objective(w):
        return -float(log_mass(w).mean())

    def gradient(w):
        # A boundary derivative can exceed float64 even when its log mixture
        # is finite. Saturation preserves the entering direction, never marks
        # that finite support impossible, and cannot pass the stationarity gate.
        ratios = shifted - log_mass(w)[:, None]
        return -np.exp(np.minimum(ratios, 345.)).mean(axis=0)

    # An impossible supplied boundary is an optimizer start, not an extra
    # constraint. Uniform weights are feasible because every row has support.
    if not np.isfinite(objective(weights)):
        weights = np.ones(k)/k
    # Huge boundary derivatives are unsuitable for SLSQP's quadratic step;
    # retain the better boundary as a candidate, but begin from the interior.
    boundary = weights.copy()
    if np.max(-gradient(weights)) > 1e100:
        weights = np.ones(k)/k

    tolerance = 1e-8
    record("weight_slsqp_calls")
    fit = minimize(
        objective,
        weights,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * k,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(k)},
        options={"ftol": 1e-13, "maxiter": 2000},
    )
    record("weight_slsqp_iterations", int(fit.nit))
    if np.all(np.isfinite(fit.x)):
        candidate = np.maximum(fit.x, 0.0)
        if candidate.sum() > 0:
            candidate /= candidate.sum()
            if objective(candidate) <= objective(weights):
                weights = candidate
    if objective(boundary) < objective(weights):
        weights = boundary
    # SLSQP's success flag is not an optimality check. EM can stall indefinitely
    # between nearly identical columns; instead transfer mass between the most
    # and least favorable coordinates. The 1-D log likelihood is concave, so
    # its derivative either brackets the maximum or chooses the simplex edge.
    # Previously zero weights may enter; no likelihood or support is discarded.
    for _ in range(10000):
        score = -gradient(weights)
        positive = np.flatnonzero(weights > 0)
        active_gap = float(score.max() - score[positive].min())
        # A tiny but positive unsupported weight barely changes the global
        # gap. Check active-coordinate stationarity as well, so exact line
        # searches reach true boundary zeros without an arbitrary weight floor.
        if max(0.0, float(score.max() - 1.0)) <= tolerance and active_gap <= tolerance:
            break
        grow = int(np.argmax(score))
        shrink = int(positive[np.argmin(score[positive])])
        if grow == shrink:
            break
        record("weight_transfer_steps")
        # Exact one-dimensional mass transfer, evaluated in the log domain
        # at endpoints too; no tiny-mass replacement changes its objective.
        amount = weights[shrink]
        high = np.maximum(shifted[:, grow], shifted[:, shrink])
        with np.errstate(divide="ignore", invalid="ignore"):
            delta = np.abs(shifted[:, grow]-shifted[:, shrink])
            numerator = np.log(amount) + high + np.log(-np.expm1(-delta))
            sign = np.sign(shifted[:, grow]-shifted[:, shrink])
        numerator = np.where(shifted[:, grow] == shifted[:, shrink], -np.inf, numerator)
        sign = np.where(np.isneginf(high), 0., sign)

        def derivative(fraction):
            trial = weights.copy()
            trial[grow] += fraction*amount
            trial[shrink] = amount*(1.-fraction)
            log_ratio = numerator-log_mass(trial)
            return float(np.mean(sign*np.exp(np.minimum(log_ratio, 345.))))

        fraction = 1.0 if derivative(1.0) >= 0 else brentq(derivative, 0.0, 1.0, xtol=1e-14)
        transfer = fraction * weights[shrink]
        weights[grow] += transfer
        weights[shrink] = 0.0 if fraction == 1.0 else weights[shrink] - transfer
        weights /= weights.sum()
    score = -gradient(weights)
    optimality_gap = n * max(0.0, float(score.max() - 1.0))
    active_gap = float(score.max() - score[weights > 0].min())
    if (
        not np.isfinite(objective(weights))
        or optimality_gap > n * tolerance
        or active_gap > tolerance
    ):
        record("weight_convergence_failures")
        raise RuntimeError("Cluster-weight likelihood optimization did not converge")
    likelihood = float(np.sum(offset + log_mass(weights)))
    return {
        "cluster_weights": weights,
        "log_likelihood": likelihood,
        "weight_optimality_gap": optimality_gap,
        "weight_active_score_gap": active_gap,
    }
