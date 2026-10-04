"""CliPP2 observed-mixture weights at fixed conditionally fitted centers."""
# Adapted from CliPP1.5; see ../NOTICE.

import numpy as np
from scipy.optimize import brentq, minimize


def fit_cluster_weights(model, centers, initial_weights=None, *, log_kernel=None, telemetry=None):
    """Maximize the observed likelihood over the simplex at fixed CP centers.

    This is a concave weight problem. The reported Frank-Wolfe gap bounds the
    remaining log-likelihood improvement at these centers, even with duplicate
    centers or boundary weights. Zero weights are returned for explicit repair
    by the caller; this optimizer does not change partition memberships.
    Center fitting itself remains conditional on the frozen chain partition;
    this function does not claim a joint mixture-center maximum.
    """

    def record(name, amount=1):
        if telemetry is not None:
            telemetry[name] = telemetry.get(name, 0) + amount

    record("weight_problem_calls")
    centers = np.atleast_1d(np.asarray(centers, dtype=float))
    if (
        centers.ndim != 1
        or not centers.size
        or np.any(~np.isfinite(centers))
        or np.any(centers < 0)
        or np.any(centers > model.purity)
    ):
        raise ValueError("Expected finite cluster centers in [0, purity]")
    if log_kernel is None:
        log_kernel = np.column_stack([model.log_likelihood(cp) for cp in centers])
    else:
        log_kernel = np.asarray(log_kernel, dtype=float)
        if (
            log_kernel.shape != (len(model), len(centers))
            or np.any(np.isnan(log_kernel))
            or np.any(np.isposinf(log_kernel))
        ):
            raise ValueError("Invalid cached center likelihoods")
    offset = np.max(log_kernel, axis=1)
    if np.any(~np.isfinite(offset)):
        raise ValueError("Every mutation needs finite likelihood in some cluster")
    k = len(centers)
    if k == 1:
        return {
            "cluster_weights": np.ones(1),
            "log_likelihood": float(offset.sum()),
            "weight_optimality_gap": 0.0,
            "weight_active_score_gap": 0.0,
        }
    kernel = np.exp(log_kernel - offset[:, None])
    weights = np.ones(k) / k if initial_weights is None else np.asarray(initial_weights, dtype=float)
    if weights.shape != (k,) or np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Initial cluster weights must be positive and finite")
    weights = weights / weights.sum()

    mass_key, mass_value = None, None

    def mixture_mass(w):
        # SLSQP often asks for objective and gradient at the identical iterate.
        # Keep one exact-key matvec, never a rounded or approximate iterate.
        # Bytes also detect in-place changes during pairwise mass transfers.
        nonlocal mass_key, mass_value
        key = (w.dtype.str, w.shape, w.tobytes())
        if key != mass_key:
            mass_value = kernel @ w
            mass_key = key
            record("weight_matvec_calls")
        else:
            record("weight_matvec_cache_hits")
        return mass_value

    def objective(w):
        mass = mixture_mass(w)
        return -float(np.log(mass).mean()) if np.all(mass > 0) else np.inf

    def gradient(w):
        mass = np.maximum(mixture_mass(w), np.finfo(float).tiny)
        return -(kernel / mass[:, None] / len(model)).sum(axis=0)

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
        mass = mixture_mass(weights)
        direction = weights[shrink] * (kernel[:, grow] - kernel[:, shrink])

        def derivative(fraction):
            normalizer = np.maximum(mass + fraction * direction, np.finfo(float).tiny)
            return float(np.sum(direction / normalizer / len(model)))

        fraction = 1.0 if derivative(1.0) >= 0 else brentq(derivative, 0.0, 1.0, xtol=1e-14)
        transfer = fraction * weights[shrink]
        weights[grow] += transfer
        weights[shrink] = 0.0 if fraction == 1.0 else weights[shrink] - transfer
        weights /= weights.sum()
    score = -gradient(weights)
    optimality_gap = len(model) * max(0.0, float(score.max() - 1.0))
    active_gap = float(score.max() - score[weights > 0].min())
    if (
        not np.isfinite(objective(weights))
        or optimality_gap > len(model) * tolerance
        or active_gap > tolerance
    ):
        record("weight_convergence_failures")
        raise RuntimeError("Cluster-weight likelihood optimization did not converge")
    likelihood = float(np.sum(offset + np.log(mixture_mass(weights))))
    return {
        "cluster_weights": weights,
        "log_likelihood": likelihood,
        "weight_optimality_gap": optimality_gap,
        "weight_active_score_gap": active_gap,
    }
