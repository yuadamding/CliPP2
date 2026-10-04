"""CliPP2 pooled CP pilots for frozen chain or regional-tree construction.

Independent mutation maxima cannot resolve multiplicity aliases: a mutation
with multiplicity one in a major-CN-four region can be placed near one quarter
of its true CP. Pooling the *same* marginal likelihood across the tumor provides
an unsupervised initialization. This grid mixture is neither the fitted model
nor a new penalty, and its weights never enter the native objective or BIC.
"""
# Adapted from CliPP1.5; see ../NOTICE.

import numpy as np
from scipy.optimize import brentq, minimize

from .likelihood import MultiplicityModel


GRID_SIZE = 257
MAX_ADDITIONAL_GRID_POINTS = 2048
MAX_ITERATIONS = 500
MEAN_LOGLIK_TOLERANCE = 1e-8
GRID_OPTIMALITY_TOLERANCE = 1e-8
MAX_ACTIVE_REFIT_SIZE = 64
ACTIVE_REFIT_INTERVAL = 10
MAX_ACTIVE_REFIT_ITERATIONS = 200
MAX_MATRIX_BYTES = 64 * 1024 * 1024


def _initialization_grid(observations, purity):
    """Resolve narrow binomial modes without growing the ordinary-depth grid.

    A component needs an extra proposal when its binomial-mode CP standard
    error is smaller than one quarter of the base grid spacing. This numerical
    resolution criterion uses observed counts/CN only, never truth or K. A
    half-count floor handles variance estimates at binomial endpoints. Modes
    at CP=0 or purity are already included. If the proposal budget is exceeded,
    take evenly spaced ranks through sorted observations and multiplicities;
    report this deterministic compression rather than claim full resolution.
    """
    grid = np.linspace(0.0, purity, GRID_SIZE)
    threshold = purity / (4 * (GRID_SIZE - 1))
    if threshold == 0.0:
        return grid, {"narrow_component_modes": 0, "mode_grid_compressed": False, "added_grid_points": 0}
    r, n, major, total = observations.T
    vaf = r / n
    denominator = 2 * (1 - purity) + purity * total
    mode_numerator = vaf * denominator
    se_numerator = denominator * np.maximum(np.sqrt(vaf * (1 - vaf) / n), 0.5 / n)
    # CP mode and its standard error both divide their numerator by m.
    # Compute eligible m ranges analytically; do not materialize N*major modes.
    bound = np.maximum(mode_numerator / purity, se_numerator / threshold)
    first = np.minimum(major + 1, np.floor(bound) + 1).astype(np.int64)
    counts = np.maximum(0, major.astype(np.int64) - first + 1)
    counts[r == 0] = 0
    cumulative = np.cumsum(counts, dtype=np.int64)
    count = int(cumulative[-1])
    if count:
        ranks = np.linspace(0, count - 1, min(count, MAX_ADDITIONAL_GRID_POINTS), dtype=np.int64)
        row = np.searchsorted(cumulative, ranks, side="right")
        previous = np.r_[0, cumulative[:-1]]
        multiplicity = first[row] + ranks - previous[row]
        proposals = mode_numerator[row] / multiplicity
        grid = np.unique(np.r_[grid, proposals])
    return grid, {
        "narrow_component_modes": count,
        "mode_grid_compressed": count > MAX_ADDITIONAL_GRID_POINTS,
        "added_grid_points": len(grid) - GRID_SIZE,
    }


def pooled_cp_initialization(model):
    """Return posterior-mean CPs and finite-grid convex-optimization diagnostics.

    The base grid, resolution rule and optimization budget are data-independent.
    Narrow likelihood modes add bounded numerical proposals. Identical count/CN
    observations are collapsed in sorted order, preserving their frequencies;
    this also makes results exactly equivariant to input-row permutations.
    Large inputs stream likelihood blocks rather than allocate an unbounded
    N-by-grid matrix. Full 1..major CN support remains in MultiplicityModel.
    """
    observations = np.column_stack((model.r, model.n, model.major, model.total))
    unique, inverse, frequencies = np.unique(observations, axis=0, return_inverse=True, return_counts=True)
    unique_count = len(unique)
    grid, grid_diagnostics = _initialization_grid(unique, model.purity)
    grid_size = len(grid)
    frequencies = frequencies.astype(float)
    n = float(len(model))
    # Limit both the likelihood matrix and padded multiplicity temporaries.
    matrix_rows = max(1, MAX_MATRIX_BYTES // (8 * grid_size))
    support_rows = max(1, 1_000_000 // int(unique[:, 2].max()))
    block_rows = min(matrix_rows, support_rows)
    cache = [] if unique_count <= matrix_rows else None

    def calculate(first, stop, positions=None):
        block = MultiplicityModel(*unique[first:stop].T, model.purity)
        values = np.column_stack(
            [block.log_likelihood(cp) for cp in (grid if positions is None else grid[positions])]
        )
        offset = values.max(axis=1)
        if not np.isfinite(offset).all():
            raise RuntimeError("No finite grid likelihood for CP initialization")
        return np.exp(values - offset[:, None]), offset

    if cache is not None:
        for first in range(0, unique_count, block_rows):
            stop = min(unique_count, first + block_rows)
            cache.append((first, stop, *calculate(first, stop)))

    def blocks(positions=None):
        if cache is not None:
            for first, stop, likelihood, offset in cache:
                yield (first, stop, likelihood if positions is None else likelihood[:, positions], offset)
        else:
            for first in range(0, unique_count, block_rows):
                stop = min(unique_count, first + block_rows)
                yield first, stop, *calculate(first, stop, positions)

    evaluations = 0

    def expectation(weights, collect_pilot=False, positions=None):
        nonlocal evaluations
        evaluations += 1
        score = np.zeros(len(weights))
        log_likelihood = 0.0
        pilot = np.empty(unique_count) if collect_pilot else None
        for first, stop, likelihood, offset in blocks(positions):
            normalizer = likelihood @ weights
            if np.any(normalizer <= 0) or not np.isfinite(normalizer).all():
                return None, -np.inf, None
            frequency = frequencies[first:stop]
            with np.errstate(over="ignore", invalid="ignore"):
                score += (frequency / normalizer) @ likelihood
            if not np.isfinite(score).all():
                return None, -np.inf, None
            log_likelihood += float(frequency @ (np.log(normalizer) + offset))
            if collect_pilot:
                pilot[first:stop] = likelihood @ (weights * grid) / normalizer
        return score / n, log_likelihood, pilot

    def normalize(candidate):
        candidate = np.maximum(candidate, 0.0)
        return candidate / candidate.sum()

    def exchange(weights, score):
        """Exact concave line search; zero-weight grid points can re-enter."""
        grow = int(np.argmax(score))
        positive = np.flatnonzero(weights > 0)
        shrink = int(positive[np.argmin(score[positive])])
        if grow == shrink:
            return weights
        mass, direction = np.empty(unique_count), np.empty(unique_count)
        for first, stop, likelihood, _ in blocks():
            mass[first:stop] = likelihood @ weights
            direction[first:stop] = weights[shrink] * (likelihood[:, grow] - likelihood[:, shrink])

        def derivative(fraction):
            denominator = np.maximum(mass + fraction * direction, np.finfo(float).tiny)
            return float((frequencies / n * direction / denominator).sum())

        if derivative(0.0) <= 0.0:
            return weights
        fraction = 1.0 if derivative(1.0) >= 0.0 else brentq(derivative, 0.0, 1.0, xtol=1e-14)
        candidate = weights.copy()
        transfer = fraction * weights[shrink]
        candidate[grow] += transfer
        candidate[shrink] = 0.0 if fraction == 1.0 else weights[shrink] - transfer
        return normalize(candidate)

    def correct_active_support(weights):
        # Solve only a small *existing* support. This is an acceleration, not
        # pruning: global grid scores and exchange steps still admit every CP.
        active = np.flatnonzero(weights > 0)
        previous, value, gradient = None, None, None

        def objective(candidate):
            nonlocal previous, value, gradient
            if previous is None or not np.array_equal(candidate, previous):
                score, likelihood, _ = expectation(candidate, positions=active)
                previous = candidate.copy()
                value = -likelihood / n
                gradient = -score if score is not None else np.zeros(len(active))
            return value, gradient

        fit = minimize(
            objective,
            weights[active],
            jac=True,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * len(active),
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(len(w))},
            options={"ftol": 1e-13, "maxiter": MAX_ACTIVE_REFIT_ITERATIONS},
        )
        candidate = weights.copy()
        if np.isfinite(fit.x).all() and np.maximum(fit.x, 0).sum() > 0:
            candidate[active] = normalize(fit.x)
        return candidate

    weights = np.full(grid_size, 1.0 / grid_size)
    score, log_likelihood, _ = expectation(weights)
    if score is None:
        raise RuntimeError("Nonfinite pooled CP initialization likelihood")
    initial_log_likelihood = log_likelihood
    improvement = 0.0
    accelerated_steps = exchange_steps = active_refits = iteration = 0
    step_limit = 1.0
    for iteration in range(1, MAX_ITERATIONS + 1):
        if max(0.0, float(score.max() - 1.0)) <= GRID_OPTIMALITY_TOLERANCE:
            iteration -= 1
            break
        previous_log_likelihood = log_likelihood
        # Safeguarded SQUAREM extrapolates two EM updates. Negative entries
        # are clipped and renormalized; accepted iterates improve likelihood.
        # Exact zeros are allowed; the global-score exchange below restores
        # components that EM alone cannot reintroduce.
        first = normalize(weights * score)
        first_score, _, _ = expectation(first)
        second = normalize(first * first_score)
        second_score, second_likelihood, _ = expectation(second)
        residual = first - weights
        curvature = second - 2.0 * first + weights
        step = max(
            1.0,
            min(
                float(np.sqrt((residual @ residual) / max(curvature @ curvature, np.finfo(float).tiny))),
                step_limit,
            ),
        )
        candidate = normalize(weights + 2.0 * step * residual + step * step * curvature)
        candidate_score, candidate_likelihood, _ = expectation(candidate)
        while candidate_likelihood < second_likelihood and step > 1.01:
            step = (step + 1.0) / 2.0
            candidate = normalize(weights + 2.0 * step * residual + step * step * curvature)
            candidate_score, candidate_likelihood, _ = expectation(candidate)
        if candidate_likelihood >= second_likelihood:
            weights, score, log_likelihood = candidate, candidate_score, candidate_likelihood
            accelerated_steps += int(step > 1.0)
        else:
            weights, score, log_likelihood = second, second_score, second_likelihood
        if step >= 0.99 * step_limit:
            step_limit *= 4.0

        candidate = exchange(weights, score)
        candidate_score, candidate_likelihood, _ = expectation(candidate)
        if candidate_likelihood >= log_likelihood:
            exchange_steps += int(not np.array_equal(candidate, weights))
            weights, score, log_likelihood = candidate, candidate_score, candidate_likelihood
        if iteration % ACTIVE_REFIT_INTERVAL == 0 and np.count_nonzero(weights) <= MAX_ACTIVE_REFIT_SIZE:
            candidate = correct_active_support(weights)
            candidate_score, candidate_likelihood, _ = expectation(candidate)
            active_refits += 1
            if candidate_likelihood >= log_likelihood:
                weights, score, log_likelihood = candidate, candidate_score, candidate_likelihood
        improvement = (log_likelihood - previous_log_likelihood) / n
        if improvement < -1e-10 * max(1.0, abs(previous_log_likelihood) / n):
            raise RuntimeError("Pooled CP initialization likelihood decreased")
    score, log_likelihood, pilot = expectation(weights, collect_pilot=True)
    # Concavity bounds the missing mean log likelihood by max(score)-1:
    # the weighted mean score equals one for every feasible mixture.
    gap = float(max(0.0, score.max() - 1.0))
    converged = gap <= GRID_OPTIMALITY_TOLERANCE
    pilot = np.clip(pilot[inverse], 0.0, model.purity)
    diagnostics = {
        "method": "pooled_uniform_multiplicity_grid_accelerated_simplex_v2",
        "role": "initialization_only_frozen_before_all_K",
        "grid_size": grid_size,
        "base_grid_size": GRID_SIZE,
        "max_additional_grid_points": MAX_ADDITIONAL_GRID_POINTS,
        **grid_diagnostics,
        "cp_bounds": [0.0, float(model.purity)],
        "num_mutations": len(model),
        "num_unique_observations": unique_count,
        "iterations": iteration,
        "max_iterations": MAX_ITERATIONS,
        "mean_loglik_tolerance": MEAN_LOGLIK_TOLERANCE,
        "mean_loglik_improvement": float(improvement),
        "converged_mean_loglik": bool(improvement <= MEAN_LOGLIK_TOLERANCE),
        "converged_grid_weights": converged,
        "termination_reason": "grid_duality_gap" if converged else "iteration_budget",
        "grid_optimality_tolerance": GRID_OPTIMALITY_TOLERANCE,
        "grid_score_excess": gap,
        "grid_weight_optimality_gap": n * gap,
        "grid_weight_optimum_certified": converged,
        "likelihood_evaluations": evaluations,
        "accelerated_steps": accelerated_steps,
        "exchange_steps": exchange_steps,
        "active_support_refits": active_refits,
        "max_active_refit_size": MAX_ACTIVE_REFIT_SIZE,
        "active_refit_interval": ACTIVE_REFIT_INTERVAL,
        "max_active_refit_iterations": MAX_ACTIVE_REFIT_ITERATIONS,
        "initial_log_likelihood": float(initial_log_likelihood),
        "log_likelihood": float(log_likelihood),
        "likelihood_matrix_cached": cache is not None,
        "max_matrix_bytes": MAX_MATRIX_BYTES,
        "constrained_optimum_certified": False,
    }
    return pilot, diagnostics
