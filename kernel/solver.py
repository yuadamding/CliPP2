"""CliPP2 multi-region vector-jump continuation on one frozen mutation tree.

The positive-curvature, fixed-support majorization uses regional CCF vectors:
only uncut edges contribute quadratic springs, and the actual marginalized
objective accepts steps. Finite-penalty labels come from cuts, never from a
claim of exact fusion or constrained/global optimality.
"""

from dataclasses import dataclass

import numpy as np

from .topology import _cut_mask, partition_from_cuts, project_cuts
from .native import forest_quadratic


@dataclass(frozen=True)
class ContinuationPolicy:
    levels: int = 20
    iterations_per_level: int = 300
    stationarity_tolerance: float = 1e-6
    constraint_tolerance: float = 1e-6
    maximum_penalty: float = 1e12

    def __post_init__(self):
        if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1
               for value in (self.levels, self.iterations_per_level)):
            raise ValueError("Continuation budgets must be positive integers")
        if not all(np.isfinite(v) and v > 0 for v in (
            self.stationarity_tolerance, self.constraint_tolerance, self.maximum_penalty
        )):
            raise ValueError("Continuation tolerances and penalty bound must be finite and positive")


def _spring_gradient(x, edges, penalty):
    result = np.zeros_like(x)
    jumps = x[edges[:, 0]] - x[edges[:, 1]]
    np.add.at(result, edges[:, 0], penalty * jumps)
    np.add.at(result, edges[:, 1], -penalty * jumps)
    return result


def boxed_forest_quadratic(curvature, rhs, edges, penalty, lower, upper):
    """Solve positive diagonal plus forest Laplacian box QPs by active sets.

    Return (X, diagnostics). ``solved`` requires an independently reconstructed
    componentwise KKT backward error, not just active-set loop termination.
    Runtime is linear per free-system sweep; the complete active-set solve may
    require multiple sweeps. The native host kernel performs the active-set
    solve; this wrapper independently checks its original-input residual.
    """
    h, b, lo, hi = [np.asarray(value, dtype=float) for value in (curvature, rhs, lower, upper)]
    if h.ndim != 2 or not all(h.shape) or any(value.shape != h.shape for value in (b, lo, hi)):
        raise ValueError("Quadratic arrays must have the same nonempty N by R shape")
    if not all(np.isfinite(value).all() for value in (h, b, lo, hi)) or np.any(h <= 0) or np.any(lo > hi):
        raise ValueError("Quadratic requires finite positive curvature and feasible finite bounds")
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError("Quadratic penalty must be finite and nonnegative")
    edges = np.asarray(edges)
    if edges.ndim != 2 or edges.shape[1] != 2 or not np.issubdtype(edges.dtype, np.integer):
        raise ValueError("Forest edges must be an integer (E, 2) array")
    n = h.shape[0]
    if np.any(edges < 0) or np.any(edges >= n):
        raise ValueError("Forest endpoint outside node range")
    # Reject accidental cyclic graphs: positive tree elimination is not a
    # general sparse-Laplacian solver.
    roots = np.arange(n)
    for i, j in edges:
        a, c = int(i), int(j)
        while roots[a] != a:
            roots[a] = roots[roots[a]]
            a = roots[a]
        while roots[c] != c:
            roots[c] = roots[roots[c]]
            c = roots[c]
        if a == c:
            raise ValueError("Quadratic graph must be a forest")
        roots[c] = a
    result, sweeps, budget_hits = forest_quadratic(h, b, edges, penalty, lo, hi)
    extended = result.astype(np.longdouble)
    gradient = h.astype(np.longdouble) * extended - b
    scale = 1 + np.abs(h.astype(np.longdouble) * extended) + np.abs(b)
    delta = np.longdouble(penalty) * (extended[edges[:, 0]] - extended[edges[:, 1]])
    np.add.at(gradient, edges[:, 0], delta)
    np.add.at(gradient, edges[:, 1], -delta)
    np.add.at(scale, edges[:, 0], np.abs(delta))
    np.add.at(scale, edges[:, 1], np.abs(delta))
    residual = np.where(result == lo, np.minimum(gradient, 0), gradient)
    residual = np.where(result == hi, np.maximum(residual, 0), residual)
    residual[lo == hi] = 0
    backward_error = float(np.max(np.abs(residual) / scale))
    bound_violation = float(max(np.max(lo - result), np.max(result - hi), 0.0))
    return result, {
        "solved": bool(budget_hits == 0 and backward_error <= 1e-7 and bound_violation == 0),
        "backward_error": backward_error,
        "bound_violation": bound_violation,
        "active_set_sweeps": sweeps,
        "budget_hits": int(budget_hits),
    }


def _component_scale(tree, cuts, x, curvature):
    labels = partition_from_cuts(tree, cuts)
    q = int(labels.max()) + 1
    minima = np.full((q, x.shape[1]), np.inf)
    maxima = np.full_like(minima, -np.inf)
    np.minimum.at(minima, labels, x)
    np.maximum.at(maxima, labels, x)
    sizes = np.bincount(labels)
    longest = int(sizes.max())
    # The path has the smallest algebraic connectivity among trees of this
    # size. Its gap is a conservative scale, not the exact forest spectrum.
    means = np.zeros_like(minima)
    np.add.at(means, labels, curvature)
    means /= sizes[:, None]
    h = float(means[sizes == longest].max())
    return labels, float((maxima - minima).max()), longest, h


def _support_warm_start(model, x, labels, lower, upper):
    q, r = int(labels.max()) + 1, x.shape[1]
    sizes = np.bincount(labels)
    lo, hi = np.full((q, r), -np.inf), np.full((q, r), np.inf)
    np.maximum.at(lo, labels, lower)
    np.minimum.at(hi, labels, upper)
    if np.any(lo > hi):
        return None
    center = np.zeros((q, r))
    np.add.at(center, labels, x)
    center = np.clip(center / sizes[:, None], lo, hi)
    warm = center[labels]
    values, gradient, curvature = model.node_terms(warm)
    for _ in range(50):
        g, h = np.zeros_like(center), np.zeros_like(center)
        np.add.at(g, labels, gradient)
        np.add.at(h, labels, curvature)
        direction = np.clip(center - g / np.maximum(h, 1e-8), lo, hi) - center
        residual = np.max(np.abs(center - np.clip(center - g / sizes[:, None], lo, hi)))
        slope = float(np.sum(g * direction))
        if residual <= 1e-8 or not slope < 0:
            break
        previous = float(np.sum(values))
        accepted = False
        for backtrack in range(40):
            step = 0.5 ** backtrack
            trial = (center + step * direction)[labels]
            candidate = model.node_terms(trial)
            objective = float(np.sum(candidate[0]))
            if np.isfinite(objective) and objective <= previous + 1e-4 * step * slope:
                center += step * direction
                warm, (values, gradient, curvature) = trial, candidate
                accepted = True
                break
        if not accepted:
            break
    return warm, values, gradient, curvature


def generate_candidates(model, tree, pilot, capacities, *, policy=None):
    """Generate one finite-continuation connected partition per requested K.

    Numerical stationarity, edge feasibility and component-range feasibility
    are separate diagnostics. A candidate is still a finite-penalty proposal
    when tolerances are unmet; final scalar refits/selection determine its use.
    """
    policy = ContinuationPolicy() if policy is None else policy
    pilot = np.asarray(pilot, dtype=float).copy()
    if tree.n != model.n or pilot.shape != (model.n, model.r):
        raise ValueError("Tree, model and pilot dimensions must agree")
    observed = np.asarray(getattr(model, "observed", np.ones_like(pilot, dtype=bool)), dtype=bool)
    if observed.shape != pilot.shape or not np.isfinite(pilot[observed]).all():
        raise ValueError("Observed pilot coordinates must be finite")
    missing_initializations = int(np.count_nonzero(~observed))
    for region in range(model.r):
        values = pilot[observed[:, region], region]
        # This initializes only latent raw coordinates AFTER the immutable
        # missing-overlap topology exists. It is not a fitted center and never
        # creates an observed edge or qualifies an unsupported score column.
        fill = float(values.mean()) if len(values) else .5
        pilot[~observed[:, region], region] = fill
    requested = list(capacities)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("Distinct nonempty capacities are required")
    for k in requested:
        project_cuts(tree, pilot, k)
    lower = np.maximum(np.asarray(model.lower, dtype=float), 1e-8)
    upper = np.minimum(np.asarray(model.upper, dtype=float), 1 - 1e-8)
    if lower.shape != pilot.shape or upper.shape != pilot.shape or np.any(lower > upper):
        raise ValueError("No nonempty interior box for raw tree continuation")
    candidates = []
    for k in requested:
        x = np.clip(pilot, lower, upper)
        values, gradient, curvature = model.node_terms(x)
        if not np.isfinite(values).all():
            raise ArithmeticError("Initial tree likelihood is nonfinite")
        rho = 1.0
        total_iterations = qp_unqualified = warm_starts = 0
        numerical_limit = stalled = False
        warm_labels = None
        warm = None
        accepted_steps = 0
        for level in range(policy.levels):
            stalled = False
            for _ in range(policy.iterations_per_level):
                total_iterations += 1
                cuts = project_cuts(tree, x, k)
                forest = tree.edges[~_cut_mask(tree, cuts)]
                jumps = x[forest[:, 0]] - x[forest[:, 1]]
                g = gradient + _spring_gradient(x, forest, rho)
                stationarity = float(np.max(np.abs(x - np.clip(x - g, lower, upper))))
                if stationarity <= policy.stationarity_tolerance:
                    break
                h = np.maximum(curvature, 1e-8)
                qp, audit = boxed_forest_quadratic(h, h * x - gradient, forest, rho, lower, upper)
                qp_unqualified += not audit["solved"]
                degree = np.bincount(forest.ravel(), minlength=tree.n)[:, None]
                old_objective = float(np.sum(values) + 0.5 * rho * np.sum(jumps * jumps))
                accepted = False
                proposals = [qp] if audit["solved"] else []
                proposals.append(np.clip(x - g / (h + rho * degree), lower, upper))
                for proposal in proposals:
                    direction = proposal - x
                    slope = float(np.sum(g * direction))
                    if not slope < 0:
                        continue
                    for backtrack in range(40):
                        step = 0.5 ** backtrack
                        trial = np.clip(x + step * direction, lower, upper)
                        trial_terms = model.node_terms(trial)
                        delta = trial[forest[:, 0]] - trial[forest[:, 1]]
                        objective = float(np.sum(trial_terms[0]) + 0.5 * rho * np.sum(delta * delta))
                        slack = 1e-12 * max(1.0, abs(old_objective))
                        if np.isfinite(objective) and objective <= old_objective + 1e-4 * step * slope + slack:
                            stalled = bool(np.max(np.abs(trial - x)) <= 8 * np.finfo(float).eps)
                            x, (values, gradient, curvature) = trial, trial_terms
                            accepted = True
                            accepted_steps += 1
                            break
                    if accepted:
                        break
                if not accepted:
                    stalled = True
                if stalled:
                    break
            cuts = project_cuts(tree, x, k)
            forest = tree.edges[~_cut_mask(tree, cuts)]
            g = gradient + _spring_gradient(x, forest, rho)
            stationarity = float(np.max(np.abs(x - np.clip(x - g, lower, upper))))
            violation = float(np.linalg.norm(x[forest[:, 0]] - x[forest[:, 1]]))
            labels, block_range, longest, mean_h = _component_scale(tree, cuts, x, curvature)
            feasible = max(violation, block_range) <= policy.constraint_tolerance
            if stationarity <= policy.stationarity_tolerance and feasible:
                break
            roundoff = 8 * rho * np.finfo(float).eps
            if rho >= policy.maximum_penalty or (stalled and feasible and roundoff >= policy.stationarity_tolerance):
                numerical_limit = True
                break
            if level + 1 < policy.levels:
                gap = 4 * np.sin(np.pi / (2 * longest)) ** 2
                factor = max(2.0, min(10.0, max(mean_h / gap / rho, max(violation, block_range) / policy.constraint_tolerance)))
                next_rho = min(policy.maximum_penalty, rho * factor)
                if warm_labels is None or not np.array_equal(labels, warm_labels):
                    warm = _support_warm_start(model, x, labels, lower, upper)
                    warm_labels = labels.copy()
                if warm is not None:
                    old_objective = float(np.sum(values) + 0.5 * next_rho * violation ** 2)
                    if np.isfinite(warm[1]).all() and float(np.sum(warm[1])) < old_objective:
                        x, values, gradient, curvature = [array.copy() for array in warm]
                        warm_starts += 1
                rho = next_rho
        tolerance_met = stationarity <= policy.stationarity_tolerance and max(violation, block_range) <= policy.constraint_tolerance
        status = ("penalty_stationary_constraint_tolerance" if tolerance_met else
                  "penalty_numerical_limit_candidate" if numerical_limit else
                  "line_search_stalled_candidate" if stalled else "finite_budget_candidate")
        candidates.append({
            "labels": labels, "cuts": cuts, "requested_k": int(k), "raw_phi": x.copy(),
            "diagnostics": {
                "status": status, "backend": "host_forest", "tree_identity": tree.identity,
                "missing_raw_coordinates_initialized": missing_initializations,
                "constrained_optimum_certified": False, "global_optimum_certified": False,
                "rho": rho, "iterations": total_iterations, "continuation_levels": level + 1,
                "stationarity_CCF": stationarity, "constraint_residual_CCF": violation,
                "max_block_range_CCF": block_range, "longest_block": longest,
                "negative_log_likelihood": float(np.sum(values)),
                "objective": float(np.sum(values) + 0.5 * rho * violation ** 2),
                "tolerance_met": bool(tolerance_met), "qp_unqualified": int(qp_unqualified),
                "feasible_warm_starts": warm_starts, "accepted_steps": accepted_steps,
                "penalty_gradient_roundoff_scale": 8 * rho * np.finfo(float).eps,
            },
        })
    return candidates
