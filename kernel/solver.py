"""CliPP2 multi-region vector-jump continuation on one frozen mutation tree.

The positive-curvature, fixed-support majorization uses regional CCF vectors:
only uncut edges contribute quadratic springs, and the actual marginalized
objective accepts steps. Finite-penalty labels come from cuts, never from a
claim of exact fusion or constrained/global optimality.
"""

from dataclasses import dataclass
from collections import Counter

import numpy as np

from .topology import _cut_mask, _find_root, partition_from_cuts, project_cuts
from .native import forest_quadratic
from .anchor import project_anchor, anchor_penalty, anchored_quadratic
from .numerical import FailureSnapshots, QP_TOLERANCE, likelihood_snapshot_arrays, qp_residual


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
    for name, value in zip(('curvature', 'rhs', 'lower', 'upper'), (h, b, lo, hi)):
        affected = np.argwhere(~np.isfinite(value))
        if len(affected):
            raise ValueError(f'Nonfinite quadratic {name} at node,region={affected[0].tolist()}')
    if np.any(h <= 0) or np.any(lo > hi):
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
        a, c = _find_root(roots, int(i)), _find_root(roots, int(j))
        if a == c:
            raise ValueError("Quadratic graph must be a forest")
        roots[c] = a
    result, native = forest_quadratic(h, b, edges, penalty, lo, hi, return_diagnostics=True)
    audit = qp_residual(h, b, edges, penalty, lo, hi, result)
    audit.update(native)
    if native['budget_hits']:
        audit['failure_reasons'].append('active_set_budget_exhausted')
    if max(native['regional_preclip_bound_violation']) > 0:
        audit['failure_reasons'].append('preclip_bound_violation')
    audit['solved'] = bool(not audit['failure_reasons'] and audit['backward_error'] <= QP_TOLERANCE)
    return result, audit


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


def _support_warm_start(model, x, labels, lower, upper, witness, penalty):
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
        g[labels[witness]] += penalty*(center[labels[witness]]-1.)
        h[labels[witness]] += penalty
        direction = np.clip(center - g / np.maximum(h, 1e-8), lo, hi) - center
        residual = np.max(np.abs(center - np.clip(center - g / sizes[:, None], lo, hi)))
        slope = float(np.sum(g * direction))
        if residual <= 1e-8 or not slope < 0:
            break
        previous = float(np.sum(values)) + anchor_penalty(warm, witness, penalty)
        accepted = False
        for backtrack in range(40):
            step = 0.5 ** backtrack
            trial = (center + step * direction)[labels]
            candidate = model.node_terms(trial)
            objective = float(np.sum(candidate[0])) + anchor_penalty(trial, witness, penalty)
            if np.isfinite(objective) and objective <= previous + 1e-4 * step * slope:
                center += step * direction
                warm, (values, gradient, curvature) = trial, candidate
                accepted = True
                break
        if not accepted:
            break
    return warm, values, gradient, curvature


def generate_candidates(model, tree, pilot, capacities, *, policy=None, _batch=True):
    """Generate one finite-continuation connected partition per requested K.

    Numerical stationarity, edge feasibility and component-range feasibility
    are separate diagnostics. A candidate is still a finite-penalty proposal
    when tolerances are unmet; final scalar refits/selection determine its use.
    """
    policy = ContinuationPolicy() if policy is None else policy
    capacities = tuple(capacities)
    # Construct once before dispatch; worker proxies share its bounded owner.
    if _batch and getattr(model, 'numerical_diagnostics_dir', None) is not None:
        if getattr(model, '_numerical_failure_snapshots', None) is None:
            model._numerical_failure_snapshots = FailureSnapshots(
                model.numerical_diagnostics_dir,
                getattr(model, 'numerical_diagnostics_max_bytes', 16_777_216))
    snapshots = getattr(model, '_numerical_failure_snapshots', None)
    memory = getattr(model, 'memory', None)
    if _batch and memory is not None and getattr(model, '_torch', None) and len(capacities) > 1:
        per_state = max(1, model.n*model.r*(192*model.valid.shape[-1]+256))
        workers = min(8, memory.device_workspace_bytes//per_state,
                      memory.workspace_bytes//max(1, model.n*model.r*8*64), len(capacities))
        if workers > 1:
            from .execution import ordered_continuations
            model.telemetry['continuation_batch_workers'] = workers
            return ordered_continuations(model, tree, pilot, capacities, policy, workers, generate_candidates)
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
    model.check_clonal_feasibility(max(requested))
    support = getattr(model, 'regional_support', None)
    if support is None:  # Standalone numerical models retain the original box.
        lower = np.maximum(np.asarray(model.lower, dtype=float), 1e-8)
        upper = np.minimum(np.asarray(model.upper, dtype=float), 1 - 1e-8)
    else:
        lower, upper = support.continuation_bounds(model.lower, model.upper)
    if (lower.shape != pilot.shape or upper.shape != pilot.shape
            or not np.isfinite(lower).all() or not np.isfinite(upper).all()
            or np.any(lower > upper)):
        raise ValueError("No feasible box for raw tree continuation")
    candidates = []
    for k in requested:
        x = np.clip(pilot, lower, upper)
        values, gradient, curvature = model.node_terms(x)
        if not np.isfinite(values).all():
            raise ArithmeticError("Initial tree likelihood is nonfinite")
        rho = 1.0
        total_iterations = qp_unqualified = warm_starts = 0
        numerical_limit = stalled = False
        witness, _ = project_anchor(x, model.clonal_eligible)
        switches = 0
        accepted_steps = 0
        reason_counts = Counter()
        worst_qp = None
        last_line_search = None
        max_backtracking = 0
        maximum_accepted_increase = 0.0
        precision_limited = False
        for level in range(policy.levels):
            stalled = False
            repeated_stagnation = 0
            previous_support = None
            for _ in range(policy.iterations_per_level):
                total_iterations += 1
                cuts = project_cuts(tree, x, k)
                projected_witness, anchor_residual = project_anchor(x, model.clonal_eligible)
                switches += projected_witness != witness
                witness = projected_witness
                forest = tree.edges[~_cut_mask(tree, cuts)]
                jumps = x[forest[:, 0]] - x[forest[:, 1]]
                g = gradient + _spring_gradient(x, forest, rho)
                g[witness] += rho*(x[witness]-1.)
                stationarity = float(np.max(np.abs(x - np.clip(x - g, lower, upper))))
                if stationarity <= policy.stationarity_tolerance:
                    break
                h, rhs = anchored_quadratic(x, gradient, curvature, witness, rho)
                qp, audit = boxed_forest_quadratic(h, rhs, forest, rho, lower, upper)
                audit['proposed_step_max_CCF'] = float(np.max(np.abs(qp-x)))
                qp_unqualified += not audit["solved"]
                reason_counts.update(audit['failure_reasons'])
                audit_public = {key: value for key, value in audit.items() if key != 'active_state'}
                if worst_qp is None or audit['backward_error'] > worst_qp['backward_error']:
                    worst_qp = dict(audit_public, rho=rho, iteration=total_iterations)
                if not audit['solved'] and snapshots is not None:
                    arrays = dict(H=h, b=rhs, L=lower, U=upper, rho=np.asarray(rho),
                                  edges=forest, result=qp, active_state=audit['active_state'],
                                  likelihood_gradient=gradient, likelihood_curvature=curvature,
                                  **likelihood_snapshot_arrays(model, x))
                    snapshots.record(arrays, dict(kind='unqualified_qp', audit=audit_public,
                        requested_k=k, iteration=total_iterations, witness=int(witness),
                        tree_identity=tree.identity), audit['backward_error'])
                degree = np.bincount(forest.ravel(), minlength=tree.n)[:, None]
                old_objective = float(np.sum(values) + 0.5 * rho * np.sum(jumps * jumps)) + anchor_penalty(x, witness, rho)
                accepted = False
                accepted_decrease = 0.0
                proposals = [qp] if audit["solved"] else []
                proposals.append(np.clip(x - g / (h + rho * degree), lower, upper))
                for proposal in proposals:
                    direction = proposal - x
                    slope = float(np.sum(g * direction))
                    if not np.isfinite(slope) or not slope < 0:
                        reason_counts['no_descent_direction'] += 1
                        last_line_search = dict(reason='no_descent_direction', slope=slope,
                                                qp_qualified=audit['solved'], backtracks=0)
                        continue
                    for backtrack in range(40):
                        max_backtracking = max(max_backtracking, backtrack)
                        step = 0.5 ** backtrack
                        trial = np.clip(x + step * direction, lower, upper)
                        trial_terms = model.node_terms(trial)
                        delta = trial[forest[:, 0]] - trial[forest[:, 1]]
                        objective = float(np.sum(trial_terms[0]) + 0.5 * rho * np.sum(delta * delta)) + anchor_penalty(trial, witness, rho)
                        slack = 1e-12 * max(1.0, abs(old_objective))
                        if np.isfinite(objective) and objective <= old_objective + 1e-4 * step * slope + slack:
                            stalled = bool(np.max(np.abs(trial - x)) <= 8 * np.finfo(float).eps)
                            x, (values, gradient, curvature) = trial, trial_terms
                            accepted = True
                            accepted_steps += 1
                            accepted_decrease = old_objective - objective
                            maximum_accepted_increase = max(maximum_accepted_increase, -accepted_decrease)
                            break
                    if accepted:
                        break
                    reason_counts['line_search_rejected'] += 1
                    last_line_search = dict(reason='line_search_rejected', slope=slope,
                        qp_qualified=audit['solved'], backtracks=40,
                        old_objective=old_objective, last_objective=objective,
                        last_objective_change=objective-old_objective)
                if not accepted:
                    stalled = True
                    if snapshots is not None:
                        arrays = dict(H=h, b=rhs, L=lower, U=upper, rho=np.asarray(rho),
                                      edges=forest, result=qp, active_state=audit['active_state'],
                                      likelihood_gradient=gradient, likelihood_curvature=curvature,
                                      **likelihood_snapshot_arrays(model, x))
                        snapshots.record(arrays, dict(kind='no_accepted_descent_step', audit=audit_public,
                            line_search=last_line_search, requested_k=k, iteration=total_iterations,
                            witness=int(witness), tree_identity=tree.identity),
                            max(audit['backward_error'], QP_TOLERANCE))
                support_key = (tuple(map(int, cuts)), int(witness))
                stagnates = (not audit['solved'] and audit.get('precision_risk', False)
                             and accepted_decrease <= 1e-12 * max(1., abs(old_objective)))
                repeated_stagnation = (repeated_stagnation + 1 if stagnates and support_key == previous_support
                                       else int(stagnates))
                previous_support = support_key
                if repeated_stagnation >= 3 or (stalled and stagnates):
                    # No useful progress with fixed support and a demonstrated
                    # coordinate-spacing risk. Stop this branch honestly; this
                    # is NOT a replay-proved MPNST root-cause classification.
                    precision_limited = numerical_limit = True
                    reason_counts['precision_limited_stagnation'] += 1
                    break
                if stalled:
                    break
            cuts = project_cuts(tree, x, k)
            projected_witness, anchor_residual = project_anchor(x, model.clonal_eligible)
            switches += projected_witness != witness
            witness = projected_witness
            forest = tree.edges[~_cut_mask(tree, cuts)]
            g = gradient + _spring_gradient(x, forest, rho)
            g[witness] += rho*(x[witness]-1.)
            stationarity = float(np.max(np.abs(x - np.clip(x - g, lower, upper))))
            violation = float(np.linalg.norm(x[forest[:, 0]] - x[forest[:, 1]]))
            labels, block_range, longest, mean_h = _component_scale(tree, cuts, x, curvature)
            feasible = max(violation, block_range, anchor_residual) <= policy.constraint_tolerance
            if stationarity <= policy.stationarity_tolerance and feasible:
                break
            if precision_limited:
                break
            roundoff = 8 * rho * np.finfo(float).eps
            if rho >= policy.maximum_penalty or (stalled and feasible and roundoff >= policy.stationarity_tolerance):
                numerical_limit = True
                break
            if level + 1 < policy.levels:
                gap = 4 * np.sin(np.pi / (2 * longest)) ** 2
                factor = max(2.0, min(10.0, max(mean_h / gap / rho, max(violation, block_range, anchor_residual) / policy.constraint_tolerance)))
                next_rho = min(policy.maximum_penalty, rho * factor)
                warm = _support_warm_start(model, x, labels, lower, upper, witness, next_rho)
                if warm is not None:
                    old_objective = float(np.sum(values) + 0.5 * next_rho * violation ** 2) + anchor_penalty(x, witness, next_rho)
                    warm_objective = float(np.sum(warm[1])) + anchor_penalty(warm[0], witness, next_rho)
                    if np.isfinite(warm_objective) and warm_objective < old_objective:
                        x, values, gradient, curvature = [array.copy() for array in warm]
                        warm_starts += 1
                rho = next_rho
        # Recompute all diagnostics from the exact returned raw state.
        if not np.isfinite(x).all() or np.any(x < lower) or np.any(x > upper):
            raise ArithmeticError("Raw continuation violated its coordinate bounds")
        cuts = project_cuts(tree, x, k)
        projected_witness, anchor_residual = project_anchor(x, model.clonal_eligible)
        switches += projected_witness != witness
        witness = projected_witness
        forest = tree.edges[~_cut_mask(tree, cuts)]
        values, gradient, curvature = model.node_terms(x)
        g = gradient + _spring_gradient(x, forest, rho)
        g[witness] += rho*(x[witness]-1.)
        stationarity = float(np.max(np.abs(x-np.clip(x-g, lower, upper))))
        violation = float(np.linalg.norm(x[forest[:, 0]]-x[forest[:, 1]]))
        labels, block_range, longest, _ = _component_scale(tree, cuts, x, curvature)
        objective = float(np.sum(values)+.5*rho*violation**2) + anchor_penalty(x, witness, rho)
        tolerance_met = stationarity <= policy.stationarity_tolerance and max(violation, block_range, anchor_residual) <= policy.constraint_tolerance
        status = ("penalty_stationary_constraint_tolerance" if tolerance_met else
                  "precision_limited_candidate" if precision_limited else
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
                "objective": objective, "raw_penalty_objective": objective,
                "tree_constraint_residual_ccf": violation,
                "anchor_residual_ccf": anchor_residual,
                "anchor_witness_index": witness,
                "anchor_witness_mutation_id": model.mutation_ids[witness],
                "anchor_projection_switches": int(switches),
                "raw_stationarity_ccf": stationarity,
                "raw_constraint_tolerance_met": bool(max(violation, block_range, anchor_residual) <= policy.constraint_tolerance),
                "tolerance_met": bool(tolerance_met), "qp_unqualified": int(qp_unqualified),
                "feasible_warm_starts": warm_starts, "accepted_steps": accepted_steps,
                "penalty_gradient_roundoff_scale": 8 * rho * np.finfo(float).eps,
                "failure_reason_counts": dict(reason_counts),
                "worst_qp": worst_qp, "last_failed_line_search": last_line_search,
                "maximum_backtracking_index": max_backtracking,
                "maximum_accepted_objective_increase": maximum_accepted_increase,
                "precision_limit_basis": ('stagnant fixed-support updates with coordinate-spacing risk; '
                                           'not a replay-confirmed attribution' if precision_limited else None),
                "failure_snapshots": snapshots.summary() if snapshots is not None else None,
            },
        })
    return candidates
