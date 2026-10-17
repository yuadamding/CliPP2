"""Read-likelihood boundary proposals on one frozen chain.

Only boundaries move. Component visits follow the supplied partition, including
nonmonotone CP sequences. No mixture weights or unconstrained MAP assignments
enter the dynamic-program proposal objective. Under assigned/classification
selection, authoritative refitted scores govern path advancement; the read-DP
is explicitly not an exact classification optimizer. Legacy paths retain their
conditional-likelihood admission rule.
"""

import math

import numpy as np


def fixed_center_chain_partition(log_likelihood):
    """Exactly optimize nonempty, ordered blocks for fixed component centers.

    ``log_likelihood[i, j]`` is observation i's conditional marginal log
    likelihood at the jth center *in chain visit order*. Every column must be
    used by one nonempty contiguous interval. The dynamic program takes O(Nq)
    time and memory, permits -inf entries, and deterministically chooses the
    earliest preceding cut when scores tie. Centers need not increase with j.
    """
    likelihood = np.asarray(log_likelihood, dtype=float)
    if (
        likelihood.ndim != 2
        or min(likelihood.shape) == 0
        or likelihood.shape[1] > likelihood.shape[0]
        or np.any(np.isnan(likelihood))
        or np.any(np.isposinf(likelihood))
    ):
        raise ValueError("Expected an N-by-q finite-or-negative-infinity array, 1 <= q <= N")
    n, q = likelihood.shape
    previous = np.full(n + 1, -np.inf)
    previous[0] = 0.0
    parents = np.full((q, n + 1), -1, dtype=np.int64)
    for component in range(q):
        column = likelihood[:, component]
        current = np.full(n + 1, -np.inf)
        # A block cannot cross a -inf observation. Independent finite runs
        # avoid subtracting infinite prefix sums or producing NaNs.
        finite = np.isfinite(column)
        starts = np.flatnonzero(finite & np.r_[True, ~finite[:-1]])
        stops = np.flatnonzero(finite & np.r_[~finite[1:], True]) + 1
        for start, stop in zip(starts, stops):
            prefix = np.r_[0.0, np.cumsum(column[start:stop])]
            if not np.isfinite(prefix).all():
                raise ValueError("Conditional log-likelihood accumulation overflowed")
            proposals = previous[start:stop] - prefix[:-1]
            best = np.maximum.accumulate(proposals)
            improves = proposals > np.r_[-np.inf, best[:-1]]
            cuts = np.maximum.accumulate(np.where(improves, np.arange(start, stop), -1))
            current[start + 1 : stop + 1] = prefix[1:] + best
            parents[component, start + 1 : stop + 1] = cuts
        previous = current
    if not np.isfinite(previous[n]):
        raise RuntimeError("No finite partition uses every ordered component in a nonempty block")
    labels = np.empty(n, dtype=np.int64)
    end = n
    for component in range(q - 1, -1, -1):
        start = int(parents[component, end])
        if not 0 <= start < end:
            raise RuntimeError("Invalid chain dynamic-program backtrack")
        labels[start:end] = component
        end = start
    if end != 0:
        raise RuntimeError("Chain dynamic-program backtrack did not cover every observation")
    # Sum the actual selected entries, independently of prefix-sum subtraction.
    score = math.fsum(likelihood[np.arange(n), labels])
    return {
        "labels": labels,
        "cuts": np.flatnonzero(np.diff(labels)) + 1,
        "conditional_log_likelihood": score,
        "num_clusters": q,
        "status": "exact_fixed_center_chain_partition",
    }


def _partition_state(model, result, chain_order, column_evaluator=None):
    labels = np.asarray(result["labels"])
    centers = np.asarray(result["centers"], dtype=float)
    n = len(model)
    if (
        labels.shape != (n,)
        or not np.isfinite(labels).all()
        or np.any(labels != np.rint(labels))
        or centers.ndim != 1
        or not len(centers)
        or not np.isfinite(centers).all()
    ):
        raise ValueError("Invalid refitted chain labels or centers")
    labels = labels.astype(np.int64)
    if not np.array_equal(np.unique(labels), np.arange(len(centers))):
        raise ValueError("Every refitted center must have an occupied label")
    clonal = result["clonal_cluster"]
    if not 0 <= clonal < len(centers) or centers[clonal] != model.purity:
        raise ValueError("Scalar refinement requires an occupied center fixed at purity")
    ordered = labels[chain_order]
    cuts = np.flatnonzero(np.diff(ordered)) + 1
    visits = ordered[np.r_[0, cuts]]
    if len(visits) != len(centers) or len(np.unique(visits)) != len(visits):
        raise ValueError("Each center must visit exactly one contiguous chain block")
    kernel = (column_evaluator(centers[visits]) if column_evaluator is not None else
              np.column_stack([model.log_likelihood(cp) for cp in centers[visits]]))[chain_order]
    states = np.searchsorted(cuts, np.arange(n), side="right")
    score = math.fsum(kernel[np.arange(n), states])
    if not np.isfinite(score):
        raise ValueError("Supplied chain partition has nonfinite conditional likelihood")
    return labels, cuts, kernel, score


def polish_chain_partition(
    model, result, chain_order, refit, max_iterations=100, mean_loglik_tolerance=1e-10,
    *, column_evaluator=None,
):
    """Propose read-optimal conditional boundaries and refit occupied blocks.

    ``refit(labels, centers)`` receives labels in original mutation-row order
    and initial centers in chain-visit order. It returns a result dictionary
    containing ``labels``, ``centers`` and ``clonal_cluster``. It may merge
    adjacent blocks, but cannot add or move boundaries itself. The returned
    Legacy mode returns the best conditional-likelihood path candidate. New
    objectives accept only improvements in the supplied authoritative score;
    all evaluated variants are retained by the caller. A finite iteration budget
    or a read-proposal fixed point is not a classification/global certificate.
    """
    if (
        not isinstance(max_iterations, (int, np.integer))
        or max_iterations < 1
        or not np.isfinite(mean_loglik_tolerance)
        or mean_loglik_tolerance < 0
    ):
        raise ValueError("Require a positive iteration budget and nonnegative finite tolerance")
    order = np.asarray(chain_order)
    n = len(model)
    if (
        order.shape != (n,)
        or not np.isfinite(order).all()
        or np.any(order != np.rint(order))
        or not np.array_equal(np.sort(order), np.arange(n))
    ):
        raise ValueError("Frozen chain order must be a permutation of original rows")
    order = order.astype(np.int64)
    current = result
    objective = current.get("selection_objective", getattr(model, "selection_objective", "legacy_mixture"))
    use_selection_score = objective != "legacy_mixture"
    selection_score = float(current["bic"]) if use_selection_score else None
    if use_selection_score and not np.isfinite(selection_score):
        raise ValueError("Assigned-objective refinement requires a finite authoritative score")
    initial_selection_score = selection_score
    _, cuts, kernel, score = _partition_state(model, current, order, column_evaluator)
    initial_score, initial_q = score, kernel.shape[1]
    best, best_score = current, score
    history = []
    status = "iteration_budget"
    for iteration in range(1, max_iterations + 1):
        proposal = fixed_center_chain_partition(kernel)
        fixed_score = proposal["conditional_log_likelihood"]
        roundoff = 64 * np.finfo(float).eps * max(1.0, abs(score), abs(fixed_score))
        if fixed_score < score - roundoff:
            raise RuntimeError("Exact chain boundary update decreased conditional likelihood")
        entry = {
            "iteration": iteration,
            "num_clusters": kernel.shape[1],
            "previous_conditional_log_likelihood": score,
            "fixed_center_conditional_log_likelihood": fixed_score,
            "boundaries_changed": not np.array_equal(proposal["cuts"], cuts),
            "selection_objective": objective,
            "previous_selection_score": selection_score,
        }
        if not entry["boundaries_changed"]:
            entry["status"] = "fixed_center_boundaries_stable"
            history.append(entry)
            status = ("read_proposal_boundaries_stable" if use_selection_score else entry["status"])
            entry["status"] = status
            break
        labels = np.empty(n, dtype=np.int64)
        labels[order] = proposal["labels"]
        try:
            visits = current["labels"][order][np.r_[0, cuts]]
            candidate = refit(labels, current["centers"][visits])
            _, new_cuts, new_kernel, new_score = _partition_state(model, candidate, order, column_evaluator)
        except RuntimeError as error:
            entry.update(status="refit_failed", error=str(error))
            history.append(entry)
            status = entry["status"]
            break
        if not set(new_cuts) <= set(proposal["cuts"]):
            raise ValueError("Refit may only preserve boundaries or merge adjacent blocks")
        roundoff = 64 * np.finfo(float).eps * max(1.0, abs(fixed_score), abs(new_score))
        entry.update(refit_conditional_log_likelihood=new_score, refit_num_clusters=new_kernel.shape[1])
        if use_selection_score:
            if candidate.get("selection_objective") != objective:
                raise ValueError("Boundary refit changed the declared selection objective")
            new_selection_score = float(candidate["bic"])
            if not np.isfinite(new_selection_score):
                raise ValueError("Boundary refit returned a nonfinite authoritative score")
            entry["refit_selection_score"] = new_selection_score
            improvement = (selection_score-new_selection_score)/2.
            tolerance = 64*np.finfo(float).eps*max(1., abs(selection_score), abs(new_selection_score))
            if selection_score-new_selection_score <= tolerance:
                entry["status"] = "selection_score_not_improved"
                history.append(entry)
                status = entry["status"]
                break
        elif new_score < fixed_score - roundoff or new_score < score:
            # A numerical scalar refit need not recover its supplied-center
            # witness. Keep the accepted incumbent instead of hiding a loss.
            entry["status"] = "refit_conditional_decrease"
            history.append(entry)
            status = entry["status"]
            break
        else:
            improvement = new_score-score
        entry["status"] = "accepted"
        entry[("mean_selection_score_improvement_per_2_mutations" if use_selection_score
               else "mean_loglik_improvement")] = improvement/n
        history.append(entry)
        current, cuts, kernel, score = candidate, new_cuts, new_kernel, new_score
        if use_selection_score:
            selection_score = new_selection_score
            best, best_score = current, score
        elif score > best_score or (score == best_score and len(current["centers"]) < len(best["centers"])):
            best, best_score = current, score
        if improvement / n <= mean_loglik_tolerance:
            status = "selection_improvement_tolerance" if use_selection_score else "conditional_improvement_tolerance"
            break
    return {
        "result": best,
        "history": history,
        "diagnostics": {
            "method": ("ordered_read_boundary_proposals_selected_score_refinement_v1" if use_selection_score
                       else "ordered_conditional_chain_boundary_refinement_v1"),
            "proposal_objective": "assigned_read_likelihood",
            "path_acceptance_objective": objective if use_selection_score else "conditional_read_likelihood",
            "classification_fixed_center_optimum_certified": False,
            "initial_selection_score": initial_selection_score,
            "selection_score": selection_score,
            "status": status,
            "iterations": len(history),
            "max_iterations": int(max_iterations),
            "mean_loglik_tolerance": float(mean_loglik_tolerance),
            "initial_num_clusters": initial_q,
            "final_num_clusters": len(best["centers"]),
            "initial_conditional_log_likelihood": initial_score,
            "conditional_log_likelihood": best_score,
            "conditional_log_likelihood_improvement": best_score - initial_score,
            "fixed_chain": True,
            "unconstrained_reassignment": False,
            "constrained_optimum_certified": False,
        },
    }
