"""Shared anchored emissions and qualification-only batched weight fitting.

The CPU SLSQP-plus-transfer result remains authoritative. Device solutions are
compared against that reference, never substituted merely because their concave
objective gaps pass: equally optimal solutions can have different exact zeros.
Fixed-center bounds are diagnostics, not bounds on descendants or search pruning.
"""
import numpy as np

from ..io.data import readonly_array


class AnchoredFamily:
    """One immutable N-by-(q+1) table, with one small map per clonal variant."""

    def __init__(self, free_columns, clonal_column, clonal_indices):
        free = np.asarray(free_columns, dtype=float)
        clonal = np.asarray(clonal_column, dtype=float)
        if (free.ndim != 2 or not all(free.shape) or clonal.shape != (free.shape[0],)
                or np.any(np.isnan(free)) or np.any(np.isposinf(free))
                or np.any(np.isnan(clonal)) or np.any(np.isposinf(clonal))):
            raise ValueError("Anchored emissions must be finite or negative infinity")
        self.n, self.q = free.shape
        indices = tuple(clonal_indices)
        if any(isinstance(a, (bool, np.bool_)) or not isinstance(a, (int, np.integer)) for a in indices):
            raise ValueError("Clonal designations must be integer occupied indices")
        self.anchors = tuple(int(a) for a in indices)
        if (not self.anchors or len(set(self.anchors)) != len(self.anchors)
                or any(not 0 <= a < self.q for a in self.anchors)):
            raise ValueError("Clonal designations must be distinct occupied indices")
        self.columns = readonly_array(np.column_stack((clonal, free)))
        maps = np.tile(np.arange(1, self.q+1), (len(self.anchors), 1))
        maps[np.arange(len(self.anchors)), self.anchors] = 0
        self.maps = readonly_array(maps)

    def matrix(self, anchor):
        """Materialize only one reference problem, preserving C row layout."""
        return np.ascontiguousarray(self.columns[:, self.maps[self.anchors.index(anchor)]])

    def fixed_center_bounds(self, penalty):
        """Rowwise top-two bound, with ties and impossible rows preserved."""
        best, second = np.full(self.n, -np.inf), np.full(self.n, -np.inf)
        winner = np.full(self.n, -1, dtype=int)
        for k in range(self.q):
            values = self.columns[:, k+1]
            larger = values > best
            second = np.where(larger, best, np.maximum(second, values))
            best = np.maximum(best, values)
            winner = np.where(larger, k, winner)
        bounds = {}
        for anchor in self.anchors:
            other = np.where(winner == anchor, second, best)
            upper = float(np.maximum(self.columns[:, 0], other).sum())
            bounds[anchor] = {"log_likelihood_upper_bound": upper,
                              "score_lower_bound": -2*upper+float(penalty),
                              "scope": "fixed_centers_only", "used_to_prune": False}
        return bounds


def _tensor_weight_solve(t, columns, counts, maps, *, device):
    """Pairwise simplex transfers across a shared table, no B-by-N-by-q copy.

Private CPU-tensor execution supports arithmetic tests, not CUDA qualification.
The 10,000-transfer budget and both 1e-8 checks match the reference scorer.
Common row scaling can underflow for a variant excluding the strongest column;
such variants report difficulty and retain their already-fitted CPU result.
    """
    columns = t.as_tensor(columns, dtype=t.float64, device=device)
    index = t.as_tensor(np.array(maps), dtype=t.long, device=device)
    count = t.as_tensor(counts, dtype=t.float64, device=device)
    n, c = columns.shape
    b, q = index.shape
    offset = columns.max(dim=1).values
    if not bool(t.isfinite(offset).all()):
        raise ValueError("Shared weight table has an impossible observation")
    kernel = t.exp(columns-offset[:, None])
    weights = t.zeros((b, c), dtype=t.float64, device=device)
    weights.scatter_(1, index, (count/count.sum())[None, :].expand(b, q))
    allowed = t.zeros((b, c), dtype=t.bool, device=device)
    allowed.scatter_(1, index, True)
    failed = t.zeros(b, dtype=t.bool, device=device)
    steps = t.zeros(b, dtype=t.long, device=device)
    tiny = t.finfo(t.float64).tiny

    def evaluate():
        mass = kernel @ weights.T
        score = ((1/mass.clamp_min(tiny)).T @ kernel)/n
        score = t.where(allowed, score, -t.inf)
        maximum = score.max(dim=1).values
        active_min = t.where(weights > 0, score, t.inf).min(dim=1).values
        return mass, score, (maximum-1).clamp_min(0), maximum-active_min

    for _ in range(10000):
        mass, score, gap, active_gap = evaluate()
        failed |= ~(t.isfinite(mass) & (mass > 0)).all(dim=0)
        running = ~failed & ((gap > 1e-8) | (active_gap > 1e-8))
        if not bool(running.any()):
            break
        grow = score.argmax(dim=1)
        shrink = t.where(weights > 0, score, t.inf).argmin(dim=1)
        failed |= running & (grow == shrink)
        running &= ~failed
        remaining = weights.gather(1, shrink[:, None])[:, 0]
        direction = (kernel[:, grow]-kernel[:, shrink])*remaining[None, :]

        def derivative(fraction):
            return (direction/(mass+direction*fraction[None, :]).clamp_min(tiny)).sum(dim=0)/n

        edge = derivative(t.ones(b, dtype=t.float64, device=device)) >= 0
        lower = t.zeros(b, dtype=t.float64, device=device)
        upper = t.ones_like(lower)
        # Width 2^-48 is smaller than the reference's 1e-14 root tolerance.
        # The CPU reference remains authoritative if different root arithmetic
        # changes support or fails either independent optimality check.
        for _ in range(48):
            middle = (lower+upper)*.5
            positive = derivative(middle) > 0
            lower = t.where(positive, middle, lower)
            upper = t.where(positive, upper, middle)
        fraction = t.where(edge, 1., (lower+upper)*.5)
        transfer = t.where(running, fraction*remaining, 0.)
        weights.scatter_add_(1, grow[:, None], transfer[:, None])
        after = t.where(running & edge, 0., remaining-transfer)
        weights.scatter_(1, shrink[:, None], after[:, None])
        weights /= weights.sum(dim=1)[:, None]
        steps += running
    mass, _, gap, active_gap = evaluate()
    failed |= (~(t.isfinite(mass) & (mass > 0)).all(dim=0)
               | (gap > 1e-8) | (active_gap > 1e-8))
    ll = (offset[:, None]+t.log(mass)).sum(dim=0)
    selected = weights.gather(1, index)
    return [{"cluster_weights": w, "log_likelihood": float(likelihood),
             "weight_optimality_gap": float(g*n), "weight_active_score_gap": float(a),
             "converged": not bool(bad), "transfer_steps": int(step)}
            for w, likelihood, g, a, bad, step in zip(
                selected.cpu().numpy(), ll.cpu().numpy(), gap.cpu().numpy(),
                active_gap.cpu().numpy(), failed.cpu().numpy(), steps.cpu().numpy())]


def qualify_family_weights(family, counts, reference_results, *, device,
                           vram_budget_bytes, torch_module=None):
    """Compare optional device candidates; never replace CPU weights or scores.

``reference_results`` maps anchors to completed CPU result dictionaries. Omit
cache-hit/reference-failed anchors to avoid repeat qualification. Returned
diagnostics are keyed by anchor in original family order. CPU fallback is always
explicit, including duplicate or dependent columns with nonunique optimal zeros.
    """
    anchors = tuple(a for a in family.anchors if a in reference_results)
    counts = np.asarray(counts, dtype=float)
    if counts.shape != (family.q,) or not np.isfinite(counts).all() or np.any(counts <= 0):
        raise ValueError("Initial cluster counts must be positive and finite")
    budget = int(vram_budget_bytes)
    if budget < 0:
        raise ValueError("Device weight workspace budget must be nonnegative")
    result = {a: {"publication_backend": "cpu_reference", "device_qualified": False,
                  "status": "disabled", "scope": "qualification_only"} for a in anchors}
    if not anchors or not budget or not str(device).startswith("cuda"):
        return result
    if torch_module is None:
        import torch as torch_module
    t = torch_module
    if not t.cuda.is_available():
        raise RuntimeError("CUDA weight qualification requested but CUDA is unavailable")
    allocation_errors = (t.OutOfMemoryError,) if hasattr(t, "OutOfMemoryError") else ()
    # Include log/scaled tables, mass/direction/line-search temporaries and small
    # optimizer arrays; this is a workspace ceiling, not a GPU memory request.
    base = family.n*(16*(family.q+1)+8)
    per_variant = 8*(6*family.n+8*(family.q+1)+32)
    width = min(len(anchors), max(0, (budget-base)//per_variant))
    if not width:
        for diag in result.values():
            diag["status"] = "insufficient_workspace"
        return result
    for first in range(0, len(anchors), width):
        selected = anchors[first:first+width]
        maps = family.maps[[family.anchors.index(a) for a in selected]]
        try:
            with t.no_grad():
                candidates = _tensor_weight_solve(t, np.array(family.columns), counts, maps, device=device)
        except allocation_errors as error:
            for anchor in selected:
                result[anchor].update(status="workspace_allocation_failed", error=str(error))
            continue
        for anchor, candidate in zip(selected, candidates):
            reference = reference_results[anchor]
            delta = abs(candidate["log_likelihood"]-reference["log_likelihood"])
            support = np.array_equal(candidate["cluster_weights"] > 0, reference["cluster_weights"] > 0)
            weights_close = np.allclose(candidate["cluster_weights"], reference["cluster_weights"],
                                        rtol=1e-6, atol=1e-7)
            audit = _audit_weights(family.matrix(anchor), candidate["cluster_weights"])
            parity = bool(candidate["converged"] and support and weights_close
                          and audit["passed"] and delta <= family.n*1e-8+1e-9)
            result[anchor].update(status="reference_agreement" if parity else "reference_difference",
                device_qualified=parity, support_equal=support, weights_close=bool(weights_close),
                log_likelihood_absolute_difference=_finite(delta),
                max_weight_absolute_difference=_finite(np.max(np.abs(
                    candidate["cluster_weights"]-reference["cluster_weights"]))),
                weight_optimality_gap=_finite(candidate["weight_optimality_gap"]),
                weight_active_score_gap=_finite(candidate["weight_active_score_gap"]),
                independent_host_audit=audit,
                transfer_steps=candidate["transfer_steps"],
                workspace_allowance_bytes=budget, workspace_estimate_bytes=base+per_variant*len(selected))
    return result


def _finite(value):
    return float(value) if np.isfinite(value) else None


def _audit_weights(matrix, weights):
    """Reference-order independent host certification of a device candidate."""
    n, q = matrix.shape
    if (weights.shape != (q,) or not np.isfinite(weights).all() or np.any(weights < 0)
            or abs(float(weights.sum())-1.) > 1e-12):
        return {"passed": False, "reason": "invalid_simplex_weights"}
    offset = matrix.max(axis=1)
    kernel = np.exp(matrix-offset[:, None])
    mass = kernel @ weights
    if not np.isfinite(mass).all() or np.any(mass <= 0):
        return {"passed": False, "reason": "invalid_mixture_mass"}
    scores = (kernel/mass[:, None]/n).sum(axis=0)
    gap = n*max(0., float(scores.max()-1.))
    active_gap = float(scores.max()-scores[weights > 0].min())
    return {"passed": gap <= n*1e-8 and active_gap <= 1e-8,
            "weight_optimality_gap": gap, "weight_active_score_gap": active_gap,
            "log_likelihood": float((offset+np.log(mass)).sum())}
