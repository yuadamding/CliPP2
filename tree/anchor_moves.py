"""Bounded atomic anchor edits; proposals, never a new scientific objective.

Intermediate edits need not be feasible or downhill. Only the complete edited
partition is screened, then the ordinary candidate bank refits and scores it.
The finite shortlist is explicitly not an exhaustive joint neighborhood.
"""

from dataclasses import dataclass
import hashlib
from itertools import islice, product, zip_longest
import math

import numpy as np

from ..io.data import readonly_array
from ..kernel.objectives import assigned_objective


MASK_WIDTH = 2
PACKET_SIZE = 3
PACKET_WIDTH = 8
WEAK_LIMIT = 8
FAMILY_ATTEMPTS = 128


@dataclass(frozen=True)
class AnchorProposal:
    labels: np.ndarray
    centers: np.ndarray
    clonal_cluster: int
    target_q: int
    proposal_id: str
    details: dict

    def __post_init__(self):
        for name in ("labels", "centers"):
            object.__setattr__(self, name, readonly_array(getattr(self, name)))


def _mask(values):
    # Python integers also support panels with more than 64 regions.
    return sum(1 << int(region) for region in np.flatnonzero(values))


def _canonical(labels, centers, anchor):
    order = list(dict.fromkeys(map(int, labels)))
    if anchor not in order:
        raise ValueError("An atomic anchor edit cannot delete its designated anchor")
    mapping = {old: new for new, old in enumerate(order)}
    return (np.asarray([mapping[int(k)] for k in labels], dtype=np.int64),
            np.asarray(centers, dtype=float)[order].copy(), mapping[anchor], order)


def _destinations(rows, anchor, columns, keys):
    """Deterministic finite assignments, without sequential legality gates."""
    options = []
    for row in rows:
        destinations = [k for k in range(columns.shape[1])
                        if k != anchor and np.isfinite(columns[row, k])]
        destinations.sort(key=lambda k: (-float(columns[row, k]), k))
        if not destinations:
            return
        options.append(destinations)
    seen = set()
    plans = [tuple(destinations[0] for destinations in options)]
    # Small packets may send complementary-missing rows to different blocks.
    if len(rows) <= PACKET_SIZE:
        plans.extend(product(*(destinations[:2] for destinations in options)))
    plans.extend((k,) * len(rows) for k in range(columns.shape[1])
                 if k != anchor and all(k in destinations for destinations in options))
    plans.sort(key=lambda values: (
        -math.fsum(columns[row, k] for row, k in zip(rows, values)), values))
    for plan in plans:
        if plan not in seen:
            seen.add(plan)
            yield plan


def _packets(model, labels, anchor, columns, keys):
    """Observation-mask beam, not one-move feasibility or a size prior."""
    assigned = columns[np.arange(len(labels)), labels]
    gain = np.asarray(model.clonal_log_column) - assigned
    donors = [i for i in range(len(labels)) if labels[i] != anchor
              and model.clonal_eligible[i] and np.isfinite(gain[i])]
    donors.sort(key=lambda i: (-float(gain[i]), keys[i], i))
    by_mask = {}
    for row in donors:
        group = by_mask.setdefault(_mask(model.observed[row]), [])
        if len(group) < MASK_WIDTH:
            group.append(row)
    pool = sorted((row for group in by_mask.values() for row in group),
                  key=lambda i: (-float(gain[i]), keys[i], i))
    # The number of observed masks is at most 64 for six regions. For wider
    # panels cap states explicitly instead of allocating an exponential array.
    pool = pool[:128]
    layers, packets = {0: [()]}, []
    for _ in range(PACKET_SIZE):
        next_layer = {}
        for prior_mask, choices in layers.items():
            for packet in choices:
                for row in pool:
                    if row in packet:
                        continue
                    proposal = tuple(sorted((*packet, row)))
                    mask = prior_mask | _mask(model.observed[row])
                    options = next_layer.setdefault(mask, set())
                    options.add(proposal)
        layers = {}
        for mask in sorted(next_layer, key=lambda value: (-value.bit_count(), value))[:128]:
            choices = sorted(next_layer[mask], key=lambda packet: (
                -math.fsum(gain[list(packet)]), tuple(keys[i] for i in packet), packet))
            layers[mask] = choices[:MASK_WIDTH]
            packets.extend((mask, packet, math.fsum(gain[list(packet)]))
                           for packet in layers[mask])
    return packets, {"eligible_donors": len(donors), "observed_donor_masks": len(by_mask),
                     "retained_donor_rows": len(pool), "mask_state_cap": 128,
                     "packet_size_cap": PACKET_SIZE, "packets_shortlisted": len(packets)}


def _interleave(streams):
    for row in zip_longest(*streams):
        yield from (item for item in row if item is not None)


def joint_proposals(model, candidate, columns, *, limit, keys=None):
    """Return a stable ranked prefix and honest finite-neighborhood coverage.

    ``limit`` changes only the returned prefix, never the declared generators.
    A proposal's ``target_q`` is its actual occupied count; deletion must not be
    routed through a same-K budget. The designated anchor is remapped with the
    labels/centers after deletion, never guessed from its old numeric label.
    """
    if isinstance(limit, bool) or not isinstance(limit, (int, np.integer)) or limit < 0:
        raise ValueError("Joint proposal limit must be a nonnegative integer")
    fit = candidate.refit
    labels, centers = np.asarray(fit.labels), np.asarray(fit.centers)
    columns = np.asarray(columns, dtype=float)
    anchor, n, q = int(fit.clonal_cluster), len(labels), len(centers)
    if (labels.shape != (model.n,) or labels.dtype.kind not in "iu"
            or not np.array_equal(np.unique(labels), np.arange(q))
            or centers.shape != (q, model.r) or columns.shape != (n, q)
            or not 0 <= anchor < q or not np.array_equal(centers[anchor], np.ones(model.r))
            or np.isnan(columns).any() or np.isposinf(columns).any()
            or not np.isfinite(columns[np.arange(n), labels]).all()):
        raise ValueError("Atomic anchor moves require an eligible aligned anchored parent")
    keys = tuple(model.mutation_ids if keys is None else keys)
    if len(keys) != n:
        raise ValueError("Atomic anchor keys must align with mutation rows")
    rows = np.flatnonzero(labels == anchor).tolist()
    best_other = columns.copy()
    best_other[:, anchor] = -np.inf
    alternatives = best_other.max(axis=1)
    weak = sorted(rows, key=lambda i: (
        -float(alternatives[i]-columns[i, anchor]), keys[i], i))[:WEAK_LIMIT]
    packets, packet_coverage = _packets(model, labels, anchor, columns, keys)
    remove_sets = [tuple(weak[:count]) for count in range(1, len(weak)+1)]
    if len(rows) <= WEAK_LIMIT and tuple(sorted(rows)) not in remove_sets:
        remove_sets.append(tuple(sorted(rows)))

    def replacements(outgoing):
        retained = np.array([i for i in rows if i not in outgoing], dtype=int)
        covered = _mask(model.observed[retained].any(axis=0)) if len(retained) else 0
        required = ((1 << model.r)-1) & ~covered
        donors = sorted((item for item in packets if item[0] & required == required),
                        key=lambda item: (-item[2], item[1]))[:PACKET_WIDTH]
        plans = tuple(islice(_destinations(outgoing, anchor, columns, keys), PACKET_WIDTH))
        for packet, destinations in product((item[1] for item in donors), plans):
            edited = labels.copy()
            edited[list(outgoing)] = destinations
            edited[list(packet)] = anchor
            kind = "anchor_swap" if len(packet) == len(outgoing) == 1 else "anchor_packet_replacement"
            yield edited, centers, anchor, columns, {
                "family": kind, "donor_rows": list(packet), "evacuated_rows": list(outgoing),
                "evacuation_destinations": list(destinations)}

    def reanchors(new_anchor):
        fixed = centers.copy()
        fixed[new_anchor] = 1.
        emission = columns.copy()
        emission[:, new_anchor] = model.clonal_log_column
        for destinations in _destinations(rows, anchor, emission, keys):
            edited = labels.copy()
            edited[rows] = destinations
            yield edited, fixed, new_anchor, emission, {
                "family": "reanchor_evacuation", "old_anchor": anchor,
                "new_anchor_before_canonicalization": new_anchor,
                "evacuated_rows": rows, "evacuation_destinations": list(destinations)}

    alternate_anchors = [k for k in range(q) if k != anchor
                         and np.all(model.clonal_eligible[labels == k])]
    alternate_anchors.sort(key=lambda k: (
        -math.fsum(model.clonal_log_column[labels == k]-columns[labels == k, k]), k))
    streams = {
        "anchor_replacement": _interleave([replacements(outgoing) for outgoing in remove_sets]),
        "reanchor_evacuation": _interleave([reanchors(k) for k in alternate_anchors]),
    }
    coverage = {"schema": "clipp2_atomic_anchor_neighborhood_v1", "declared": True,
                "scope": "bounded_mask_packets_reanchor_evacuation_and_anchor_swaps_not_exhaustive",
                "all_joint_neighbors_checked": False, "sequential_feasibility_required": False,
                "weak_member_cap": WEAK_LIMIT, "packet_width": PACKET_WIDTH,
                "attempts_per_family_cap": FAMILY_ATTEMPTS, "packet_search": packet_coverage,
                "alternate_anchor_blocks": len(alternate_anchors), "families": {}}
    proposed, identities = [], set()
    for family, stream in streams.items():
        stats = {"attempted": 0, "feasibility_rejected": 0, "duplicates": 0, "valid_unique": 0}
        coverage["families"][family] = stats
        for edited, fixed, new_anchor, emission, details in islice(stream, FAMILY_ATTEMPTS):
            stats["attempted"] += 1
            canonical, aligned, remapped_anchor, order = _canonical(edited, fixed, new_anchor)
            summary = model.regional_support.summarize_partition(canonical)
            invalid = model.regional_support.check_partition(summary)
            expected_q = q-1 if family == "reanchor_evacuation" else q
            if (len(order) != expected_q or invalid
                    or not np.all(model.clonal_eligible[canonical == remapped_anchor])
                    or not model.regional_support.assigned_mask(canonical, aligned).all()
                    or not np.isfinite(emission[np.arange(n), edited]).all()
                    or np.any(aligned[canonical] < model.lower)
                    or np.any(aligned[canonical] > model.upper)):
                stats["feasibility_rejected"] += 1
                continue
            identity = (canonical.tobytes(), remapped_anchor, aligned.tobytes())
            if identity in identities:
                stats["duplicates"] += 1
                continue
            identities.add(identity)
            read_ll = math.fsum(emission[np.arange(n), edited])
            objective = getattr(model, "selection_objective", "classification")
            ranking_objective = "classification" if objective == "legacy_mixture" else objective
            scored = assigned_objective(read_ll, np.bincount(canonical), model.r, ranking_objective)
            digest = hashlib.sha256(b"clipp2_atomic_anchor_proposal_v1")
            digest.update(canonical.astype("<i8").tobytes())
            digest.update(aligned.astype("<f8").tobytes())
            digest.update(str(remapped_anchor).encode())
            proposal_id = digest.hexdigest()
            details.update(proposal_id=proposal_id, target_q=len(order),
                           fixed_center_ranking_score=scored["score"],
                           ranking_objective=ranking_objective,
                           fixed_center_classification_count_term=scored["classification_count_term"],
                           fixed_center_complexity_penalty=scored["complexity_penalty"],
                           fixed_center_conditional_log_likelihood=read_ll,
                           original_to_canonical_label={str(old): new for new, old in enumerate(order)},
                           feasibility="complete_partition_atomic", requires_conditional_refit=True)
            proposed.append(AnchorProposal(canonical, aligned, remapped_anchor, len(order),
                                           proposal_id, details))
            stats["valid_unique"] += 1
        stats["generator_shortlist_exhausted"] = next(stream, None) is None
    proposed.sort(key=lambda proposal: (proposal.details["fixed_center_ranking_score"],
                                       proposal.proposal_id))
    family_order = ("anchor_packet_replacement", "reanchor_evacuation", "anchor_swap")
    by_family = {name: [proposal for proposal in proposed if proposal.details["family"] == name]
                 for name in family_order}
    # One slot per available family per cycle. A low-score reanchor stream
    # cannot starve packet replacements/swaps; increasing the allowance retains
    # the identical concrete prefix. This ordering is a search allocation, not
    # a changed scoring criterion or an imposed anchor preference.
    allocated = tuple(_interleave([by_family[name] for name in family_order]))
    returned = allocated[:limit]
    coverage.update(total_generated=len(proposed), returned=len(returned),
                    truncated=len(proposed)-len(returned),
                    exhausted=len(returned) == len(proposed),
                    declared_generation_complete=True,
                    family_allocation="one_slot_per_nonempty_family_per_cycle",
                    proposal_families={name: {
                        "available": len(by_family[name]),
                        "returned": sum(proposal.details["family"] == name for proposal in returned),
                        "all_declared_proposals_returned": sum(
                            proposal.details["family"] == name for proposal in returned) == len(by_family[name])}
                        for name in family_order},
                    proposal_ids=[proposal.proposal_id for proposal in returned])
    return returned, coverage


def anchor_diagnostics(model, candidate, columns=None, *, fit_free_anchor=False):
    """Evidence for the designated block; no biological or panel certificate."""
    fit = candidate.fit
    labels, anchor = np.asarray(fit.labels), int(fit.clonal_cluster)
    rows = np.flatnonzero(labels == anchor)
    observed = np.asarray(model.observed)[rows]
    counts = observed.sum(axis=0)
    positive = (np.asarray(model.alt)[rows] > 0) & observed
    diagnostics = fit.diagnostics
    loss = diagnostics.get("anchoring_loss")
    loss_reason = diagnostics.get("anchoring_loss_unavailable_reason")
    # Optional diagnostic work must not consume hidden finalization refits.
    # The caller explicitly opts in and must account for this extra fit.
    extra_fits = 0
    if loss is None and fit_free_anchor:
        try:
            free = []
            for region in range(model.r):
                extra_fits += 1
                _, value = model._fit_scalar(rows, region)
                free.append(value)
            clonal = math.fsum(model.clonal_log_column[rows])
            loss = max(0., math.fsum(free)-clonal)
            loss_reason = None
        except (ValueError, RuntimeError) as error:
            loss_reason = f"diagnostic_free_anchor_fit_failed:{error}"
    result = {"schema": "clipp2_anchor_evidence_v1", "clonal_cluster": anchor,
              "members": len(rows), "mutation_ids": [model.mutation_ids[i] for i in rows],
              "region_ids": list(model.region_ids), "observed_members_per_region": counts.tolist(),
              "positive_ALT_members_per_region": positive.sum(axis=0).tolist(),
              "single_observation_support_regions": [model.region_ids[r] for r in np.flatnonzero(counts == 1)],
              "critical_support_mutations": [
                  {"mutation_id": model.mutation_ids[row],
                   "regions": [model.region_ids[r] for r in np.flatnonzero(observed[j] & (counts == 1))]}
                  for j, row in enumerate(rows) if np.any(observed[j] & (counts == 1))],
              "total_anchoring_loss": loss,
              "anchoring_loss_per_member_diagnostic_only": None if loss is None else float(loss)/len(rows),
              "anchoring_loss_available": loss is not None,
              "anchoring_loss_unavailable_reason": loss_reason,
              "diagnostic_free_anchor_fit_opt_in": bool(fit_free_anchor),
              "diagnostic_free_anchor_regional_fit_calls": extra_fits,
              "single_cell_panel_coverage": None,
              "single_cell_panel_coverage_status": "not_supplied_no_claim",
              "biological_support_certified": False}
    if columns is not None:
        columns = np.asarray(columns)
        alternatives = columns[rows].copy()
        alternatives[:, anchor] = -np.inf
        assigned, other = columns[rows, anchor], alternatives.max(axis=1)
        result["assigned_vs_alternative_read_likelihood"] = [
            {"mutation_id": model.mutation_ids[row], "assigned_read_log_likelihood": float(assigned[j]),
             "best_alternative_cluster": int(alternatives[j].argmax()) if np.isfinite(other[j]) else None,
             "best_alternative_read_log_likelihood": float(other[j]) if np.isfinite(other[j]) else None,
             "alternative_read_gain": float(other[j]-assigned[j]) if np.isfinite(other[j]) else None}
            for j, row in enumerate(rows)]
    return result
