"""Complete fixed-q single-mutation refinement under the existing refit policy.

The certificate belongs to an exact fitted state, not just a partition or score.
No conditional-gain pruning or iteration limit can close this neighborhood.
"""

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

from .candidates import score_key
from .proposals import SCORE_ATOL, SCORE_RTOL
from ..io.data import readonly_array
from ..kernel.backend import ScientificCandidateRejection
from ..kernel.topology import canonical_partition

AUDIT_POLICY = "single_mutation_existing_conditional_refit_v1"


class IncompleteMembershipAudit(RuntimeError):
    """A required terminal neighbor lacks a resolved numerical evaluation."""

    def __init__(self, audit):
        self.audit = audit
        super().__init__(f"Incomplete membership audit for state {audit.state_index}, "
                         f"q={audit.occupied_q}: {audit.unresolved_evaluations} unresolved evaluations")


@dataclass(frozen=True, slots=True)
class SingleMove:
    row: int
    destination: int
    outcome: str
    reason: str | None = None
    attempt_index: int | None = None


@dataclass(frozen=True, slots=True)
class MembershipAudit:
    state_index: int
    state_fingerprint: str
    model_identity: str
    numerical_identity: tuple
    numerical_policy: str
    occupied_q: int
    mutation_count: int
    moves: tuple
    complete: bool = True

    @property
    def unresolved_evaluations(self):
        return sum(move.outcome == "unresolved_numerical" for move in self.moves)


def fitted_state_fingerprint(model, tree, candidate, numerical_policy):
    """Length-framed exact identity; native source IDs and refit policy included."""
    fit = candidate.fit
    digest = hashlib.sha256()
    fields = [json.dumps([AUDIT_POLICY, model.identity, tree.identity,
                          model.backend.policy, model.backend.identity, numerical_policy],
                         sort_keys=True, separators=(",", ":")).encode(),
              fit.membership.kind.encode(), np.asarray(fit.labels, dtype=np.int64).tobytes(),
              np.asarray(fit.centers, dtype=float).tobytes(),
              np.asarray(fit.weights, dtype=float).tobytes(),
              np.asarray([fit.conditional_log_likelihood, fit.mixture_log_likelihood,
                          fit.complexity_penalty, fit.score], dtype=float).tobytes()]
    for value in fields:
        digest.update(len(value).to_bytes(8, "little"))
        digest.update(value)
    return digest.hexdigest()


def _structural_reason(model, prepared):
    # Use the same original all-member domains and observed-row support as the
    # conditional refitter, not admissibility at the incumbent destination.
    for group in prepared.groups:
        for region in range(model.r):
            try:
                model._coordinate_task(group, region)
            except ScientificCandidateRejection as error:
                reason = str(error)
                if not reason.startswith(("unsupported_cluster_region:", "empty_cluster_region_box:")):
                    raise
                return reason
    return None


def audit_single_moves(model, bank, parent, *, batch_size=1):
    """Evaluate every feasible neighbor from one immutable parent, row-major.

    Frozen-pass batching changes execution only. Logical publication and cache
    accounting still use the candidate bank's ordinary ordered refit path.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Membership audit batch_size must be a positive integer")
    if (type(parent.state_index) is not int or not 0 <= parent.state_index < len(bank.candidates)
            or bank.candidates[parent.state_index] is not parent):
        raise ValueError("Membership audit parent must be an exact retained state")
    labels = readonly_array(parent.fit.labels, dtype=np.int64)
    centers = readonly_array(parent.fit.centers, dtype=float)
    q = len(centers)
    fingerprint = fitted_state_fingerprint(model, bank.tree, parent, bank.numerical_policy)
    moves, pending = [], []
    prepared = None
    if q > 1:
        columns = model.joint_column_set(centers)
        prepared = bank.prepare(parent.cuts, labels, fitted=parent.fit, columns=columns)
    sizes = np.bincount(labels, minlength=q)

    def flush():
        if not pending:
            return
        requests = [request for _, request in pending]
        if batch_size == 1:
            _, proposed, initial, provenance, parent_partition = requests[0]
            results = [bank.consider_membership(proposed, initial, provenance, parent=parent_partition)]
        else:
            results = bank.consider_independent(requests)
        for (index, request), candidate in zip(pending, results, strict=True):
            _, proposed, initial, _, _ = request
            attempt_index = bank.refit_attempt_index(proposed, initial)
            attempt = bank._outcome_records[attempt_index]
            if (attempt.outcome == "eligible") != (candidate is not None):
                raise ValueError("Membership audit and candidate outcome disagree")
            move = moves[index]
            moves[index] = SingleMove(move.row, move.destination, attempt.outcome,
                                      attempt.reason, attempt_index)
        pending.clear()

    for row in range(model.n):
        source = int(labels[row])
        for destination in range(q):
            if destination == source:
                continue
            if sizes[source] == 1:
                flush()
                moves.append(SingleMove(row, destination, "structural_exclusion", "empty_source_cluster"))
                continue
            proposed = labels.copy()
            proposed[row] = destination
            canonical, aligned = canonical_partition(proposed, centers)
            neighbor = bank.prepare(None, canonical, parent=prepared)
            reason = _structural_reason(model, neighbor)
            if reason is not None:
                flush()
                moves.append(SingleMove(row, destination, "structural_exclusion", reason))
                continue
            index = len(moves)
            moves.append(SingleMove(row, destination, "unresolved_numerical"))
            provenance = {"route": AUDIT_POLICY, "parent_state_index": parent.state_index,
                          "parent_state_fingerprint": fingerprint, "occupied_q": q,
                          "mutation_index": row, "destination": destination}
            pending.append((index, (None, canonical, aligned, provenance, prepared)))
            if len(pending) >= batch_size:
                flush()
    flush()
    if len(moves) != model.n*(q-1):
        raise AssertionError("Incomplete single-mutation neighborhood enumeration")
    if fitted_state_fingerprint(model, bank.tree, parent, bank.numerical_policy) != fingerprint:
        raise ValueError("Membership audit parent changed during evaluation")
    complete = all(move.outcome != "unresolved_numerical" for move in moves)
    return MembershipAudit(parent.state_index, fingerprint, model.identity,
                           (model.backend.policy, model.backend.identity), bank.numerical_policy,
                           q, model.n, tuple(moves), complete)


def refine_memberships(model, bank, *, batch_size=1):
    """Audit the exact best retained state at every represented occupied q.

    Even a sub-tolerance or tie-rule winner change requires another full pass.
    An unresolved earlier pass can advance, but never certify a terminal state.
    No later stage may add candidates without invalidating these certificates.
    """
    terminal, passes = [], []
    represented = sorted({len(candidate.fit.centers) for candidate in bank.candidates})
    for q in represented:
        while True:
            parent = min((c for c in bank.candidates if len(c.fit.centers) == q), key=score_key)
            audit = audit_single_moves(model, bank, parent, batch_size=batch_size)
            current = min((c for c in bank.candidates if len(c.fit.centers) == q), key=score_key)
            record = {"parent_state_index": parent.state_index, "occupied_q": q,
                      "examined_pairs": len(audit.moves),
                      "unresolved_evaluations": audit.unresolved_evaluations,
                      "winning_state_index": current.state_index,
                      "score_improvement": parent.score-current.score,
                      "meaningful_improvement": parent.score-current.score >
                          SCORE_ATOL+SCORE_RTOL*max(abs(parent.score), abs(current.score))}
            if bank.retain_evidence:
                record["audit"] = asdict(audit)
            passes.append(record)
            if current.state_index != parent.state_index:
                continue
            if not audit.complete:
                raise IncompleteMembershipAudit(audit)
            terminal.append(audit)
            break
    return tuple(terminal), passes
