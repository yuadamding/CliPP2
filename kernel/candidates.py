"""Shared registry of compact memberships, refit attempts and immutable states."""

from dataclasses import dataclass, field
from time import perf_counter
from types import MappingProxyType

import numpy as np

from ..io.data import readonly_array
from ..tree.refinement import canonical_partition
from .topology import FrozenTree, partition_from_cuts


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(map(_freeze, value))
    return readonly_array(value) if isinstance(value, np.ndarray) else value


def _plain(value):
    if isinstance(value, (dict, MappingProxyType)):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return list(map(_plain, value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value.item() if isinstance(value, np.generic) else value


@dataclass(frozen=True, eq=False)
class FittedState:
    """One region-axis-preserving state for either membership representation."""

    tree: object
    cuts: tuple | None
    explicit_labels: np.ndarray | None
    centers: np.ndarray
    weights: np.ndarray
    conditional_log_likelihood: float
    mixture_log_likelihood: float
    complexity_penalty: float
    score: float
    eligible: bool
    reason: str | None
    diagnostics: object

    def __post_init__(self):
        for name in ("centers", "weights", "explicit_labels"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, readonly_array(value))
        object.__setattr__(self, "diagnostics", _freeze(dict(self.diagnostics)))

    @property
    def labels(self):
        return (readonly_array(partition_from_cuts(self.tree, self.cuts))
                if self.cuts is not None else self.explicit_labels)

    @property
    def penalty(self):
        return self.complexity_penalty


@dataclass(eq=False)
class Candidate:
    cuts: tuple | None
    refit: object
    provenance: list = field(default_factory=list)
    membership_kind: str = "tree"

    @property
    def score(self):
        return float(self.refit.score)

    @property
    def fit(self):
        return self.refit

    @property
    def requested_k(self):
        return self.provenance[0].get("requested_k", len(self.refit.centers))

    @property
    def kind(self):
        return self.provenance[0].get("route", "unspecified")

    @property
    def diagnostics(self):
        return {"provenance": _plain(self.provenance), "refit": _plain(self.refit.diagnostics)}


def score_key(candidate):
    tree = candidate.membership_kind == "tree"
    return (candidate.score, len(candidate.refit.centers), 0 if tree else 1,
            candidate.cuts if tree else tuple(candidate.refit.labels))


def _canonical_inputs(labels, centers, n, r):
    labels = np.asarray(labels)
    if (labels.shape != (n,) or labels.dtype.kind not in "iuf"
            or not np.isfinite(labels).all() or np.any(labels != np.rint(labels))):
        raise ValueError("Candidate labels must be one finite integer per mutation")
    occupied = np.unique(labels)
    if not np.array_equal(occupied, np.arange(len(occupied))):
        raise ValueError("Candidate labels must be occupied consecutive IDs starting at zero")
    labels = labels.astype(np.int64)
    if centers is None:
        return canonical_partition(labels), None
    centers = np.asarray(centers, dtype=float)
    if centers.shape != (len(occupied), r) or not np.isfinite(centers).all():
        raise ValueError("Supplied centers must be finite occupied q by R rows aligned with labels")
    return canonical_partition(labels, centers)


class CandidateBank:
    """Retain eligible states before advancement; cache only exact attempts."""

    numerical_policy = "regional_conditional_refit_v1"

    def __init__(self, model, tree):
        self.model, self.tree = model, tree
        self.candidates, self.general_rejections, self.attempt_records = [], [], []
        self.by_cuts, self.by_membership, self.ineligible = {}, {}, {}
        self.membership_attempts, self.structural_rejections = {}, {}
        self._attempts, self._states, self._structural = {}, {}, {}
        self.duplicate_routes = self.refit_calls = 0
        self.attempt_cache_hits = self.structural_cache_hits = 0
        self.refit_seconds = 0.0

    def consider(self, cuts, labels, centers, provenance):
        raw = np.asarray(cuts)
        if raw.ndim != 1 or not np.isfinite(raw).all() or np.any(raw != np.rint(raw)):
            raise ValueError("Tree cuts must be finite integer edge IDs")
        cuts = tuple(sorted(map(int, cuts)))
        if len(set(cuts)) != len(cuts) or any(not 0 <= edge < len(self.tree.edges) for edge in cuts):
            raise ValueError("Tree cuts must be distinct valid edge IDs")
        labels, centers = _canonical_inputs(labels, centers, self.model.n, self.model.r)
        if not np.array_equal(labels, partition_from_cuts(self.tree, cuts)):
            raise ValueError("Candidate labels and frozen-tree cuts disagree")
        return self._consider(("tree", cuts), cuts, labels, centers, provenance)

    def consider_membership(self, labels, initial_centers, provenance):
        labels, centers = _canonical_inputs(labels, initial_centers, self.model.n, self.model.r)
        candidate = self._consider(("general", labels.tobytes()), None, labels, centers, provenance)
        self.membership_attempts[(labels.tobytes(), None if centers is None else centers.tobytes())] = candidate
        return candidate

    def _consider(self, membership, cuts, labels, centers, provenance):
        provenance = _plain(provenance)
        if membership in self._structural:
            self.structural_cache_hits += 1
            self._structural[membership]["provenance"].append(provenance)
            return None
        attempt = (membership, None if centers is None else centers.tobytes(), self.numerical_policy)
        if attempt in self._attempts:
            self.attempt_cache_hits += 1
            candidate = self._attempts[attempt]
            if candidate is not None:
                candidate.provenance.append(provenance)
                self.duplicate_routes += 1
            return candidate
        started = perf_counter()
        self.refit_calls += 1
        try:
            fitted = self.model.refit(labels, initial_centers=centers)
        finally:
            self.refit_seconds += perf_counter()-started
        record = {"membership_kind": membership[0],
                  "cuts" if cuts is not None else "labels": list(cuts) if cuts is not None else labels.tolist(),
                  "initial_centers_ccf": None if centers is None else centers.tolist(),
                  "numerical_policy": self.numerical_policy, "provenance": provenance,
                  "eligible": bool(fitted.eligible), "reason": fitted.reason}
        self.attempt_records.append(record)
        self._attempts[attempt] = None
        if not fitted.eligible:
            rejection = {"cuts" if cuts is not None else "labels": record.get("cuts", record.get("labels")),
                         "reason": fitted.reason, "provenance": [provenance]}
            if cuts is None:
                self.general_rejections.append(rejection)
            else:
                self.ineligible[cuts] = rejection
            if str(fitted.reason).startswith(("unsupported_cluster_region", "empty_cluster_region_box")):
                self._structural[membership] = rejection
                if cuts is None:
                    self.structural_rejections[labels.tobytes()] = rejection
            return None
        canonical, aligned = _canonical_inputs(fitted.labels, fitted.centers, self.model.n, self.model.r)
        if not np.array_equal(canonical, labels):
            raise ValueError("Conditional refit changed the proposed membership")
        order = list(dict.fromkeys(np.asarray(fitted.labels, dtype=int).tolist()))
        weights = np.asarray(fitted.weights, dtype=float)
        if (weights.shape != (len(order),) or not np.isfinite(weights).all()
                or np.any(weights <= 0) or not np.isclose(weights.sum(), 1., atol=1e-12, rtol=0)):
            raise ValueError("Eligible fitted weights must be positive normalized occupied rows")
        if not np.isfinite([fitted.conditional_log_likelihood, fitted.mixture_log_likelihood,
                            fitted.complexity_penalty, fitted.score]).all():
            raise ValueError("An eligible candidate has nonfinite likelihood, penalty or score")
        weights = weights[order]
        state_key = membership, aligned.tobytes(), weights.tobytes()
        candidate = self._states.get(state_key)
        if candidate is None:
            state = FittedState(self.tree, cuts, labels if cuts is None else None, aligned, weights,
                                float(fitted.conditional_log_likelihood), float(fitted.mixture_log_likelihood),
                                float(fitted.complexity_penalty), float(fitted.score), True,
                                fitted.reason, fitted.diagnostics)
            candidate = Candidate(cuts, state, [provenance], membership[0])
            self.candidates.append(candidate)
            self._states[state_key] = candidate
        else:
            if any(getattr(candidate.refit, name) != getattr(fitted, name) for name in (
                    "conditional_log_likelihood", "mixture_log_likelihood", "complexity_penalty", "score")):
                raise ValueError("Identical fitted parameters produced inconsistent candidate scores")
            candidate.provenance.append(provenance)
            self.duplicate_routes += 1
        self._attempts[attempt] = candidate
        record["state_index"] = next(i for i, item in enumerate(self.candidates) if item is candidate)
        index, key = (self.by_cuts, cuts) if cuts is not None else (self.by_membership, labels.tobytes())
        previous = index.get(key)
        if previous is None or score_key(candidate) < score_key(previous):
            index[key] = candidate
        return candidate

    def evidence(self, selected):
        return candidate_evidence(self.tree, self.candidates, selected, self.attempt_records)


def candidate_evidence(tree, candidates, selected, attempts=()):
    """JSON-ready replay payload; connected memberships never store N labels."""
    states = []
    for candidate in candidates:
        fit = candidate.refit
        states.append({
            "membership_kind": candidate.membership_kind,
            "cuts" if candidate.cuts is not None else "labels":
                list(candidate.cuts) if candidate.cuts is not None else fit.labels.tolist(),
            "centers_ccf": fit.centers.tolist(), "weights": fit.weights.tolist(),
            "conditional_log_likelihood": float(fit.conditional_log_likelihood),
            "mixture_log_likelihood": float(fit.mixture_log_likelihood),
            "complexity_penalty": float(fit.complexity_penalty), "score": float(fit.score),
            "eligible": bool(fit.eligible), "reason": fit.reason,
            "diagnostics": _plain(fit.diagnostics), "provenance": _plain(candidate.provenance),
        })
    selected_index = next((i for i, candidate in enumerate(candidates) if candidate is selected), None)
    if selected_index is None:
        raise ValueError("Selected fitted state is absent from the candidate bank")
    return {"version": 1, "tree": {"identity": tree.identity, "n": tree.n,
            "edges": tree.edges.tolist(), "kind": tree.kind,
            "mutation_ids": list(tree.mutation_ids)},
            "selected_state_index": selected_index, "states": states,
            "refit_attempts": _plain(attempts)}


def replay_candidate_evidence(evidence):
    """Reconstruct memberships and replay score selection, not refit numerics.

    The persisted centers and weights permit a separate likelihood replay with
    original inputs. This check alone does not verify those inputs/emissions.
    """
    if evidence.get("version") != 1:
        raise ValueError("Unknown candidate-evidence version")
    record = evidence["tree"]
    edges = (np.empty((0, 2), dtype=np.int64) if record["n"] == 1 and record["edges"] == []
             else np.asarray(record["edges"]))
    tree = FrozenTree(edges, record["n"], tuple(record["mutation_ids"]), record["kind"])
    if tree.identity != record["identity"]:
        raise ValueError("Candidate evidence tree identity mismatch")
    candidates, region_count = [], None
    for item in evidence["states"]:
        kind = item["membership_kind"]
        cuts = tuple(item["cuts"]) if kind == "tree" else None
        if kind not in ("tree", "general"):
            raise ValueError("Unknown candidate membership kind")
        if cuts is not None and (any(type(edge) is not int or not 0 <= edge < len(tree.edges)
                                    for edge in cuts) or tuple(sorted(set(cuts))) != cuts):
            raise ValueError("Candidate evidence has invalid tree cuts")
        labels = partition_from_cuts(tree, cuts) if cuts is not None else np.asarray(item["labels"])
        centers = np.asarray(item["centers_ccf"], dtype=float)
        if (centers.ndim != 2 or not all(centers.shape)
                or not np.isfinite(centers).all()
                or np.any(centers < 0) or np.any(centers > 1)):
            raise ValueError("Candidate evidence needs finite q-by-R CCF centers in [0,1]")
        if region_count is None:
            region_count = centers.shape[1]
        elif centers.shape[1] != region_count:
            raise ValueError("Candidate evidence mixes different region counts")
        canonical, aligned = _canonical_inputs(labels, centers, tree.n, centers.shape[1])
        if not np.array_equal(canonical, labels) or not np.array_equal(aligned, centers):
            raise ValueError("Candidate evidence must retain canonical aligned memberships")
        weights = np.asarray(item["weights"], dtype=float)
        if (weights.shape != (len(centers),) or not np.isfinite(weights).all()
                or np.any(weights <= 0) or not np.isclose(weights.sum(), 1., rtol=0., atol=1e-12)):
            raise ValueError("Candidate evidence has invalid occupied weights")
        q, r = centers.shape
        penalty = float((q*r+q-1)*np.log(tree.n))
        score = -2*float(item["mixture_log_likelihood"])+penalty
        if (item["eligible"] is not True or not np.isfinite(score)
                or not np.isfinite(item["conditional_log_likelihood"])
                or not np.isclose(penalty, item["complexity_penalty"], atol=1e-7, rtol=1e-10)
                or not np.isclose(score, item["score"], atol=1e-7, rtol=1e-10)):
            raise ValueError("Candidate evidence score arithmetic is inconsistent")
        fit = FittedState(tree, cuts, labels if cuts is None else None, centers, weights,
                          item["conditional_log_likelihood"], item["mixture_log_likelihood"],
                          item["complexity_penalty"], item["score"], True,
                          item["reason"], item["diagnostics"])
        candidates.append(Candidate(cuts, fit, item["provenance"], kind))
    # Arithmetic-only callers may omit attempts. A saved search ledger that
    # includes them must account for every retained eligible fitted state.
    attempts = evidence.get("refit_attempts", ())
    if not isinstance(attempts, (list, tuple)):
        raise ValueError("Candidate evidence refit attempts must be a sequence")
    retained = set()
    for attempt in attempts:
        if type(attempt.get("eligible")) is not bool:
            raise ValueError("Refit attempt eligibility must be explicit")
        kind = attempt.get("membership_kind")
        if kind == "tree":
            cuts = tuple(attempt["cuts"])
            if (any(type(edge) is not int or not 0 <= edge < len(tree.edges) for edge in cuts)
                    or tuple(sorted(set(cuts))) != cuts):
                raise ValueError("Refit attempt has invalid tree cuts")
            labels = partition_from_cuts(tree, cuts)
        elif kind == "general":
            cuts, labels = None, np.asarray(attempt["labels"])
            canonical, _ = _canonical_inputs(labels, None, tree.n, region_count)
            if not np.array_equal(canonical, labels):
                raise ValueError("Refit attempt must retain canonical memberships")
        else:
            raise ValueError("Unknown refit-attempt membership kind")
        initial = attempt.get("initial_centers_ccf")
        if initial is not None:
            _canonical_inputs(labels, initial, tree.n, region_count)
        if not attempt["eligible"]:
            if "state_index" in attempt:
                raise ValueError("A rejected refit attempt cannot reference an eligible state")
            continue
        index = attempt.get("state_index")
        if type(index) is not int or not 0 <= index < len(candidates):
            raise ValueError("An eligible refit attempt is missing its retained fitted state")
        candidate = candidates[index]
        if (kind != candidate.membership_kind or cuts != candidate.cuts
                or not np.array_equal(labels, candidate.refit.labels)):
            raise ValueError("Refit attempt and retained fitted-state memberships disagree")
        retained.add(index)
    if attempts and retained != set(range(len(candidates))):
        raise ValueError("Candidate evidence contains fitted states without an eligible refit attempt")
    selected_index = evidence["selected_state_index"]
    if type(selected_index) is not int or not 0 <= selected_index < len(candidates):
        raise ValueError("Candidate evidence has no selected state")
    selected = candidates[selected_index]
    if selected is not min(candidates, key=score_key):
        raise ValueError("Candidate evidence selection is not the complete-bank minimum")
    return selected, tuple(candidates)
