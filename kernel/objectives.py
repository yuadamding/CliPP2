"""Explicit statistical objectives for occupied, anchored hard partitions.

The classification objective includes the likelihood of the *published*
memberships. Its profiled proportions are counts/N, not mixture-optimal
weights. The nominal penalty is a specification to qualify experimentally,
not a claim of regular-model BIC theory for searched discrete partitions.
"""
import math

import numpy as np


SELECTION_OBJECTIVES = ("classification", "legacy_mixture", "assigned_reads")
DEFAULT_SELECTION_OBJECTIVE = "classification"
LEGACY_SCORE_DEFINITION = "occupied_clonal_joint_read_validity_bic_v3"
CLASSIFICATION_SCORE_DEFINITION = "occupied_clonal_profiled_classification_score_v1"
ASSIGNED_READS_SCORE_DEFINITION = "occupied_clonal_assigned_reads_nominal_penalty_v1"


def validate_selection_objective(objective):
    if not isinstance(objective, str) or objective not in SELECTION_OBJECTIVES:
        raise ValueError(f"selection_objective must be one of {SELECTION_OBJECTIVES}")
    return objective


def score_definition_for(objective):
    objective = validate_selection_objective(objective)
    return {"classification": CLASSIFICATION_SCORE_DEFINITION,
            "legacy_mixture": LEGACY_SCORE_DEFINITION,
            "assigned_reads": ASSIGNED_READS_SCORE_DEFINITION}[objective]


def _counts(counts):
    result = np.asarray(counts, dtype=float)
    if (result.ndim != 1 or not result.size or not np.isfinite(result).all()
            or np.any(result < 0) or np.any(result != np.rint(result))
            or result.sum() <= 0):
        raise ValueError("Cluster counts must be finite nonnegative integers with positive sum")
    return result


def count_term(counts):
    """Return sum n_k log(n_k/N), with 0 log 0 = 0."""
    counts = _counts(counts)
    positive = counts[counts > 0]
    return math.fsum(positive * np.log(positive / counts.sum()))


def transfer_count_gain(counts, source, destination):
    """Exact profiled count-term change, scalar or broadcast index arrays.

    Removing the last member of a source returns -inf (fixed occupied q).
    The N terms cancel, so the subtraction uses g(n)=n log n to avoid
    unnecessary large common terms. No count array is mutated.
    """
    counts = _counts(counts)
    source, destination = np.broadcast_arrays(np.asarray(source), np.asarray(destination))
    for indices in (source, destination):
        if (indices.dtype.kind not in "iu" or np.any(indices < 0)
                or np.any(indices >= len(counts))):
            raise ValueError("Transfer source and destination must be valid integer cluster indices")
    a, b = counts[source], counts[destination]

    def g(values):
        return values * np.log(np.maximum(values, 1.))

    gain = g(np.maximum(a-1, 0.)) + g(b+1) - g(a) - g(b)
    gain = np.where(a > 1, gain, -np.inf)
    gain = np.where(source == destination, 0., gain)
    return float(gain) if gain.ndim == 0 else gain


def assigned_objective(assigned_log_likelihood, counts, regions, objective="classification"):
    """Authoritative score/proportions from block summaries, without N-by-q work."""
    objective = validate_selection_objective(objective)
    if objective == "legacy_mixture":
        raise ValueError("legacy_mixture requires its independent mixture-weight optimizer")
    counts = _counts(counts)
    if np.any(counts <= 0):
        raise ValueError("Every reported cluster must be occupied")
    if isinstance(regions, bool) or not isinstance(regions, (int, np.integer)) or regions < 1:
        raise ValueError("The number of fitted regions must be a positive integer")
    assigned = float(assigned_log_likelihood)
    if not np.isfinite(assigned):
        raise ValueError("Assigned read likelihood must be finite")
    allocation = count_term(counts)
    selected = assigned + (allocation if objective == "classification" else 0.)
    parameters = (len(counts)-1)*(int(regions)+1)
    penalty = parameters*math.log(float(counts.sum()))
    return {"cluster_weights": counts/counts.sum(),
            "selection_objective": objective,
            "score_definition": score_definition_for(objective),
            "selection_log_likelihood": selected,
            "conditional_log_likelihood": assigned,
            "classification_log_likelihood": assigned+allocation,
            "classification_count_term": allocation,
            "complexity_penalty": penalty,
            "num_parameters": parameters,
            "score": -2*selected+penalty}
