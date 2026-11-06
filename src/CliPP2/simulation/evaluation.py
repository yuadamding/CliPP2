"""Truth-side evaluation for the prespecified biological simulation study.

Inference never imports this module. All denominators, mappings and information
strata are derived from frozen original observations and physical-history truth.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
import math
from types import SimpleNamespace

import numpy as np
from scipy.special import logsumexp, xlogy, xlog1py
from scipy.optimize import linear_sum_assignment
from CliPP2.assignment import posterior_passes
from CliPP2.io.data import content_hash

FAMILIES = tuple(f"S{i:02}" for i in range(1, 13))
TRANSITIONS = ("correct_retained", "error_corrected", "error_persisted", "error_introduced",
               "correct_resolution_lost", "error_withheld")
UNMAPPED = "__unmapped__"


def json_safe(value):
    """Preserve infinite loss explicitly, without nonstandard JSON numbers."""
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
    return value


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _choose2(values):
    values = np.asarray(values, dtype=np.int64)
    return int(np.sum(values * (values - 1) // 2))


def partition_scores(truth, predicted):
    """ARI and pair precision/recall; empty pair denominators remain unavailable."""
    truth, predicted = list(truth), list(predicted)
    if len(truth) != len(predicted):
        raise ValueError("Partition axes differ")
    n = len(truth)
    if not n:
        return dict(n=0, ari=None, pair_precision=None, pair_recall=None, group_count_error=None)
    cells = Counter(zip(truth, predicted))
    t, p = Counter(truth), Counter(predicted)
    tp, actual, called = _choose2(list(cells.values())), _choose2(list(t.values())), _choose2(list(p.values()))
    pairs = n*(n-1)/2
    expected = actual*called/pairs if pairs else 0.
    maximum = (actual+called)/2
    ari = (tp-expected)/(maximum-expected) if maximum != expected else 1.
    return dict(n=n, ari=float(ari), pair_precision=_ratio(tp, called), pair_recall=_ratio(tp, actual),
                group_count_error=len(p)-len(t), true_groups=len(t), occupied_groups=len(p))


def transition(baseline_correct, accepted_correct):
    if accepted_correct is None:
        return "correct_resolution_lost" if baseline_correct else "error_withheld"
    if accepted_correct:
        return "correct_retained" if baseline_correct else "error_corrected"
    return "error_introduced" if baseline_correct else "error_persisted"


def candidate_correspondence(labels, clonal_label, source_members, truth_ids, clonal_truth_id, *, purity_min=.8):
    """Label calibration by frozen source overlap, not reference-center recovery."""
    mapping, audit = {}, []
    for label in labels:
        members = tuple(source_members.get(label, ()))
        counts = Counter(truth_ids[m] for m in members if m in truth_ids)
        selected = min(counts, key=lambda key: (-counts[key], str(key))) if counts else None
        purity = counts[selected]/sum(counts.values()) if counts else None
        mapped = clonal_truth_id if label == clonal_label else selected if purity is not None and purity >= purity_min else None
        mapping[label] = mapped
        audit.append(dict(candidate_label=label, truth_class=mapped, source_count=len(members),
                          source_truth_count=sum(counts.values()), source_purity=purity,
                          mapping_purpose="frozen_source_member_purity_detailed_membership_not_center_recovery",
                          mapping_status="mapped" if mapped is not None else "unmapped_or_ambiguous",
                          designated_clonal=label == clonal_label, mapping_threshold=purity_min))
    return mapping, audit


def normalize_evaluation_config(config=None):
    """Validate reporting choices without changing any fitted decision or threshold."""
    config = dict(config or {})
    allowed = {"schema", "reference_match_atol", "reference_distance"}
    if set(config) - allowed:
        raise ValueError(f"Unknown evaluation configuration fields: {sorted(set(config)-allowed)}")
    if config.get("schema", "clipp2.simulation.evaluation-config.v1") != "clipp2.simulation.evaluation-config.v1":
        raise ValueError("Unsupported evaluation configuration schema")
    tolerance = config.get("reference_match_atol")
    if tolerance is not None:
        if isinstance(tolerance, (bool, str)) or not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Reference match tolerance must be finite and nonnegative or None")
        tolerance = float(tolerance)
    distance = config.get("reference_distance", "l2")
    if distance not in ("l2", "linf"):
        raise ValueError("Reference distance must be l2 or explicit legacy linf")
    return dict(schema="clipp2.simulation.evaluation-config.v1", reference_match_atol=tolerance,
                reference_distance=distance)


def evaluation_definition(config=None):
    config = normalize_evaluation_config(config)
    return dict(schema="clipp2.simulation.evaluation-definition.v2", config=config,
        binary_endpoint="accepted_designated_clonal_status_equals_physical_truth_clonality",
        binary_objective="unchanged_original_population_equal_family_history_acquisition",
        detailed_endpoint="frozen_source_member_purity_observable_truth_class",
        source_member_purity_min=.8, unmapped_acceptance="separate_Mc_Mw_U_not_automatic_error",
        geometric_endpoint="accepted_center_matches_truth_and_quality_passes_informative_regions",
        reference_distance=config["reference_distance"], reference_vector="full_requested_regions_unnormalized",
        reference_match_atol=config["reference_match_atol"],
        matching_policy="maximum_valid_cardinality_then_minimum_valid_distance_then_lexicographic_truth_reference",
        matching_ambiguity="alternative_equal_cardinality_exact_binary_rational_distance_sum_optima",
        availability_quality_regions="original_eligible_observed_positive_depth",
        uncertainty_bounds="mapping_error_lower_Mw_over_A_upper_Mw_plus_U_over_A_not_confidence_intervals")


def _integer_assignment(cost):
    """Rectangular Hungarian algorithm with exact Python-integer potentials."""
    n, k = len(cost), len(cost[0]) if cost else 0
    u, v, owner = [0]*(n+1), [0]*(k+1), [0]*(k+1)
    for row in range(1, n+1):
        owner[0], column = row, 0
        minima, previous, used = [None]*(k+1), [0]*(k+1), [False]*(k+1)
        while True:
            used[column] = True
            current, delta, next_column = owner[column], None, None
            for j in range(1, k+1):
                if not used[j]:
                    value = cost[current-1][j-1]-u[current]-v[j]
                    if minima[j] is None or value < minima[j]:
                        minima[j], previous[j] = value, column
                    if delta is None or minima[j] < delta:
                        delta, next_column = minima[j], j
            for j in range(k+1):
                if used[j]:
                    u[owner[j]] += delta
                    v[j] -= delta
                elif minima[j] is not None:
                    minima[j] -= delta
            column = next_column
            if owner[column] == 0:
                break
        while column:
            before = previous[column]
            owner[column] = owner[before]
            column = before
    return [(owner[j]-1, j-1) for j in range(1, k+1) if owner[j]]


def _qualified_matching(distance, valid):
    """Maximize cardinality, minimize exact distance sum, then break ties lexically.

    Represented float distances are binary rationals. Common-denominator integer
    costs avoid rounded Hungarian subproblems or an arbitrary tie tolerance.
    Unmatched dummy columns ensure invalid edges never influence valid costs.
    """
    n, k = distance.shape
    if not n or not k or not valid.any():
        return [], []
    ratios = {tuple(pair): float(distance[tuple(pair)]).as_integer_ratio() for pair in np.argwhere(valid)}
    denominator = max(den for _, den in ratios.values())
    weights = {pair: numerator*(denominator//den) for pair, (numerator, den) in ratios.items()}
    base, lexical_scale = k+1, (k+1)**n
    # The whole secondary objective is strictly smaller than one unmatched cost.
    unmatched = (n*max(weights.values())+1)*lexical_scale
    forbidden = (n+1)*unmatched
    costs = []
    for i in range(n):
        place = base**(n-1-i)
        costs.append([weights[i, j]*lexical_scale+j*place if (i, j) in weights else forbidden
                      for j in range(k)] + [unmatched+k*place]*n)

    def solve(blocked=None):
        current = [row[:] for row in costs] if blocked is not None else costs
        if blocked is not None:
            current[blocked[0]][blocked[1]] = forbidden
        pairs = sorted((i, j) for i, j in _integer_assignment(current) if j < k)
        objective = (len(pairs), sum(weights[i, j] for i, j in pairs))
        return pairs, objective

    pairs, objective = solve()
    ambiguity = []
    for i, j in pairs:
        alternative, value = solve((i, j))
        if value == objective:
            ambiguity.append(dict(truth_index=i, alternative_reference_indices=[j, dict(alternative).get(i)]))
    return pairs, ambiguity


def reference_center_diagnostics(labels, centers, truth_centers, *, match_atol=None, distance="l2"):
    """Continuous center errors and declared full-vector one-to-one recovery.

    New assessments use unnormalized L2. Explicit ``distance='linf'`` identifies
    legacy geometry; cached v1 results keep their original matching definition.
    """
    config = normalize_evaluation_config(dict(reference_match_atol=match_atol, reference_distance=distance))
    labels, truth_ids = tuple(labels), tuple(sorted(truth_centers, key=str))
    centers = np.asarray(centers, float)
    truth = np.asarray([truth_centers[key] for key in truth_ids], float)
    if len(labels) != len(centers) or len(set(labels)) != len(labels):
        raise ValueError("Reference centers must align with distinct labels")
    if (centers.size and (centers.ndim != 2 or not np.isfinite(centers).all())
            or truth.size and (truth.ndim != 2 or not np.isfinite(truth).all())
            or centers.size and truth.size and centers.shape[1] != truth.shape[1]):
        raise ValueError("Reference and truth centers require finite aligned regional vectors")
    order = sorted(range(len(labels)), key=lambda i: (type(labels[i]).__name__, str(labels[i])))
    labels = tuple(labels[i] for i in order)
    centers = centers[order]
    paired, nearest, qualified, ambiguity = [], [], [], []
    if labels and truth_ids:
        delta = truth[:, None, :] - centers[None, :, :]
        l2, linf = np.linalg.norm(delta, axis=2), np.max(np.abs(delta), axis=2)

        def pair(i, j):
            return dict(reference_label=labels[j], truth_class=truth_ids[i], l2_error=float(l2[i, j]),
                        max_absolute_error=float(linf[i, j]))

        rows, columns = linear_sum_assignment(l2)
        paired = [pair(i, j) for i, j in zip(rows, columns)]
        nearest = [pair(i, int(np.argmin(l2[i]))) for i in range(len(truth_ids))]
        if match_atol is not None:
            distances = l2 if distance == "l2" else linf
            matches, alternatives = _qualified_matching(distances, distances <= match_atol)
            qualified = [pair(i, j) for i, j in matches]
            ambiguity = [dict(truth_class=truth_ids[row["truth_index"]],
                alternative_reference_labels=[labels[j] if j is not None else None
                                              for j in row["alternative_reference_indices"]]) for row in alternatives]
    return dict(status="assessed" if match_atol is not None else "unassessed_no_declared_tolerance",
        reference_match_atol=config["reference_match_atol"], reference_distance=distance,
        tolerance_metric="full_vector_unnormalized_l2" if distance == "l2" else "maximum_absolute_regional_error",
        matching_policy=evaluation_definition(config)["matching_policy"],
        matching_ambiguous=bool(ambiguity) if match_atol is not None else None, matching_ambiguity=ambiguity,
        reference_count=len(labels), truth_group_count=len(truth_ids),
        one_to_one_pairs=paired, nearest_reference_by_truth=nearest,
        mean_one_to_one_l2_error=float(np.mean([row["l2_error"] for row in paired])) if paired else None,
        mean_nearest_l2_error=float(np.mean([row["l2_error"] for row in nearest])) if nearest else None,
        recovered_count=len(qualified) if match_atol is not None else None,
        reference_recall=_ratio(len(qualified), len(truth_ids)) if match_atol is not None else None,
        qualified_pairs=qualified if match_atol is not None else None)


def target_reference_availability(true_center, observed, reference, *, match_atol=None, distance="l2", accepted_label=None):
    """Full-vector geometry; only quality is restricted by target information."""
    result = dict(geometric_reference_available=None, usable_reference_available=None,
        geometric_correct_reference_acceptance=None, geometric_matching_labels=None, usable_matching_labels=None)
    if match_atol is None:
        return dict(result, geometric_availability_status="unassessed_no_declared_tolerance",
                    usable_availability_status="unassessed_no_declared_tolerance")
    if not reference:
        return dict(result, geometric_availability_status="unavailable_reference",
                    usable_availability_status="unavailable_reference")
    true_center, observed = np.asarray(true_center, float), np.asarray(observed, bool)
    labels = tuple(reference["labels"])
    centers = np.asarray(reference["centers"], float).reshape(len(labels), len(true_center))
    if not np.isfinite(true_center).all():
        return dict(result, geometric_availability_status="unavailable_truth",
                    usable_availability_status="unavailable_truth")
    delta = centers - true_center
    distances = np.linalg.norm(delta, axis=1) if distance == "l2" else np.max(np.abs(delta), axis=1)
    matches = distances <= match_atol
    result.update(geometric_reference_available=bool(matches.any()),
        geometric_matching_labels=[label for label, keep in zip(labels, matches) if keep],
        geometric_availability_status="assessed")
    if not observed.any():
        return dict(result, usable_availability_status="not_evaluable_no_informative_observation")
    quality = reference.get("regional_quality")
    if quality is None:
        return dict(result, usable_availability_status="unavailable_reference_quality")
    quality = np.asarray(quality)
    if quality.shape != centers.shape or quality.dtype.kind != "b":
        raise ValueError("Reference quality must be explicit boolean candidate-by-region flags")
    usable = matches & quality[:, observed].all(axis=1)
    result.update(usable_reference_available=bool(usable.any()),
        usable_matching_labels=[label for label, keep in zip(labels, usable) if keep],
        usable_availability_status="assessed",
        geometric_correct_reference_acceptance=bool(accepted_label in {label for label, keep in zip(labels, usable) if keep}))
    return result


def detailed_membership_status(accepted_label, truth_class, mapping):
    if accepted_label is None:
        return "unresolved"
    mapped = mapping.get(accepted_label)
    return "U" if mapped is None else "Mc" if mapped == truth_class else "Mw"


def detailed_membership_summary(rows):
    """Mapped error bounds describe unresolved mapping, not confidence intervals."""
    rows = [row for row in rows if row["evaluable"]]
    counts = Counter(row["detailed_membership_status"] for row in rows)
    correct, wrong, unmapped = (counts[key] for key in ("Mc", "Mw", "U"))
    accepted, population = correct+wrong+unmapped, len(rows)
    return dict(population=population, accepted=accepted, Mc=correct, Mw=wrong, U=unmapped,
        unresolved=counts["unresolved"], mapped_correct_resolution=_ratio(correct, population),
        mapped_wrong_resolution=_ratio(wrong, population), mapped_selective_error=_ratio(wrong, correct+wrong),
        unmapped_acceptance_fraction=_ratio(unmapped, accepted), unmapped_resolution=_ratio(unmapped, population),
        detailed_selective_error_lower_bound=_ratio(wrong, accepted),
        detailed_selective_error_upper_bound=_ratio(wrong+unmapped, accepted))


def mapped_probabilities(probabilities, labels, mapping, truth_classes):
    result = dict.fromkeys((*truth_classes, UNMAPPED), 0.)
    for label, probability in zip(labels, probabilities):
        result[mapping.get(label) or UNMAPPED] += float(probability)
    if not np.isclose(sum(result.values()), 1., rtol=0, atol=1e-12):
        raise ValueError("Candidate mapping lost probability mass")
    return result


def probability_scores(probability, truth, *, truth_class=None, true_log_probability=None):
    if isinstance(probability, Mapping):
        true_p = float(probability.get(truth_class, 0.))
        brier = sum((value - (key == truth_class))**2 for key, value in probability.items())
        if truth_class not in probability:
            brier += 1.
    else:
        probability = float(probability)
        true_p = probability if truth else 1-probability
        brier = (probability-float(truth))**2
    loss = (-float(true_log_probability) if true_log_probability is not None else
            float(-np.log(true_p)) if true_p > 0 else float("inf"))
    return dict(brier=float(brier), log_loss=loss, true_probability=true_p,
                infinite_log_loss=bool(np.isposinf(loss)))


def log_probability_mass(scores, indices):
    """Probability of a candidate subset, retaining finite tiny truth mass."""
    scores, indices = np.asarray(scores, float), list(indices)
    if not indices or not np.isfinite(scores[indices]).any():
        return -float("inf")
    winner = int(np.argmax(scores))
    norm = np.log1p(np.exp(np.delete(scores, winner)-scores[winner]).sum())
    if winner in indices:
        chosen = [index for index in indices if index != winner]
        return float(np.log1p(np.exp(scores[chosen]-scores[winner]).sum())-norm)
    return float(logsumexp(scores[indices])-scores[winner]-norm)


def reliability(probabilities, outcomes, *, bins=10):
    probabilities, outcomes = np.asarray(probabilities, float), np.asarray(outcomes, float)
    if probabilities.shape != outcomes.shape or not np.isfinite(probabilities).all():
        raise ValueError("Reliability requires aligned finite probabilities")
    indices = np.minimum((probabilities*bins).astype(int), bins-1)
    return [dict(bin=index, lower=index/bins, upper=(index+1)/bins,
                 count=int((indices == index).sum()),
                 mean_probability=float(probabilities[indices == index].mean()) if np.any(indices == index) else None,
                 observed_frequency=float(outcomes[indices == index].mean()) if np.any(indices == index) else None)
            for index in range(bins)]


def _bernoulli_kl(p, q):
    # Endpoint-safe KL: 0*log(0) is zero, a genuinely impossible q gives infinity.
    return xlogy(p, p) + xlog1py(1-p, -p) - xlogy(p, q) - xlog1py(1-p, -q)


def information_diagnostics(phi, multiplicity, major, minor, purity, normal_cn, depth, observed,
                            truth_ids, *, candidates=None, support=None):
    """Complete-truth component expected KL; this is not a marginal fit score.

    The primary call supplies true CN/purity, original support and all observable
    classes. Alternate candidates/support are explicitly separate diagnostics.
    """
    phi, multiplicity, major, minor, purity, normal_cn, depth, observed = np.broadcast_arrays(
        phi, multiplicity, major, minor, purity, normal_cn, depth, observed)
    classes = tuple(sorted(set(truth_ids)))
    if candidates is None:
        candidates = {key: phi[list(truth_ids).index(key)] for key in classes}
    upper = major if support is None else np.broadcast_to(support, major.shape)
    scaling = purity / ((1-purity)*normal_cn + purity*(major+minor))
    theta = scaling*np.where(phi > 0, multiplicity, 0.)*phi
    result = []
    for i, own in enumerate(truth_ids):
        use = np.asarray(observed[i], bool) & (depth[i] > 0)
        alternatives = []
        for label, center in candidates.items():
            if label == own:
                continue
            distance, exact = 0., True
            for r in np.flatnonzero(use):
                q = np.minimum(scaling[i, r]*np.arange(1, int(upper[i, r])+1)*center[r], 1.)
                divergence = np.maximum(0., _bernoulli_kl(theta[i, r], q))
                distance += depth[i, r]*float(np.min(divergence))
                exact &= bool(np.min(np.abs(theta[i, r]-q)) <= 1e-12)
            alternatives.append((float(distance), str(label), exact))
        if not use.any():
            result.append(dict(information_stratum="no_information", expected_kl=None, competing_truth_class=None,
                               exact_alias=None, informative_regions=0))
        elif not alternatives:
            result.append(dict(information_stratum="not_applicable_no_competing_truth", expected_kl=None,
                               competing_truth_class=None, exact_alias=None, informative_regions=int(use.sum())))
        else:
            value, label, exact = min(alternatives, key=lambda row: (row[0], row[1]))
            result.append(dict(information_stratum="exact_alias" if exact else "weak" if value <= 2 else "separated",
                               expected_kl=value, competing_truth_class=label, exact_alias=exact,
                               informative_regions=int(use.sum())))
    return result


def cap_audit(history, catalog, remainder_ids, contributors, upper, *, scope="selected_qualified"):
    """Replace calls only for the exact selected contributor IDs, without requalification."""
    rows = {m: i for i, m in enumerate(history.mutation_ids)}
    regions = {r: j for j, r in enumerate(history.region_ids)}
    qualified = defaultdict(list)
    for row in contributors:
        if row.get("usable", row.get("qualified", False)):
            qualified[row["region_id"], row["segment_id"]].append(row)
    upper = np.asarray(upper)
    if upper.shape != np.asarray(history.phi).shape:
        raise ValueError("Cap audit requires bounds on the original mutation/region axes")
    result = []
    for mutation in remainder_ids:
        i = rows[mutation]
        for region, j in regions.items():
            if not catalog.eligible[i, j] or not history.phi[i, j] > 0 or not np.isfinite(history.multiplicity[i, j]):
                continue
            selected = qualified[region, str(catalog.segment_ids[i, j])]
            replacements = [float(history.multiplicity[rows[row["mutation_id"]], j]) for row in selected]
            valid = all(np.isfinite(value) and value >= 1 and history.phi[rows[row["mutation_id"]], j] > 0
                        for value, row in zip(replacements, selected))
            true_upper = min(replacements) if selected and valid else catalog.major_cn[i, j] if not selected else None
            truth, original = float(history.multiplicity[i, j]), float(catalog.major_cn[i, j])
            inherited = truth > original
            result.append(dict(mutation_id=mutation, region_id=region, segment_id=str(catalog.segment_ids[i, j]),
                scope=scope, true_clonal=bool(np.all(np.isclose(history.phi[i], 1., rtol=0, atol=1e-12))),
                true_multiplicity=truth, original_upper=original, estimated_upper=float(upper[i, j]),
                observed=bool(catalog.count_observed[i, j] and catalog.total_counts[i, j] > 0),
                cap_active=bool(upper[i, j] < original), inherited_exclusion=inherited,
                excluded=bool(truth > upper[i, j]), additional_exclusion=bool(not inherited and truth > upper[i, j]),
                contributor_count=len(selected), contributor_ids=[row["mutation_id"] for row in selected],
                fallback=not selected, truth_replacement_evaluable=valid, truth_replacement_upper=true_upper,
                truth_replacement_exclusion=bool(truth > true_upper) if true_upper is not None else None,
                harmful_minimum_call_error=bool(truth > upper[i, j] and truth <= true_upper) if true_upper is not None else None))
    return result


def _fit_view(fit):
    if not fit:
        return {}, {}, {}
    labels = {m: int(label) for m, label in zip(fit["mutation_ids"], fit["labels"])}
    centers = np.asarray(fit["centers"], float)
    estimates = {m: centers[label] for m, label in labels.items()}
    calls = {m: np.asarray(call, float) for m, call in zip(fit["mutation_ids"], fit["multiplicities"])}
    return labels, estimates, calls


def _truth_clonal(history):
    if hasattr(history, "mutant_copy_counts") and hasattr(history, "clone_fractions"):
        return np.all((np.asarray(history.mutant_copy_counts)[:, :, None] > 0)
                      | (np.asarray(history.clone_fractions)[None, :, :] == 0), axis=(1, 2))
    # Equality tolerance only accommodates floating summation of physical fractions.
    return np.all(np.isclose(history.phi, 1., rtol=0, atol=1e-12), axis=1)


def evaluate(history, catalog, result, *, acquisition=None, reference_match_atol=None, evaluation_config=None):
    """Evaluate one paired A/B/C/B3/C3 cell; return only JSON-safe data.

    ``result`` is the serializable paired payload from ``experiments.run_paired``.
    Original mutation and region axes must match; unavailable arms retain zero
    correct-decision credit on the original evaluable population, while scientific
    conditional metrics remain unavailable.
    """
    if tuple(history.mutation_ids) != tuple(catalog.mutation_ids) or tuple(history.region_ids) != tuple(catalog.region_ids):
        raise ValueError("Evaluation requires the complete aligned original universe")
    config_input = dict(evaluation_config or {})
    if reference_match_atol is not None:
        if "reference_match_atol" in config_input and config_input["reference_match_atol"] != reference_match_atol:
            raise ValueError("Conflicting reference match tolerances")
        config_input["reference_match_atol"] = reference_match_atol
    evaluation_config = normalize_evaluation_config(config_input)
    definition = evaluation_definition(evaluation_config)
    reference_match_atol = evaluation_config["reference_match_atol"]
    reference_distance = evaluation_config["reference_distance"]
    acquisition = dict(acquisition or {})
    metadata = dict(history.metadata)
    identity = dict(family=metadata.get("family", getattr(history, "family", "unknown")),
                    protocol_version=result.get("protocol_version"),
                    run_manifest_id=result.get("run_manifest_id"),
                    source_cohort_protocol_version=result.get("source_cohort_protocol_version"),
                    history_id=metadata.get("history_id", getattr(history, "history_id", "unknown")),
                    acquisition_id=acquisition.get("id", acquisition.get("acquisition_id", "unspecified")),
                    stratum=result.get("stratum", "estimated"))
    ids, regions = tuple(history.mutation_ids), tuple(history.region_ids)
    index = {mutation: i for i, mutation in enumerate(ids)}
    truth_ids = dict(zip(ids, history.observable_truth_ids))
    classes = tuple(sorted(set(history.observable_truth_ids)))
    clonal = dict(zip(ids, map(bool, _truth_clonal(history))))
    clonal_classes = {truth_ids[m] for m in ids if clonal[m]}
    if len(clonal_classes) > 1:
        raise ValueError("Observable truth assigned multiple IDs to the same clonal vector")
    clonal_class = next(iter(clonal_classes), None)
    observed = np.asarray(catalog.eligible & catalog.count_observed & (catalog.total_counts > 0), bool)
    known = np.isfinite(history.phi).all(axis=1)
    evaluable = known & observed.any(axis=1)
    original_n = int(evaluable.sum())
    metrics, mutation_rows, correspondence, calibration_rows, applicability, containment = [], [], [], [], [], []
    reference_geometry, detailed_membership = {}, {}
    truth_centers = {truth_ids[m]: history.phi[index[m]] for m in ids if evaluable[index[m]]}

    def metric(name, arm, numerator=None, denominator=None, *, value=None, view="", stratum="all", reason=None):
        if reason in ("unavailable_analysis", "unavailable_geometric_acceptance"):
            value = None
        elif denominator is not None:
            value = _ratio(numerator, denominator)
        metrics.append(dict(name=name, arm=arm, view=view, stratum=stratum, value=value,
            numerator=numerator, denominator=denominator,
            status=reason or ("available" if value is not None else "not_applicable_zero_denominator")))

    def partition(arm, labels, selected, view):
        selected = [m for m in selected if m in labels and known[index[m]]]
        scores = partition_scores([truth_ids[m] for m in selected], [labels[m] for m in selected])
        for key in ("ari", "pair_precision", "pair_recall", "group_count_error"):
            metric(key, arm, value=scores[key], view=view)
        metric("partition_coverage", arm, len(selected), len(ids), view=view)
        metric("occupied_groups", arm, value=scores.get("occupied_groups"), view=view)

    def ccf(arm, estimates, view, selected=ids):
        differences = [np.square(np.asarray(estimates[m], float)-history.phi[index[m]])
                       for m in selected if m in estimates and known[index[m]]]
        values = np.asarray(differences)
        use = np.isfinite(values)
        metric("ccf_rmse", arm, value=float(np.sqrt(values[use].mean())) if use.any() else None, view=view)
        metric("ccf_estimate_coverage", arm, int(use.sum()), len(selected)*len(regions), view=view)

    def multiplicity(arm, calls, view, qualified=None):
        valid, exact, errors, qualified_count, qualified_correct = 0, 0, [], 0, 0
        denominator = 0
        for m in ids:
            i = index[m]
            use = observed[i] & (history.phi[i] > 0) & np.isfinite(history.multiplicity[i])
            denominator += int(use.sum())
            values = np.asarray(calls.get(m, np.full(len(regions), np.nan)), float)
            finite = use & np.isfinite(values)
            valid += int(finite.sum())
            exact += int(np.sum(values[finite] == history.multiplicity[i, finite]))
            errors.extend(np.abs(values[finite]-history.multiplicity[i, finite]).tolist())
            if qualified is not None:
                q = finite & np.array([(m, r) in qualified for r in regions])
                qualified_count += int(q.sum())
                qualified_correct += int(np.sum(values[q] == history.multiplicity[i, q]))
        metric("multiplicity_accuracy", arm, exact, valid, view=view)
        metric("multiplicity_mae", arm, value=float(np.mean(errors)) if errors else None, view=view)
        metric("multiplicity_call_coverage", arm, valid, denominator, view=view)
        if qualified is not None:
            metric("qualified_call_coverage", arm, qualified_count, denominator, view=view)
            metric("qualified_call_correctness", arm, qualified_correct, qualified_count, view=view)

    first_labels, first_centers, first_calls = _fit_view(result.get("stage_one"))
    g0 = set(result.get("g0_ids", ()))
    contributors = result.get("contributors", ())
    qualified = {(row["mutation_id"], row["region_id"]) for row in contributors if row.get("usable", False)}
    metric("original_evaluable", "all", value=original_n)
    metric("true_clonal_availability", "A", value=int(any(clonal[m] and evaluable[index[m]] for m in ids)))
    available_clonal = any(clonal[m] and evaluable[index[m]] for m in ids)
    metric("available_reference_not_recognized", "A", value=int(available_clonal and not g0)
           if result.get("stage_one") else None, reason=None if result.get("stage_one") else "unavailable_analysis")
    metric("appropriate_no_reference", "A", value=int(not available_clonal and not g0)
           if result.get("stage_one") else None, reason=None if result.get("stage_one") else "unavailable_analysis")
    metric("selected_reference_purity", "A", sum(clonal[m] for m in g0), len(g0))
    metric("true_clonal_outside_reference", "A", sum(clonal[m] and m not in g0 for m in ids), sum(clonal.values()))
    clonal_segments = {(r, str(catalog.segment_ids[index[m], j])) for m in ids if clonal[m] for j, r in enumerate(regions)}
    selected_segments = {(r, str(catalog.segment_ids[index[m], j])) for m in g0 for j, r in enumerate(regions)}
    metric("reference_segment_coverage", "A", len(clonal_segments & selected_segments), len(clonal_segments))
    true_classes = {(r, str(catalog.segment_ids[index[m], j]), float(history.multiplicity[index[m], j]))
                    for m in ids if clonal[m] for j, r in enumerate(regions)}
    source_classes = {(r, str(catalog.segment_ids[index[m], j]), float(history.multiplicity[index[m], j]))
                      for m in g0 for j, r in enumerate(regions) if history.phi[index[m], j] > 0}
    metric("reference_segment_multiplicity_coverage", "A", len(true_classes & source_classes), len(true_classes))
    contributor_truth = []
    for row in contributors:
        i, j = index[row["mutation_id"]], regions.index(row["region_id"])
        if history.phi[i, j] > 0 and np.isfinite(history.multiplicity[i, j]) and observed[i, j]:
            contributor_truth.append((bool(row.get("usable", False)),
                                      float(row.get("native_call", np.nan)) == history.multiplicity[i, j]))
    metric("contributor_qualification_sensitivity", "A", sum(q and correct for q, correct in contributor_truth),
           sum(correct for _, correct in contributor_truth))
    metric("accepted_contributor_accuracy", "A", sum(q and correct for q, correct in contributor_truth),
           sum(q for q, _ in contributor_truth))
    partition("A", first_labels, ids, "native")
    ccf("A", first_centers, "native_source")
    multiplicity("A", first_calls, "native_source", qualified)
    stage_one_available = bool(result.get("stage_one"))
    applicability.append(dict(arm="A", available=stage_one_available, status=result.get("status", "complete")
                             if not stage_one_available else "complete", planned=True))
    arms = result.get("arms", {})
    native_views, combined_views, final_views, arm_states = {}, {}, {}, {}
    for arm in ("B", "C"):
        payload = arms.get(arm, {})
        second_labels, second_centers, second_calls = _fit_view(payload.get("stage_two"))
        native_views[arm] = second_labels
        combined = {m: int(label) for m, label in payload.get("combined_labels", {}).items()}
        combined_views[arm] = combined
        reference = payload.get("reference") or {}
        labels = tuple(reference.get("labels", ()))
        centers = np.asarray(reference.get("centers", []), float)
        geometry = reference_center_diagnostics(labels, centers, truth_centers,
            match_atol=reference_match_atol, distance=reference_distance)
        if not reference:
            geometry.update(status="unavailable_reference", recovered_count=None, reference_recall=None)
        reference_geometry[arm] = geometry
        for name in ("mean_one_to_one_l2_error", "mean_nearest_l2_error", "reference_recall"):
            metric(name, arm, value=geometry[name], view="reference_center_geometry",
                   reason=geometry["status"] if name == "reference_recall" and geometry["status"] != "assessed" else None)
        center_lookup = dict(zip(labels, centers))
        clonal_label = reference.get("clonal_label", 0)
        targets, non_targets = set(payload.get("targets", ())), set(payload.get("non_targets", ()))
        if combined and (targets & non_targets or targets | non_targets != set(combined)):
            raise ValueError("Targets and unchanged members must partition accepted combined membership")
        status = payload.get("status", "unavailable")
        available = bool(reference) and bool(combined)
        if status == "complete" and not reference:
            raise ValueError("Complete arm has no frozen reference")
        applicability.append(dict(arm=arm, available=available, status=status, planned=True))
        partition(arm, second_labels, result.get("remainder_ids", ()), "native_remainder")
        partition(arm, combined, ids, "combined")
        native_centers = {m: np.asarray(value, float) for m, value in payload.get("native_centers", {}).items()}
        if not native_centers:
            native_centers = {m: first_centers[m] for m in g0 if m in first_centers}
            native_centers.update(second_centers)
        ccf(arm, native_centers, "native_source")
        ccf(arm, {m: center_lookup[label] for m, label in combined.items()}, "constructed_reference")
        calls = {m: first_calls[m] for m in g0 if m in first_calls}
        calls.update(second_calls)
        multiplicity(arm, calls, "native_source")
        newly_merged = targets-g0
        near = {m for m, center in second_centers.items()
                if np.linalg.norm(center - 1) < result["config"]["clonal_distance_max"]}
        metric("geometric_near_clonal_routing_coverage", arm, len(near & targets), len(near))
        remaining_true_nonclonal = {m for m in second_labels if not clonal[m]}
        remaining_true_clonal = {m for m in second_labels if clonal[m]}
        false_merged = {m for m in newly_merged if not clonal[m]}
        metric("false_merge_fraction", arm, len(false_merged), len(newly_merged))
        metric("false_merge_nonclonal_rate", arm, len(false_merged), len(remaining_true_nonclonal))
        metric("unmerged_true_clonal_fraction", arm, len(remaining_true_clonal-newly_merged), len(remaining_true_clonal))
        groups = payload.get("group_records", ())
        for classification in ("clonal_merge_eligible", "nonclonal", "unresolved"):
            metric("group_classification_fraction", arm,
                   sum(row.get("classification") == classification for row in groups), len(groups), stratum=classification)
        sources = {label: [m for m, value in combined.items() if value == label] for label in labels}
        mapping, map_records = candidate_correspondence(labels, clonal_label, sources, truth_ids, clonal_class)
        for row in map_records:
            row["source_classifications"] = sorted({source.get("classification", "unavailable")
                for source in reference.get("lineage", ()) if source.get("baseline_reference_label") == row["candidate_label"]})
        correspondence.extend(dict(arm=arm, **row) for row in map_records)
        baseline_correct = {m: (label == clonal_label) == clonal[m] for m, label in combined.items()}
        baseline_multi = {m: mapping.get(label) == truth_ids[m] for m, label in combined.items()}
        metric("correct_per_original_evaluable", arm,
               sum(baseline_correct.get(m, False) and evaluable[index[m]] for m in ids), original_n,
               reason=None if available else "unavailable_no_correct_decision_credit")
        assignment = payload.get("assignment")
        records = {row["mutation_id"]: row for row in assignment.get("assignments", ())} if assignment else {}
        stage = arm+"3"
        assignment_available = assignment is not None
        applicability.append(dict(arm=stage, available=assignment_available,
            status=payload.get("assignment_status", "complete" if assignment_available else status), planned=True))
        if assignment_available and set(records) != targets:
            raise ValueError("Assignment targets differ from the frozen target population")
        arm_mutation_start = len(mutation_rows)
        for row in payload.get("original_rows", ()):
            if row["mutation_id"] in non_targets and row.get("final_assignment_label") != combined[row["mutation_id"]]:
                raise ValueError("Reported Stage 3 output changed an unchanged non-target membership")
        final = {m: label for m, label in combined.items() if m in non_targets}
        accepted_estimates, conditional_calls, assigned_calls = {}, {}, {}
        finite_probability_count, accepted_count, correct_count, wrong_count = 0, 0, 0, 0
        mapped_correct, mapped_accepted, unmapped_accepted, absent_accepted = 0, 0, 0, 0
        transitions, multi_transitions = Counter(), Counter()
        target_evaluable = {m for m in targets if evaluable[index[m]]}
        for mutation in sorted(targets):
            record = records.get(mutation)
            accepted = record.get("accepted_label") if record else None
            is_eval = mutation in target_evaluable
            correct = (accepted == clonal_label) == clonal[mutation] if accepted is not None else None
            multi_correct = mapping.get(accepted) == truth_ids[mutation] if accepted is not None else None
            if accepted is not None:
                if accepted not in center_lookup:
                    raise ValueError("Accepted target label is outside the fixed reference")
                final[mutation] = accepted
                accepted_estimates[mutation] = center_lookup[accepted]
            if is_eval and assignment_available:
                accepted_count += accepted is not None
                correct_count += correct is True
                wrong_count += correct is False
                transitions[transition(baseline_correct[mutation], correct)] += 1
                multi_transitions[transition(baseline_multi[mutation], multi_correct)] += 1
                mapped_accepted += accepted is not None
                mapped_correct += multi_correct is True
                unmapped_accepted += accepted is not None and mapping.get(accepted) is None
                absent_accepted += accepted is not None and truth_ids[mutation] not in set(mapping.values())
            probabilities = np.asarray(record.get("probabilities", []), float) if record else np.array([])
            probability_defined = probabilities.shape == (len(labels),) and bool(labels) and np.isfinite(probabilities).all()
            if is_eval and probability_defined:
                finite_probability_count += 1
                mapped = mapped_probabilities(probabilities, labels, mapping, classes)
                q = float(probabilities[labels.index(clonal_label)])
                scores = np.asarray(record["log_scores"], float)
                binary_indices = [i for i, label in enumerate(labels) if (label == clonal_label) == clonal[mutation]]
                multiclass_indices = [i for i, label in enumerate(labels) if mapping.get(label) == truth_ids[mutation]]
                binary = probability_scores(q, clonal[mutation], true_log_probability=log_probability_mass(scores, binary_indices))
                multiclass = probability_scores(mapped, None, truth_class=truth_ids[mutation],
                                                 true_log_probability=log_probability_mass(scores, multiclass_indices))
                calibration_rows.append(dict(arm=stage, mutation_id=mutation, source_stage=record.get("source_stage"),
                    K=len(labels), truth_clonal=clonal[mutation], clonal_probability=q,
                    biological_clonal_mass=mapped.get(clonal_class, 0.) if clonal_class else 0.,
                    truth_class=truth_ids[mutation], mapped_probabilities=mapped, unmapped_mass=mapped[UNMAPPED],
                    true_class_present=truth_ids[mutation] in set(mapping.values()),
                    binary_brier=binary["brier"], binary_log_loss=binary["log_loss"],
                    multiclass_brier=multiclass["brier"], multiclass_log_loss=multiclass["log_loss"],
                    confidence=float(np.max(probabilities)), confidence_correct=bool(mapping.get(labels[int(np.argmax(probabilities))]) == truth_ids[mutation]),
                    best_log_probability=float(record["best_log_probability"]),
                    log_competitor_ratio=float(record["log_competitor_ratio"]),
                    has_finite_alternative=bool(record["has_finite_alternative"]),
                    best_clonal_correct=bool((labels[int(np.argmax(probabilities))] == clonal_label) == clonal[mutation]),
                    nonconfidence_gates_pass=bool(not record.get("cluster_tie", True)
                        and record.get("reference_quality_pass", False)
                        and record.get("fit_screen_status") == "pass"),
                    mean_depth=float(catalog.total_counts[index[mutation]][observed[index[mutation]]].mean()),
                    major_cn=float(np.max(catalog.major_cn[index[mutation]]))))
            mutation_rows.append(dict(arm=stage, mutation_id=mutation, evaluable=is_eval, truth_clonal=clonal[mutation],
                truth_class=truth_ids[mutation], role="target", baseline_reference_label=combined.get(mutation),
                baseline_correct=baseline_correct.get(mutation), accepted_label=accepted, correct=correct,
                multiclass_correct=multi_correct, status=record.get("status") if record else "unavailable_analysis",
                transition=transition(baseline_correct[mutation], correct) if is_eval and assignment_available else None,
                source_stage=record.get("source_stage") if record else None,
                target_reason=record.get("target_reason") if record else
                    "stage1_clonal_recheck" if mutation in g0 else "stage2_near_clonal_recheck",
                reference_quality_pass=record.get("reference_quality_pass") if record else None,
                historical_cap_contributor=any((mutation, region) in qualified for region in regions),
                historical_cap_contributor_reassigned=bool(accepted is not None and accepted != clonal_label
                    and any((mutation, region) in qualified for region in regions)),
                final_assignment_label=accepted, probability_available=probability_defined))
        for mutation in sorted(non_targets):
            mutation_rows.append(dict(arm=stage, mutation_id=mutation, evaluable=bool(evaluable[index[mutation]]),
                truth_clonal=clonal[mutation], truth_class=truth_ids[mutation], role="unchanged_non_target",
                baseline_reference_label=combined[mutation], final_assignment_label=final[mutation],
                accepted_label=final[mutation], baseline_correct=baseline_correct[mutation],
                correct=baseline_correct[mutation], multiclass_correct=baseline_multi[mutation],
                status="unchanged_non_target", probability_available=False))
        for mutation in sorted(set(ids)-set(combined)):
            mutation_rows.append(dict(arm=stage, mutation_id=mutation, evaluable=bool(evaluable[index[mutation]]),
                truth_clonal=clonal[mutation], truth_class=truth_ids[mutation], role="outside_combined",
                baseline_reference_label=None, final_assignment_label=None, accepted_label=None, baseline_correct=None, correct=None,
                status="not_in_accepted_combined_population", probability_available=False))
        arm_rows = mutation_rows[arm_mutation_start:]
        for row in arm_rows:
            mutation = row["mutation_id"]
            row["binary_clonality_correct"] = row.get("correct")
            row["detailed_membership_status"] = detailed_membership_status(
                row["accepted_label"], truth_ids[mutation], mapping)
            row["detailed_membership_correct"] = {"Mc": True, "Mw": False}.get(row["detailed_membership_status"])
            row.update(target_reference_availability(history.phi[index[mutation]], observed[index[mutation]],
                reference, match_atol=reference_match_atol, distance=reference_distance,
                accepted_label=row["accepted_label"]))
            if row["role"] == "target" and not assignment_available:
                row["geometric_correct_reference_acceptance"] = None
        populations = {
            "targets": [row for row in arm_rows if row["role"] == "target"],
            "unchanged_F": [row for row in arm_rows if row["role"] == "unchanged_non_target"],
            "whole_original": arm_rows,
            **{origin: [row for row in arm_rows if row.get("target_reason") == origin]
               for origin in ("stage1_clonal_recheck", "stage2_near_clonal_recheck")}}
        detailed_membership[stage] = {}
        for population_name, rows in populations.items():
            summary = detailed_membership_summary(rows)
            summary["status"] = ("unavailable_analysis" if population_name != "unchanged_F"
                and not assignment_available else "available")
            detailed_membership[stage][population_name] = summary
            view = "" if population_name == "targets" else population_name
            for name in ("mapped_correct_resolution", "mapped_wrong_resolution", "mapped_selective_error",
                    "unmapped_acceptance_fraction", "unmapped_resolution", "detailed_selective_error_lower_bound",
                    "detailed_selective_error_upper_bound"):
                metric(name, stage, value=summary[name], view=view,
                    reason="unavailable_analysis" if summary["status"] == "unavailable_analysis" else None)
            evaluable_rows = [row for row in rows if row["evaluable"]]
            for kind in ("geometric", "usable"):
                field = kind+"_reference_available"
                assessed = [row for row in evaluable_rows if row[field] is not None]
                unavailable_statuses = sorted({row[kind+"_availability_status"] for row in evaluable_rows})
                metric(field, stage, sum(row[field] for row in assessed), len(assessed), view=view,
                    reason=None if assessed else unavailable_statuses[0] if len(unavailable_statuses) == 1
                    else "not_evaluable_empty_population" if not evaluable_rows else "unavailable_mixed_reasons")
                metric(kind+"_availability_assessment_coverage", stage, len(assessed), len(evaluable_rows), view=view)
            geometric = [row for row in evaluable_rows if row["geometric_correct_reference_acceptance"] is not None]
            metric("geometric_correct_reference_resolution", stage,
                sum(row["geometric_correct_reference_acceptance"] for row in geometric), len(evaluable_rows), view=view,
                reason=None if len(geometric) == len(evaluable_rows) else "unavailable_geometric_acceptance")
        # Availability strata use both origins and F, never alter binary recovery reach.
        for origin in ("stage1_clonal_recheck", "stage2_near_clonal_recheck", "unchanged_F"):
            rows = [row for row in populations[origin] if row["evaluable"]]
            for kind in ("geometric", "usable"):
                for value in (True, False, None):
                    selected = [row for row in rows if row[kind+"_reference_available"] is value]
                    stratum = f"{origin}:{kind}:{'available' if value is True else 'absent' if value is False else 'unassessed'}"
                    counts = Counter(row["detailed_membership_status"] for row in selected)
                    for name, count in (("accepted", sum(row["accepted_label"] is not None for row in selected)),
                            ("binary_correct", sum(row["binary_clonality_correct"] is True for row in selected)),
                            ("binary_wrong", sum(row["binary_clonality_correct"] is False for row in selected)),
                            *((key, counts[key]) for key in ("Mc", "Mw", "U", "unresolved"))):
                        metric("availability_stratified_"+name, stage, count, len(selected),
                            view="availability", stratum=stratum,
                            reason="unavailable_analysis" if origin != "unchanged_F" and not assignment_available else None)
        neval = len(target_evaluable)
        reason = None if assignment_available else "unavailable_analysis"
        for name, num, den in (("acceptance_coverage", accepted_count, neval), ("correct_resolution", correct_count, neval),
                ("wrong_per_evaluable", wrong_count, neval), ("selective_error", wrong_count, accepted_count),
                ("unresolved_fraction", neval-accepted_count, neval), ("probability_availability", finite_probability_count, neval),
                ("multiclass_correct_resolution", mapped_correct, neval), ("unmapped_accepted_fraction", unmapped_accepted, mapped_accepted),
                ("absent_class_false_acceptance", absent_accepted, neval),
                ("multiclass_accuracy_lower", mapped_correct, mapped_accepted),
                ("multiclass_accuracy_upper", mapped_correct+unmapped_accepted, mapped_accepted)):
            metric(name, stage, num if assignment_available else None, den if assignment_available else None,
                   value=None, reason=reason)
        for name in TRANSITIONS:
            metric("transition_fraction", stage, transitions[name] if assignment_available else None,
                   neval if assignment_available else None, stratum=name, reason=reason)
            metric("multiclass_transition_fraction", stage, multi_transitions[name] if assignment_available else None,
                   neval if assignment_available else None, stratum=name, reason=reason)
        wrong_baseline = sum(not baseline_correct[m] for m in target_evaluable)
        correct_baseline = neval-wrong_baseline
        metric("baseline_error_recovery", stage, transitions["error_corrected"], wrong_baseline, reason=reason)
        metric("baseline_correct_introduced_error", stage, transitions["error_introduced"], correct_baseline, reason=reason)
        final_correct = {m: (label == clonal_label) == clonal[m] for m, label in final.items()}
        metric("correct_per_original_evaluable", stage,
               sum(final_correct.get(m, False) and evaluable[index[m]] for m in ids), original_n,
               reason=None if assignment_available else "unavailable_targets_no_correct_decision_credit")
        metric("whole_workflow_wrong_acceptance", stage,
               sum(final_correct.get(m) is False and evaluable[index[m]] for m in ids), original_n)
        metric("whole_workflow_unresolved_fraction", stage,
               sum(m not in final and evaluable[index[m]] for m in ids), original_n)
        unchanged_evaluable = {m for m in non_targets if evaluable[index[m]]}
        metric("unchanged_population_membership_error", stage,
               sum(final_correct.get(m) is False for m in unchanged_evaluable), len(unchanged_evaluable))
        for role, members in (("target", targets), ("unchanged_non_target", non_targets)):
            metric("correct_contribution_per_original", stage,
                   sum(final_correct.get(m, False) and evaluable[index[m]] for m in members), original_n, stratum=role)
        metric("reference_candidates", stage, value=len(labels) if labels else None)
        metric("target_count", stage, value=len(targets))
        metric("unchanged_non_target_count", stage, value=len(non_targets))
        for origin, population in (("stage1_clonal_recheck", target_evaluable & g0),
                                   ("stage2_near_clonal_recheck", target_evaluable - g0)):
            for status in ("assigned_relative", "unresolved_reference", "unresolved_fit", "low_confidence", "ambiguous_tie"):
                metric("target_origin_status_fraction", stage,
                    sum(records.get(m, {}).get("status") == status for m in population), len(population),
                    stratum=origin+":"+status, reason=reason)
            for outcome, count in (("correct_accepted", sum(final_correct.get(m) is True for m in population)),
                                   ("wrong_accepted", sum(final_correct.get(m) is False for m in population)),
                                   ("unresolved", sum(m not in final for m in population))):
                metric("target_origin_outcome_fraction", stage, count, len(population),
                       stratum=origin+":"+outcome, reason=reason)
        partition(stage, final, ids, "accepted_only")
        ccf(stage, accepted_estimates, "newly_accepted_assignment")
        if assignment:
            for row in assignment.get("multiplicities", ()):
                i, j = index[row["mutation_id"]], regions.index(row["region_id"])
                if catalog.eligible[i, j] and (row["original_lower"] != 1 or row["original_upper"] != catalog.major_cn[i, j]):
                    raise ValueError("Stage-three original support invariant failed")
                if row["mutation_id"] not in conditional_calls:
                    conditional_calls[row["mutation_id"]] = np.full(len(regions), np.nan)
                    assigned_calls[row["mutation_id"]] = np.full(len(regions), np.nan)
                conditional_calls[row["mutation_id"]][j] = float(row["conditional_map"]) if float(row["phi"]) > 0 else np.nan
                assigned_calls[row["mutation_id"]][j] = float(row["assigned_multiplicity"])
                native = float(row["native_call"])
                native_valid = bool(observed[i, j] and catalog.mapping_genuine[i, j] and float(row["native_phi"]) > 0
                                    and np.isfinite(native) and row["native_lower"] <= native <= row["native_upper"])
                positive = bool(catalog.eligible[i, j] and history.phi[i, j] > 0 and np.isfinite(history.multiplicity[i, j]))
                source_violation = bool(not row["original_lower"] <= native <= row["original_upper"]) if native_valid else None
                if source_violation:
                    raise ValueError("Stage-three valid native source-call containment failed")
                correct_cn = bool(catalog.major_cn[i, j] == history.major_cn[i, j]
                                  and catalog.minor_cn[i, j] == history.minor_cn[i, j])
                truth_excluded = bool(history.multiplicity[i, j] > row["original_upper"]) if positive else None
                if positive and correct_cn and truth_excluded:
                    raise ValueError("Stage-three core true-multiplicity containment failed")
                containment.append(dict(arm=stage, mutation_id=row["mutation_id"], region_id=row["region_id"],
                    source_stratum="qualified_G0" if (row["mutation_id"], row["region_id"]) in qualified else
                                   "other_G0" if row["mutation_id"] in g0 else "newly_merged_stage_two",
                    original_eligible=bool(catalog.eligible[i, j]), candidates_checked=len(labels), support_violation=False,
                    source_call_valid=native_valid, source_call_violation=source_violation,
                    positive_carrier_truth=positive, true_support_exclusion=truth_excluded, correct_supplied_cn=correct_cn,
                    inherited_input_exclusion=bool(history.multiplicity[i, j] > catalog.major_cn[i, j]) if positive else None))
            multiplicity(stage, conditional_calls, "displayed_best_candidate")
            multiplicity(stage, assigned_calls, "assigned_call")
        arm_states[arm] = dict(baseline=baseline_correct, baseline_multiclass=baseline_multi, final=final_correct,
                              final_labels=final, targets=targets, non_targets=non_targets, available=available,
                              assignment_available=assignment_available, evaluable_target_count=neval)
        final_views[stage] = final
    common = set(first_labels) & set(native_views.get("B", {})) & set(native_views.get("C", {}))
    for arm, labels in (("A", first_labels), *native_views.items()):
        partition(arm, labels, sorted(common), "common_native_remainder")
    common_combined = set(combined_views.get("B", {})) & set(combined_views.get("C", {}))
    partition("A", first_labels, sorted(common_combined), "common_combined")
    for arm, labels in combined_views.items():
        partition(arm, labels, sorted(common_combined), "common_combined")
    common_targets = arm_states["B"]["targets"] & arm_states["C"]["targets"]
    for arm in ("B", "C"):
        state = arm_states[arm]
        selected = {m for m in common_targets if evaluable[index[m]]}
        metric("exclusive_target_count", arm+"3", value=len(state["targets"]-common_targets))
        metric("common_target_correct_resolution", arm+"3",
               sum(state["final"].get(m) is True for m in selected), len(selected),
               reason=None if state["assignment_available"] else "unavailable_analysis")
        metric("common_target_wrong_per_evaluable", arm+"3",
               sum(state["final"].get(m) is False for m in selected), len(selected),
               reason=None if state["assignment_available"] else "unavailable_analysis")
        metric("common_target_acceptance", arm+"3", sum(m in state["final"] for m in selected), len(selected),
               reason=None if state["assignment_available"] else "unavailable_analysis")
    reach = recovery_reach(arm_states, evaluable, index)
    for name, values in reach.items():
        if isinstance(values, dict) and "numerator" in values:
            metric(name, "C3_vs_B", values["numerator"], values["denominator"], view="cap_associated_error")
    bounds = result.get("cap_bounds")
    audit = cap_audit(history, catalog, result.get("remainder_ids", ()), contributors,
                      np.asarray(bounds["upper"], float)) if bounds else []
    if bounds:
        remainder_indices = [index[m] for m in result.get("remainder_ids", ())]
        eligible = catalog.eligible[remainder_indices]
        upper = np.asarray(bounds["upper"], float)[remainder_indices]
        metric("cap_activity_all_eligible", "C", int(np.sum(eligible & (upper < catalog.major_cn[remainder_indices]))),
               int(eligible.sum()))
    _cap_metrics(audit, metric)
    for arm in ("B3", "C3"):
        rows = [row for row in containment if row["arm"] == arm]
        available = arm_states[arm[0]]["assignment_available"]
        for name, count in (("stage3_support_violation_count", sum(row["support_violation"] for row in rows)),
                ("stage3_source_call_violation_count", sum(row["source_call_violation"] is True for row in rows)),
                ("stage3_core_truth_violation_count", sum(row["positive_carrier_truth"] and row["correct_supplied_cn"]
                    and row["true_support_exclusion"] for row in rows)),
                ("stage3_inherited_support_defect_count", sum(row["inherited_input_exclusion"] is True for row in rows))):
            metric(name, arm, value=count if available else None, reason=None if available else "unavailable_analysis")
    _probability_metrics(calibration_rows, metric)
    declared_sources = tuple(metadata.get("declared_source_ids", ()))
    structural_catalog = SimpleNamespace(eligible=np.asarray(history.major_cn) <= catalog.max_major_cn,
        segment_ids=history.segment_ids, major_cn=history.major_cn,
        count_observed=np.ones_like(observed), total_counts=np.ones_like(catalog.total_counts))
    declared_contributors = [dict(mutation_id=m, region_id=r, segment_id=str(history.segment_ids[index[m], j]),
                                  usable=True) for m in declared_sources for j, r in enumerate(regions)]
    declared = cap_audit(history, structural_catalog, metadata.get("declared_remainder_ids", ()), declared_contributors,
                         np.asarray(history.major_cn), scope="declared_pre_read_source_truth")
    for row in declared:
        row["estimated_upper"] = row["truth_replacement_upper"]
        row["excluded"] = row["truth_replacement_exclusion"]
        row["cap_active"] = row["truth_replacement_upper"] is not None and row["truth_replacement_upper"] < row["original_upper"]
    declared_valid = [row for row in declared if row["truth_replacement_evaluable"]]
    metric("declared_source_truth_cap_exclusion", "structural", sum(row["excluded"] for row in declared_valid), len(declared_valid))
    true_purity = acquisition.get("true_purity", acquisition.get("purity", catalog.purity))
    true_normal = acquisition.get("normal_cn", np.full_like(history.phi, 2.))
    information = information_diagnostics(history.phi, history.multiplicity, history.major_cn,
        history.minor_cn, true_purity, true_normal, catalog.total_counts, observed, history.observable_truth_ids)
    information = [dict(mutation_id=m, **row) for m, row in zip(ids, information)]
    information_lookup = {row["mutation_id"]: row for row in information}
    for row in mutation_rows:
        row.update(original_evaluable=bool(evaluable[index[row["mutation_id"]]]),
                   positive_carrier_truth=bool(np.any(history.phi[index[row["mutation_id"]]] > 0)),
                   information_stratum=information_lookup[row["mutation_id"]]["information_stratum"],
                   expected_kl=information_lookup[row["mutation_id"]]["expected_kl"],
                   target=row["role"] == "target", unchanged_F=row["role"] == "unchanged_non_target",
                   unavailable=row["status"] == "unavailable_analysis" or not arms.get(row["arm"][0], {}).get("reference"))
    for item in applicability:
        metric("analysis_available", item["arm"], int(item["available"]), 1, view="all_planned_cells")
        failed = any(word in item["status"].lower() for word in ("fail", "error", "timeout"))
        metric("analysis_failure", item["arm"], int(failed), 1, view="all_planned_cells")
    curves = []
    for arm in ("B3", "C3"):
        state = arm_states[arm[0]]
        if not state["assignment_available"]:
            continue
        for threshold in (.8, .9, .95, .99, 1.):
            selected = [row for row in calibration_rows if row["arm"] == arm and row["nonconfidence_gates_pass"]
                        and posterior_passes(row["log_competitor_ratio"], threshold, row["has_finite_alternative"])]
            wrong = sum(not row["best_clonal_correct"] for row in selected)
            curves.append(dict(arm=arm, threshold=threshold, evaluable=state["evaluable_target_count"],
                accepted=len(selected), wrong=wrong, acceptance_coverage=_ratio(len(selected), state["evaluable_target_count"]),
                selective_error=_ratio(wrong, len(selected)),
                wrong_per_evaluable=_ratio(wrong, state["evaluable_target_count"])))
    input_error = ((catalog.major_cn != history.major_cn) | (catalog.minor_cn != history.minor_cn)
                   | (catalog.purity != np.broadcast_to(true_purity, history.phi.shape))
                   | (catalog.segment_ids != history.segment_ids) | ~catalog.mapping_genuine)
    # Original names stay available for frozen selection/report consumers. Explicit
    # aliases distinguish the unchanged binary endpoint from detailed membership.
    binary_names = {"correct_resolution", "selective_error", "wrong_per_evaluable", "correct_per_original_evaluable",
                    "whole_workflow_wrong_acceptance", "baseline_error_recovery", "unchanged_population_membership_error"}
    for row in list(metrics):
        if row["name"] in binary_names:
            row["endpoint"] = "binary_clonality"
            metrics.append(dict(row, name="binary_"+row["name"]))
    return json_safe(dict(schema="clipp2.simulation.evaluation.v2", **identity,
        evaluation_config=evaluation_config, evaluation_definition=definition,
        evaluation_definition_id=content_hash(definition), original_mutation_count=len(ids),
        original_evaluable_count=original_n, region_count=len(regions), metrics=metrics, mutations=mutation_rows,
        input_error_fraction=float(input_error.mean()),
        applicability=applicability, cap_audit=audit, declared_source_audit=declared,
        candidate_correspondence=correspondence, reference_center_geometry=reference_geometry,
        reference_match_atol=reference_match_atol, reference_distance=reference_distance,
        detailed_membership=detailed_membership, stage_three_containment=containment,
        calibration=calibration_rows, information=information, recovery_reach=reach,
        error_coverage_curves=curves,
        reliability={arm: reliability([row["clonal_probability"] for row in calibration_rows if row["arm"] == arm],
            [row["truth_clonal"] for row in calibration_rows if row["arm"] == arm]) for arm in ("B3", "C3")},
        acquisition={key: acquisition[key] for key in ("id", "purity", "mean_depth", "panel") if key in acquisition}))


def recovery_reach(states, evaluable, index):
    """Operational cap-associated errors on a fixed common accepted population."""
    if not all(states.get(arm, {}).get("available") for arm in ("B", "C")):
        return dict(status="unavailable_paired_combined_populations", H=[], H_T=[], H_F=[])
    b, c = states["B"], states["C"]
    common = {m for m in set(b["baseline"]) & set(c["baseline"]) if evaluable[index[m]]}
    harmful = {m for m in common if b["baseline"][m] and not c["baseline"][m]}
    targeted, fixed = harmful & c["targets"], harmful & c["non_targets"]
    if targeted | fixed != harmful or targeted & fixed:
        raise ValueError("Cap-associated error set does not partition into reassessed and unchanged members")
    if any(c["final"].get(m) != c["baseline"][m] for m in c["non_targets"]):
        raise ValueError("Stage three changed non-target membership correctness")
    result = dict(status="available" if c["assignment_available"] else "assignment_unavailable",
                  H=sorted(harmful), H_T=sorted(targeted), H_F=sorted(fixed), common_evaluable=len(common),
                  recovery_ceiling=dict(numerator=len(targeted), denominator=len(harmful)))
    if c["assignment_available"]:
        recovered = {m for m in targeted if c["final"].get(m) is True}
        persistent = {m for m in targeted if c["final"].get(m) is False}
        unresolved = targeted-set(c["final"])
        if len(recovered) > len(targeted):
            raise ValueError("Recovery exceeded the procedural ceiling")
        result.update(recovered_H=dict(numerator=len(recovered), denominator=len(harmful)),
            persistent_wrong_H=dict(numerator=len(persistent), denominator=len(harmful)),
            unresolved_H=dict(numerator=len(unresolved), denominator=len(harmful)),
            unchanged_errors_H=dict(numerator=len(fixed), denominator=len(harmful)),
            conditional_target_recovery=dict(numerator=len(recovered), denominator=len(targeted)))
    return result


def _cap_metrics(audit, metric):
    for stratum, rows in (("all_positive_carrier_pairs", audit),
                          ("informative_positive_carrier_pairs", [r for r in audit if r["observed"]])):
        for name, field in (("stage2_truth_exclusion", "excluded"), ("stage2_additional_exclusion", "additional_exclusion"),
                            ("inherited_original_exclusion", "inherited_exclusion"), ("cap_activity", "cap_active"),
                            ("cap_fallback", "fallback")):
            metric(name, "C", sum(row[field] for row in rows), len(rows), stratum=stratum)
        inside = [row for row in rows if not row["inherited_exclusion"]]
        metric("stage2_additional_exclusion_among_original_retained", "C",
               sum(row["additional_exclusion"] for row in inside), len(inside), stratum=stratum)
        substituted = [row for row in rows if row["truth_replacement_evaluable"]]
        metric("true_call_replacement_exclusion", "C", sum(row["truth_replacement_exclusion"] for row in substituted),
               len(substituted), stratum=stratum)
        metric("true_call_replacement_availability", "C", len(substituted), len(rows), stratum=stratum)
        metric("harmful_minimum_call_error", "C", sum(row["harmful_minimum_call_error"] for row in substituted),
               len(substituted), stratum=stratum)
        for unit, key in (("mutation", lambda row: row["mutation_id"]),
                          ("segment", lambda row: (row["region_id"], row["segment_id"]))):
            all_units, excluded_units = {key(row) for row in rows}, {key(row) for row in rows if row["excluded"]}
            metric("stage2_exclusion_"+unit, "C", len(excluded_units), len(all_units), stratum=stratum)
        metric("stage2_any_exclusion", "C", int(any(row["excluded"] for row in rows)), int(bool(rows)), stratum=stratum)
        metric("mean_support_width", "C", value=float(np.mean([r["estimated_upper"] for r in rows])) if rows else None,
               stratum=stratum)
        metric("mean_contributor_count", "C", value=float(np.mean([r["contributor_count"] for r in rows])) if rows else None,
               stratum=stratum)


def _probability_metrics(rows, metric):
    for arm in ("B3", "C3"):
        arm_rows = [row for row in rows if row["arm"] == arm]
        strata = {"all": arm_rows, "K1": [row for row in arm_rows if row["K"] == 1],
                  "K_multiple": [row for row in arm_rows if row["K"] > 1],
                  "true_class_present": [row for row in arm_rows if row["true_class_present"]],
                  "true_class_absent": [row for row in arm_rows if not row["true_class_present"]]}
        for source in sorted({row["source_stage"] for row in arm_rows if row["source_stage"] is not None}):
            strata["source:"+source] = [row for row in arm_rows if row["source_stage"] == source]
        for stratum, selected in strata.items():
            for name in ("binary_brier", "binary_log_loss", "multiclass_brier", "multiclass_log_loss", "unmapped_mass"):
                metric(name, arm, value=float(np.mean([row[name] for row in selected])) if selected else None, stratum=stratum)
            metric("infinite_multiclass_log_loss", arm, sum(np.isposinf(row["multiclass_log_loss"]) for row in selected),
                   len(selected), stratum=stratum)


def _numeric(value):
    if value in ("Infinity", "-Infinity"):
        return float("inf") if value == "Infinity" else -float("inf")
    return value


def aggregate(evaluations, *, bootstrap_replicates=1000, seed=20261106, families=FAMILIES):
    """Equal-family means and paired history bootstrap carrying all acquisitions.

    Metrics conditional on successful analyses report their available cell counts;
    failure/availability endpoints use every planned cell. Missing families never
    disappear into a more favorable macro-average.
    """
    evaluations = list(evaluations)
    definitions = {}
    for record in evaluations:
        schema = record.get("schema")
        if schema == "clipp2.simulation.evaluation.v2":
            definition = record.get("evaluation_definition")
            identity = record.get("evaluation_definition_id")
            if not isinstance(definition, dict) or identity != content_hash(definition):
                raise ValueError("Evaluation definition identity is missing or invalid")
        elif schema == "clipp2.simulation.evaluation.v1":
            # Label historical records; never silently recalculate old geometry.
            definition = dict(schema="clipp2.simulation.evaluation-definition.legacy-v1",
                binary_endpoint="accepted_designated_clonal_status_equals_physical_truth_clonality",
                reference_distance="linf", reference_match_atol=record.get("reference_match_atol"),
                matching_policy="legacy_forced_pairs_influence_valid_edge_cost",
                detailed_endpoint="legacy_multiclass_unmapped_treated_as_incorrect")
            identity = content_hash(definition)
        else:
            raise ValueError("Aggregation requires versioned evaluation records")
        definitions[identity] = definition
    if len(definitions) > 1:
        raise ValueError("Incompatible evaluation definitions require separate aggregates or explicit re-evaluation")
    if definitions:
        definition_id, definition = next(iter(definitions.items()))
    else:
        definition = evaluation_definition()
        definition_id = content_hash(definition)
    if bootstrap_replicates < 0:
        raise ValueError("Bootstrap replicate count must be nonnegative")
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for evaluation in evaluations:
        for row in evaluation["metrics"]:
            key = (row["name"], row["arm"], row.get("view", ""), row.get("stratum", "all"))
            groups[key].append((evaluation["family"], str(evaluation["history_id"]),
                                str(evaluation["acquisition_id"]), _numeric(row["value"])))
    # Paired differences are first formed inside the same history/acquisition.
    for left, right in (("C", "B"), ("B3", "B"), ("C3", "C"), ("C3", "B3")):
        keys = list(groups)
        for name, arm, view, stratum in keys:
            if arm != right or (name, left, view, stratum) not in groups:
                continue
            a = {(f, h, acq): value for f, h, acq, value in groups[name, left, view, stratum]}
            b = {(f, h, acq): value for f, h, acq, value in groups[name, right, view, stratum]}
            paired = []
            for key in a.keys() & b.keys():
                x, y = a[key], b[key]
                value = x-y if x is not None and y is not None and np.isfinite(x) and np.isfinite(y) else None
                paired.append((*key, value))
            groups[name, left+"_minus_"+right, view, stratum] = paired
    summaries = []
    for key, rows in sorted(groups.items()):
        by_family = defaultdict(lambda: defaultdict(list))
        for family, history, acquisition, value in rows:
            if value is not None:
                by_family[family][history].append(float(value))
        family_means, history_means, family_counts = {}, {}, {}
        for family in families:
            values = [float(np.mean(acquisitions)) for acquisitions in by_family[family].values()]
            history_means[family] = np.asarray(values)
            family_means[family] = float(np.mean(values)) if values else None
            family_counts[family] = len(values)
        complete = all(family_means[family] is not None for family in families)
        mean = float(np.mean(list(family_means.values()))) if complete else None
        intervals = []
        sufficient = all(family_counts[f] >= 2 for f in families)
        if complete and sufficient and mean is not None and np.isfinite(mean):
            for _ in range(bootstrap_replicates):
                intervals.append(np.mean([rng.choice(history_means[f], len(history_means[f]), replace=True).mean()
                                          for f in families]))
        low, high = np.quantile(intervals, [.025, .975]).tolist() if intervals else (None, None)
        summaries.append(dict(name=key[0], arm=key[1], view=key[2], stratum=key[3], equal_family_mean=mean,
            lower_95=low, upper_95=high, status="available" if complete else "unavailable_families",
            uncertainty_status="available" if intervals else "insufficient_histories_or_unavailable_finite_estimand",
            family_means=family_means, family_history_counts=family_counts, planned_cells=len(evaluations),
            available_cells=sum(value is not None for *_, value in rows),
            unavailable_families=[f for f in families if family_means[f] is None]))
    return json_safe(dict(schema="clipp2.simulation.aggregate.v2", evaluation_definition=definition,
        evaluation_definition_id=definition_id, weighting="equal_family_then_equal_history_then_equal_acquisition",
        uncertainty="paired_history_bootstrap_all_acquisitions_together", bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=seed, families=list(families), planned_datasets=len(evaluations), metrics=summaries))
