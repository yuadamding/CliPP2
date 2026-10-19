"""Mandatory, separately budgeted final-candidate refinement.

Completion is an exact-state local statement, never global optimality. A work
unit is one atomic *visit*: a full transfer scan, one conditional-refit pass,
up to two transfer proposals and the declared bounded joint neighborhood.
The original candidate archive is append-only. Budget exhaustion is unfinished.
"""
import hashlib
import json
import math
from time import perf_counter

import numpy as np

from ..kernel.assignment import feasible_transfer_scan
from ..kernel.objectives import assigned_objective
from .reassignment import _proposals, SCORE_ATOL, SCORE_RTOL
from .selection import _score_key


def state_digest(model, candidate, scientific_config=None):
    """Bind checks to exact labels, centers, weights, anchor and model identity."""
    fit = candidate.refit
    labels = np.asarray(fit.labels)
    if labels.dtype.kind not in 'iu' or labels.shape != (model.n,):
        raise ValueError('Finalization digest requires exact integer membership labels')
    digest = hashlib.sha256(b'clipp2_finalization_state_v1')
    digest.update(json.dumps({
        'model_identity': model.identity,
        'objective': model.selection_objective,
        'scientific_config': scientific_config or {},
        'anchor': int(fit.clonal_cluster),
        'scored_fields': {name: float(getattr(fit, name)) for name in (
            'score', 'conditional_log_likelihood', 'selection_log_likelihood',
            'classification_count_term', 'complexity_penalty')},
    }, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
    for name, dtype in (('labels', '<i8'), ('centers', '<f8'), ('weights', '<f8')):
        value = np.asarray(getattr(candidate.refit, name), dtype=dtype)
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _membership(candidate):
    """Partition and anchor identity without decoding compact archived labels."""
    fit = candidate.refit
    anchor = int(fit.clonal_cluster)
    if getattr(candidate, 'membership_kind', 'general') == 'tree':
        return ('tree', tuple(candidate.cuts)), anchor
    if getattr(fit, 'groups', None) is not None:
        # Groups map frozen-tree components to fitted cluster IDs. Canonicalize
        # this small map, not the N-vector, and move the anchor with its group.
        mapping = {}
        groups = tuple(mapping.setdefault(int(group), len(mapping)) for group in fit.groups)
        part = (('tree', tuple(fit.cuts)) if len(mapping) == len(groups) else
                ('general_compact', tuple(fit.cuts), groups))
        return part, mapping[anchor]
    return ('general', np.asarray(fit.labels, dtype='<i8').tobytes()), anchor


def finalist_indices(bank, diverse_starts=8):
    """Every evaluated K is mandatory; extras retain partition/anchor diversity."""
    ordered = sorted(range(len(bank.candidates)), key=lambda i: _score_key(bank.candidates[i]))
    best_k = {}
    for i in ordered:
        best_k.setdefault(len(bank.candidates[i].refit.centers), i)
    starts = list(dict.fromkeys([ordered[0], *[best_k[k] for k in sorted(best_k)]]))
    anchored = {_membership(bank.candidates[i]) for i in starts}
    partitions = {part for part, _ in anchored}
    extra = 0
    # First spend half the extra quota on genuinely different partitions,
    # then retain alternative anchor designations in competitive partitions.
    for require_partition in (True, False):
        for i in ordered:
            candidate = bank.candidates[i]
            key = _membership(candidate)
            part, _ = key
            if key in anchored or (require_partition and part in partitions):
                continue
            if extra >= diverse_starts or (require_partition and extra >= (diverse_starts+1)//2):
                break
            starts.append(i)
            partitions.add(part)
            anchored.add(key)
            extra += 1
    return starts


def choose_finalists(bank, diverse_starts=8):
    return [bank.candidates[i] for i in finalist_indices(bank, diverse_starts)]


def _reconstruct(model, candidate, columns):
    from .reporting import independent_log_columns
    fit = candidate.refit
    labels, centers = np.asarray(fit.labels), np.asarray(fit.centers)
    if not fit.eligible or fit.selection_objective != model.selection_objective:
        raise ValueError('Finalization requires an eligible objective-matched fitted state')
    summary = model.regional_support.summarize_partition(labels)
    if (np.any(summary.observed_counts == 0) or np.any(summary.support_conflict)
            or not np.isfinite(centers).all()
            or np.any(centers[labels] < model.lower) or np.any(centers[labels] > model.upper)
            or not np.array_equal(centers[fit.clonal_cluster], np.ones(model.r))
            or not model.regional_support.assigned_mask(labels, centers).all()):
        raise ValueError('Finalization structural revalidation failed')
    independent = independent_log_columns(model, centers)
    if not np.allclose(columns, independent, atol=1e-8, rtol=1e-10):
        raise ValueError('Finalization model emissions disagree with independent raw-input likelihood')
    assigned = math.fsum(independent[np.arange(model.n), labels])
    scored = assigned_objective(assigned, np.bincount(labels), model.r, model.selection_objective)
    for name in ('score', 'conditional_log_likelihood', 'selection_log_likelihood',
                 'classification_count_term', 'complexity_penalty'):
        if not np.isclose(scored[name], getattr(fit, name), atol=SCORE_ATOL, rtol=SCORE_RTOL):
            raise ValueError(f'Finalization independent score reconstruction failed: {name}')
    if not np.allclose(fit.weights, scored['cluster_weights'], atol=1e-12, rtol=0):
        raise ValueError('Finalization proportions disagree with published assignments')
    return scored, independent


def finalize_candidates(model, bank, *, max_visits=256, diverse_starts=8,
                        joint_limit=12, joint_moves=True, joint_proposals=None,
                        score_atol=SCORE_ATOL, score_rtol=SCORE_RTOL,
                        scientific_config=None, checkpoint_state=None, persist=None):
    """Refine the winner and every best-per-K lineage; return resumable evidence.

    The return contains ``selected`` (Candidate), ``diagnostics`` (JSON-safe),
    and ``state`` (JSON-safe). ``persist(state)`` runs after each atomic visit;
    the caller must snapshot the append-only bank in the same checkpoint.
    ``max_visits`` is cumulative when resuming, not a fresh allowance.
    """
    for value, name in ((max_visits, 'max_visits'), (diverse_starts, 'diverse_starts'),
                        (joint_limit, 'joint_limit')):
        if type(value) is not int or value < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
    if not bank.candidates:
        raise ValueError('Finalization requires an eligible candidate bank')
    if not np.isfinite([score_atol, score_rtol]).all() or min(score_atol, score_rtol) < 0:
        raise ValueError('Finalization tolerances must be finite and nonnegative')
    if joint_moves and joint_limit == 0:
        raise ValueError('Joint finalization requires a positive declared proposal limit')
    if model.selection_objective == 'legacy_mixture':
        return {'selected': min(bank.candidates, key=_score_key), 'state': None,
                'diagnostics': {'applied': False, 'complete': False,
                    'status': 'not_applicable_legacy_mixture',
                    'fixed_center_single_transfer_closure': None}}
    if joint_moves and joint_proposals is None:
        from .anchor_moves import joint_proposals
    config = {'model_identity': model.identity, 'objective': model.selection_objective,
              'scientific_config': scientific_config or {}, 'diverse_starts': diverse_starts,
              'joint_limit': joint_limit, 'joint_moves': bool(joint_moves),
              'score_atol': float(score_atol), 'score_rtol': float(score_rtol)}
    if checkpoint_state is None:
        starts = finalist_indices(bank, diverse_starts)
        state = {'schema': 'clipp2_finalization_v1', 'config': config,
                 'initial_bank_size': len(bank.candidates), 'visits': 0,
                 'pending': starts.copy(), 'lineages': {str(i): i for i in starts},
                 'checks': {}, 'events': [], 'unresolved': {},
                 'counters': {'exact_transfer_scans': 0, 'conditional_refit_passes': 0,
                              'transfer_admissions': 0, 'joint_admissions': 0,
                              'fixed_state_admissions': 0, 'refit_failures': 0}}
    else:
        state = json.loads(json.dumps(checkpoint_state, allow_nan=False))
        if state.get('schema') != 'clipp2_finalization_v1' or state.get('config') != config:
            raise ValueError('Finalization resume configuration/identity mismatch')
        if max_visits < state['visits']:
            raise ValueError('Finalization cumulative allowance cannot shrink below completed visits')
        for check in state['checks'].values():
            index = check['bank_index']
            if index >= len(bank.candidates) or state_digest(model, bank.candidates[index], scientific_config) != check['digest']:
                raise ValueError('Finalization checkpoint state digest mismatch')
    counters = state['counters']
    counters.setdefault('conditional_refit_calls', 0)
    timings = state.setdefault('work_seconds', {})
    for name in ('score_revalidation', 'exact_transfer_scan', 'conditional_refit',
                 'fixed_state_validation', 'transfer_proposals', 'joint_proposals'):
        timings.setdefault(name, 0.)
    segment_before = dict(timings)
    keys = tuple(model.observation_keys())
    support = model.regional_support
    started = perf_counter()

    def index_of(candidate):
        return next(i for i, saved in enumerate(bank.candidates) if saved is candidate)

    def checked(candidate):
        return state['checks'].get(state_digest(model, candidate, scientific_config))

    def enqueue(index):
        candidate = bank.candidates[index]
        check = checked(candidate)
        if check is not None and check['complete']:
            return
        if str(index) in state['unresolved']:
            return
        if index in state['pending']:
            return
        # FIFO rounds prevent an improving winner from indefinitely starving
        # other declared finalists. New winners enter immediately but cannot
        # overtake already queued work; resume preserves the committed order.
        state['pending'].append(index)

    def admit(labels, centers, anchor, provenance):
        # Preserve the known feasible point independently of scalar optimizer
        # success. Conditional-refit failure cannot erase its exact score gain.
        timer = perf_counter()
        try:
            fixed = bank.retain_validated_fixed_fit(labels, centers, anchor, provenance)
        finally:
            timings['fixed_state_validation'] += perf_counter()-timer
        counters['fixed_state_admissions'] += 1
        result = [fixed]
        counters['conditional_refit_calls'] += 1
        timer = perf_counter()
        try:
            variants = bank.consider_variants(None, labels, centers, provenance,
                                             clonal_cluster=anchor)
            result += variants
            if not variants:
                counters['refit_failures'] += 1
                fixed.provenance.append({**provenance, 'refit_failure': 'no_eligible_conditional_refit'})
        except (RuntimeError, ValueError) as error:
            counters['refit_failures'] += 1
            provenance = {**provenance, 'refit_failure': str(error)}
            fixed.provenance.append(provenance)
        finally:
            timings['conditional_refit'] += perf_counter()-timer
        return result

    # Reconcile the live append-only archive even when a saved completed queue
    # is empty: a newly appended winner cannot inherit old completion evidence.
    for required in finalist_indices(bank, diverse_starts=0):
        state['lineages'].setdefault(str(required), required)
        enqueue(required)
    enqueue(index_of(min(bank.candidates, key=_score_key)))

    while state['pending'] and state['visits'] < max_visits:
        index = state['pending'].pop(0)
        candidate = bank.candidates[index]
        digest = state_digest(model, candidate, scientific_config)
        old_check = state['checks'].get(digest)
        if old_check is not None and old_check['complete']:
            continue
        state['visits'] += 1
        fit = candidate.refit
        tolerance = score_atol+score_rtol*max(1., abs(candidate.score))
        timer = perf_counter()
        columns = model.log_likelihood_columns(fit.centers)
        reconstructed, columns = _reconstruct(model, candidate, columns)
        timings['score_revalidation'] += perf_counter()-timer
        timer = perf_counter()
        scan = feasible_transfer_scan(columns, fit.labels, fit.centers,
            model.observed, support.hard_zero, support.positive, model.mutation_ids,
            objective=model.selection_objective, score_tolerance=tolerance,
            lower=model.lower, upper=model.upper)
        timings['exact_transfer_scan'] += perf_counter()-timer
        counters['exact_transfer_scans'] += 1
        considered = [candidate]
        own_refit = []
        refit_error = None
        provenance = {'route': 'mandatory_finalization', 'parent_bank_index': index,
                      'parent_state_digest': digest, 'visit': state['visits']}
        counters['conditional_refit_passes'] += 1
        counters['conditional_refit_calls'] += 1
        refit_before = {key: getattr(bank, key) for key in ('attempt_cache_hits', 'refit_calls')}
        scalar_before = model.telemetry.get('scalar_refits', 0)
        timer = perf_counter()
        try:
            own_refit = bank.consider_variants(None, fit.labels, fit.centers,
                {**provenance, 'family': 'conditional_refit'}, clonal_cluster=fit.clonal_cluster)
            considered.extend(own_refit)
            if not own_refit:
                counters['refit_failures'] += 1
                refit_error = 'no_eligible_conditional_refit'
        except (ValueError, RuntimeError) as error:
            counters['refit_failures'] += 1
            refit_error = str(error)
        finally:
            timings['conditional_refit'] += perf_counter()-timer
        conditional_stable = bool(own_refit and min(c.score for c in own_refit) >= candidate.score-tolerance)
        refit_evidence = {key: getattr(bank, key)-value for key, value in refit_before.items()}
        refit_evidence['scalar_refits'] = model.telemetry.get('scalar_refits', 0)-scalar_before
        refit_evidence['mechanism'] = ('cached_declared_refitter_result'
            if refit_evidence['attempt_cache_hits'] and not refit_evidence['refit_calls']
            else 'declared_conditional_refit_with_membership_block_cache')
        refit_evidence['fixed_state_import_populates_refit_cache'] = False
        if not scan['closed']:
            timer = perf_counter()
            witness = scan['witness']
            single = np.asarray(fit.labels).copy()
            single[witness['mutation_index']] = witness['destination']
            box_allowed = ((fit.centers[None, :, :] >= model.lower[:, None, :])
                           & (fit.centers[None, :, :] <= model.upper[:, None, :])).all(axis=2)
            (_, batch), batch_scan = _proposals(np.asarray(fit.labels), fit.centers,
                np.where(box_allowed, columns, -np.inf), model.observed, keys, support=support,
                objective=model.selection_objective)
            proposals = [('single', single)]
            if batch is not None and not np.array_equal(batch, single):
                proposals.append(('dynamic_batch', batch))
            timings['transfer_proposals'] += perf_counter()-timer
            for family, labels in proposals:
                counters['transfer_admissions'] += 1
                considered.extend(admit(labels, fit.centers, fit.clonal_cluster,
                    {**provenance, 'family': family, 'batch_scan': batch_scan}))
        coverage = {'declared': False, 'reason': 'joint_moves_disabled',
                    'evaluated': 0, 'completed': True}
        if joint_moves:
            timer = perf_counter()
            proposals, coverage = joint_proposals(model, candidate, columns, limit=joint_limit, keys=keys)
            timings['joint_proposals'] += perf_counter()-timer
            coverage = dict(coverage)
            coverage['evaluated'] = 0
            for proposal in proposals:
                if proposal.target_q != len(np.unique(proposal.labels)):
                    raise ValueError('Joint proposal declared the wrong occupied cluster count')
                counters['joint_admissions'] += 1
                considered.extend(admit(proposal.labels, proposal.centers, proposal.clonal_cluster,
                    {**provenance, 'family': 'joint_anchor', **proposal.details}))
                coverage['evaluated'] += 1
            coverage['completed'] = coverage['evaluated'] == len(proposals)
            coverage['scope'] = 'declared_bounded_joint_neighborhood_not_all_joint_edits'
        best = min(considered, key=_score_key)
        improvement = candidate.score-best.score
        # Joint coverage means the declared proposals were evaluated, not that
        # this parent beats them. A q-reducing neighbor can be better forever;
        # demanding joint non-improvement here would repeatedly revisit that
        # best-per-q parent instead of certifying its improved successor.
        complete = bool(scan['closed'] and conditional_stable and coverage['completed'])
        check = {'digest': digest, 'bank_index': index,
                 'structurally_verified': True, 'score': reconstructed['score'],
                 'fixed_center_single_transfer_closure': scan['closed'],
                 'single_transfer_scan': scan, 'conditional_refit_stability': conditional_stable,
                 'conditional_refit_evidence': refit_evidence,
                 'conditional_refit_failure': refit_error, 'joint_neighborhood': coverage,
                 'best_evaluated_neighbor_bank_index': index_of(best),
                 'best_evaluated_neighbor_score_decrease': improvement,
                 'complete': complete, 'global_optimality_certified': False}
        state['checks'][digest] = check
        if best is not candidate and best.score < candidate.score:
            best_index = index_of(best)
            for start, endpoint in state['lineages'].items():
                if endpoint == index:
                    state['lineages'][start] = best_index
            enqueue(best_index)
        elif not complete:
            state['unresolved'][str(index)] = ('conditional_refit_failed' if not own_refit
                else 'declared_neighborhood_did_not_resolve_transfer_or_stability')
        state['events'].append({'visit': state['visits'], 'bank_index': index,
                                'digest': digest, 'complete': complete,
                                'best_score_decrease': improvement})
        # Any newly selected winner, including one created in this final visit,
        # remains pending until its OWN labels/centers/anchor pass the checks.
        for required in finalist_indices(bank, diverse_starts=0):
            state['lineages'].setdefault(str(required), required)
            enqueue(required)
        winner_index = index_of(min(bank.candidates, key=_score_key))
        enqueue(winner_index)
        if persist is not None:
            persist(state)

    selected = min(bank.candidates, key=_score_key)
    selected_check = checked(selected)
    endpoints = set(state['lineages'].values())
    unresolved = [i for i in sorted(endpoints)
                  if not (checked(bank.candidates[i]) or {}).get('complete', False)]
    complete = not unresolved and bool(selected_check and selected_check['complete'])
    diagnostics = {
        'applied': True, 'complete': complete,
        'status': 'complete_local_checks' if complete else 'unfinished',
        'termination_reason': ('declared_local_checks_completed' if complete else
            'visit_allowance_exhausted' if state['visits'] >= max_visits else 'unresolved_local_checks'),
        'max_visits': max_visits, 'visits': state['visits'], 'pending_bank_indices': state['pending'],
        'unresolved_finalist_bank_indices': unresolved, 'initial_bank_size': state['initial_bank_size'],
        'final_bank_size': len(bank.candidates), 'retained_original_bank': True,
        'selected_bank_index': index_of(selected),
        'selected_state_digest': state_digest(model, selected, scientific_config),
        'selected_check': selected_check, 'counters': dict(counters),
        'finalist_lineages': dict(state['lineages']), 'segment_seconds': perf_counter()-started,
        'work_seconds': {'cumulative': dict(timings),
            'segment': {name: value-segment_before[name] for name, value in timings.items()},
            'scope': 'nonoverlapping_completed_call_work_excluding_checkpoint_writes_and_control_overhead'},
        'scope': 'declared_conditional_refitter_and_bounded_joint_neighborhood',
        'scheduling_policy': 'deterministic_fifo_finalist_rounds_new_winner_pending_without_overtaking',
        'unresolved_failure_retry_policy': 'explicit_new_import_required_not_silent_retry_on_budget_extension',
        'global_optimality_certified': False,
    }
    return {'selected': selected, 'diagnostics': diagnostics, 'state': state}
