"""Source-bound finalization lifecycle shared by fresh fits and explicit imports."""
from pathlib import Path
from time import perf_counter

import numpy as np

from .checkpoint import identities, restore_bank, snapshot_bank, write_checkpoint
from .importing import POLISH_SCHEMA, checked_document, scientific_settings
from .polish import finalize_candidates
from .selection import TreeFit, _CandidateBank, _score_key
from ..kernel.topology import FrozenTree


class FinalizationIncomplete(RuntimeError):
    """Retained unfinished evidence is not a successfully publishable fit."""

    def __init__(self, result, checkpoint_path):
        self.result = result
        self.checkpoint_path = str(checkpoint_path)
        reason = result.diagnostics['finalization']['termination_reason']
        action = ('Resume with a larger cumulative allowance' if reason == 'visit_allowance_exhausted'
                  else 'Review unresolved checks; an explicit new import is required to retry failed refits')
        super().__init__(f'Finalization unfinished ({reason}); no public tables. '
                         f'{action}. Source-bound checkpoint: {checkpoint_path}')


def checkpoint_destination(outdir, checkpoint_path, *, importing=False):
    if checkpoint_path is not None:
        return Path(checkpoint_path) if importing else Path(str(checkpoint_path)+'.polish.json')
    return Path(str(outdir)+'.polish-checkpoint.json')


def _scalar_finalists(model, result, config):
    """Retain scalar archive as evidence; independently admit all best-per-K starts."""
    import json
    records = [r for r in result.candidate_bank if r.get('status') == 'scored'
               and r.get('publication_eligible') is True]
    records.sort(key=lambda r: (r['bic'], r['num_clusters'], r['replicate'], r['candidate_id']))
    best = {}
    for record in records:
        best.setdefault(record['num_clusters'], record)
    selected = list(best.values())
    keys = {(r['replicate'], r['candidate_id']) for r in selected}
    for record in records:
        key = (record['replicate'], record['candidate_id'])
        if key not in keys:
            selected.append(record)
            keys.add(key)
        if len(selected) >= len(best)+config.polish_diverse_starts:
            break
    tree = result.tree
    order = (np.r_[tree.edges[0, 0], tree.edges[:, 1]] if model.n > 1 else np.array([0]))
    if not np.array_equal(np.sort(order), np.arange(model.n)):
        raise ValueError('Scalar finalization requires the exact saved chain order')
    bank = _CandidateBank(model, tree, compact_general=True)
    for record in selected:
        values = record['partition_parameters']
        parameters = json.loads(values) if isinstance(values, str) else values
        labels = np.empty(model.n, dtype=int)
        labels[order] = np.asarray(parameters['block_labels'], dtype=int)[
            np.searchsorted(parameters['cuts'], np.arange(model.n), side='right')]
        centers = np.asarray(parameters['centers'])[:, None]/model.region_purities[0]
        anchor = parameters['clonal_cluster']
        centers[anchor] = 1.
        bank.retain_validated_fixed_fit(labels, centers, anchor,
            {'route': 'scalar_finalization_import', 'requested_k': len(centers),
             'original_replicate': record['replicate'], 'original_candidate_id': record['candidate_id']},
            expected_score=record['bic'])
    return bank


def load_finalization(model, path, config):
    """Same-source continuation only; changed source requires explicit import."""
    saved, _, _ = checked_document(path, schemas=(POLISH_SCHEMA,))
    for key, value in identities(model).items():
        if saved['identities'].get(key) != value:
            raise ValueError(f'Finalization checkpoint {key} changed; explicit new import required')
    if saved['scientific_config'] != scientific_settings(config):
        raise ValueError('Finalization scientific configuration changed')
    tree = FrozenTree(**saved['tree'])
    pilot = np.asarray(saved['pilot'])
    for key, value in identities(model, tree, pilot).items():
        if saved['identities'].get(key) != value:
            raise ValueError(f'Finalization checkpoint {key} changed')
    bank = restore_bank(model, tree, saved['bank'])
    scalar = saved.get('scalar_archive', [])
    result = TreeFit(min(bank.candidates, key=_score_key), [*scalar, *bank.candidates], tree, pilot,
                     saved['diagnostics'], bank=bank)
    return result, saved.get('finalization_state'), scalar


def finalize_result(model, result, config, path, *, resume=False):
    """Persist unfinished work and publishability checks without rewriting discovery."""
    if config.selection_objective == 'legacy_mixture':
        result.diagnostics['finalization'] = {'applied': False, 'complete': False,
            'status': 'not_applicable_legacy_mixture', 'global_optimality_certified': False}
        return result
    path = Path(path)
    saved_state, scalar_archive = None, []
    if resume and path.exists():
        result, saved_state, scalar_archive = load_finalization(model, path, config)
    else:
        if path.exists():
            raise FileExistsError('Finalization checkpoint exists; explicitly resume or use a fresh path')
        if getattr(result, 'bank', None) is None:
            if all(isinstance(c, dict) for c in result.candidate_bank):
                scalar_archive = list(result.candidate_bank)
                bank = _scalar_finalists(model, result, config)
            else:
                bank = _CandidateBank(model, result.tree, compact_general=True)
                # Fallback for callers providing a TreeFit without its live bank.
                for candidate in result.candidate_bank:
                    fit = candidate.fit
                    bank.retain_validated_fixed_fit(fit.labels, fit.centers, fit.clonal_cluster,
                        {'route': 'finalization_state_admission', 'requested_k': len(fit.centers)},
                        expected_score=candidate.score)
            result = TreeFit(min(bank.candidates, key=_score_key), bank.candidates,
                             result.tree, result.pilot, dict(result.diagnostics), bank=bank)
    bank = result.bank
    before = {}
    for candidate in bank.candidates:
        q = len(candidate.refit.centers)
        before[q] = min(before.get(q, float('inf')), candidate.score)
    result.diagnostics.setdefault('pre_finalization_best_score_per_k', before)
    checkpoint_io = dict(result.diagnostics.get('finalization_checkpoint_io', {
        'write_count': 0, 'write_seconds': 0., 'unrecorded_duration_count': 0,
        'current_write_duration_unrecorded': False}))
    if checkpoint_io['current_write_duration_unrecorded']:
        # The atomic snapshot proves its write completed, but cannot contain
        # its own final fsync/replace duration. Preserve this gap on resume.
        checkpoint_io['unrecorded_duration_count'] += 1
    checkpoint_io['current_write_duration_unrecorded'] = False
    checkpoint_io['scope'] = 'checkpoint_encoding_hashing_atomic_write_and_fsync'
    write_count_before = checkpoint_io['write_count']
    write_seconds_before = checkpoint_io['write_seconds']
    result.diagnostics['finalization_checkpoint_io'] = checkpoint_io

    def persist(state):
        checkpoint_io['write_count'] += 1
        checkpoint_io['segment_write_count'] = checkpoint_io['write_count']-write_count_before
        checkpoint_io['current_write_duration_unrecorded'] = True
        checkpoint_io['write_seconds_complete'] = False
        checkpoint_io['segment_write_seconds'] = checkpoint_io['write_seconds']-write_seconds_before
        timer = perf_counter()
        write_checkpoint(path, {
            'identities': identities(model, result.tree, result.pilot),
            'scientific_config': scientific_settings(config),
            'capacities': tuple(range(1, min(model.n, config.max_clusters)+1)),
            'tree': {'edges': result.tree.edges, 'n': result.tree.n,
                     'mutation_ids': result.tree.mutation_ids, 'kind': result.tree.kind},
            'pilot': result.pilot, 'bank': snapshot_bank(bank),
            'finalization_state': state, 'scalar_archive': scalar_archive,
            'diagnostics': result.diagnostics,
        }, schema=POLISH_SCHEMA)
        checkpoint_io['write_seconds'] += perf_counter()-timer
        checkpoint_io['segment_write_seconds'] = checkpoint_io['write_seconds']-write_seconds_before
        checkpoint_io['current_write_duration_unrecorded'] = False
        checkpoint_io['write_seconds_complete'] = checkpoint_io['unrecorded_duration_count'] == 0

    # Even a first-visit failure leaves the validated imported/discovery states.
    if not path.exists():
        persist(saved_state)
    refined = finalize_candidates(model, bank, max_visits=config.polish_max_visits,
        diverse_starts=config.polish_diverse_starts, joint_limit=config.polish_joint_limit,
        joint_moves=config.polish_joint_moves, scientific_config=scientific_settings(config),
        checkpoint_state=saved_state, persist=persist)
    result.selected = refined['selected']
    result.candidate_bank = [*scalar_archive, *bank.candidates]
    result.diagnostics['finalization'] = refined['diagnostics']
    result.diagnostics['finalization']['checkpoint_io'] = checkpoint_io
    result.diagnostics['finalization_checkpoint'] = str(path)
    result.diagnostics['finalization_required'] = True
    result.diagnostics['finalization_scientific_config'] = scientific_settings(config)
    after = {}
    for candidate in bank.candidates:
        q = len(candidate.refit.centers)
        after[q] = min(after.get(q, float('inf')), candidate.score)
    result.diagnostics['post_finalization_best_score_per_k'] = after
    persist(refined['state'])
    if not refined['diagnostics']['complete']:
        raise FinalizationIncomplete(result, path)
    return result
