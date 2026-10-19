"""Explicit, checksum-authorized cross-source candidate import (never resume).

Historical checkpoints remain untouched. Only selected saved states are
independently validated and retained; no full-archive-retention claim is made.
No code, pickles, or arbitrary classes are loaded from an imported document.
"""
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .checkpoint import SCHEMA, decode, restore_bank
from .selection import TreeFit, _CandidateBank, _score_key
from ..kernel.topology import FrozenTree

IMPORT_SCHEMA = 'clipp2_import_authority_v1'
POLISH_SCHEMA = 'clipp2_import_polish_checkpoint_v1'


def scientific_settings(config):
    return {name: getattr(config, name) for name in (
        'max_major_cn', 'max_clusters', 'regional_zero_policy', 'selection_objective')}


def checked_document(path, *, expected_sha256=None, schemas=(SCHEMA, POLISH_SCHEMA)):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError('Checkpoint must be a regular non-symlink file')
    blob = path.read_bytes()
    actual = hashlib.sha256(blob).hexdigest()
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError('Imported checkpoint file checksum mismatch')
    document = json.loads(blob)
    if document.get('schema') not in schemas:
        raise ValueError('Unsupported import/polish checkpoint schema')
    encoded = json.dumps(document['payload'], sort_keys=True, separators=(',', ':'), allow_nan=False)
    if hashlib.sha256(encoded.encode()).hexdigest() != document.get('sha256'):
        raise ValueError('Imported checkpoint payload checksum mismatch')
    return decode(document['payload']), actual, document['schema']


def import_candidates(model, checkpoint, authority_file, config, *, input_sha256):
    """Verify provenance and rescore finalist states against current input/model."""
    from .polish import finalist_indices
    started = perf_counter()
    authority_path = Path(authority_file)
    if not authority_path.is_file() or authority_path.is_symlink():
        raise ValueError('Import authority must be a regular non-symlink file')
    authority_blob = authority_path.read_bytes()
    authority = json.loads(authority_blob)
    if authority.get('schema') != IMPORT_SCHEMA:
        raise ValueError('A versioned candidate-import authority is required')
    required = ('checkpoint_sha256', 'source_identity', 'source_commit', 'source_sha256',
                'input_sha256', 'model_identity', 'tree_identity', 'pilot_sha256')
    if any(not isinstance(authority.get(k), str) or not authority[k] for k in required):
        raise ValueError('Import authority is missing a source/scientific identity')
    if authority.get('scientific_config') != scientific_settings(config):
        raise ValueError('Imported scientific settings differ from explicit new configuration')
    if input_sha256 != authority['input_sha256'] or model.identity != authority['model_identity']:
        raise ValueError('Imported input/model identity mismatch')
    state, file_hash, schema = checked_document(checkpoint, expected_sha256=authority['checkpoint_sha256'])
    identity = state['identities']
    expected = {'source': authority['source_identity'], 'model': model.identity,
                'objective': config.selection_objective, 'zero_policy': config.regional_zero_policy,
                'tree': authority['tree_identity'], 'pilot': authority['pilot_sha256'],
                'mutation_ids': tuple(model.mutation_ids)}
    for key, value in expected.items():
        if identity.get(key) != value:
            raise ValueError(f'Imported checkpoint {key} differs from authority/current model')
    binding = identity.get('execution_binding') or {}
    for key in ('source_commit', 'source_sha256', 'input_sha256'):
        if binding.get(key) != authority[key]:
            raise ValueError(f'Imported execution binding {key} mismatch')
    capacities = tuple(state['capacities'])
    if capacities != tuple(range(1, min(model.n, config.max_clusters)+1)):
        raise ValueError('Imported cluster capacities must match the explicit complete 1..K range')
    tree = FrozenTree(**state['tree'])
    pilot = np.asarray(state['pilot'], dtype=float)
    if (tree.identity != authority['tree_identity'] or tree.mutation_ids != tuple(model.mutation_ids)
            or pilot.shape != (model.n, model.r) or not np.isfinite(pilot[model.observed]).all()
            or hashlib.sha256(np.asarray(pilot, dtype='<f8').tobytes()).hexdigest() != authority['pilot_sha256']):
        raise ValueError('Imported tree/pilot does not match its bound identities')
    # This materializes compact records, not all dense mutation-label vectors.
    # Restored records are untrusted proposals until individually revalidated.
    old = restore_bank(model, tree, state['bank'])
    if not old.candidates:
        raise ValueError('Imported bank contains no saved candidates')
    if any(not c.refit.eligible or not np.isfinite(c.score)
           or len(c.refit.centers) not in capacities for c in old.candidates):
        raise ValueError('Imported bank has ineligible/nonfinite/out-of-capacity records')
    chosen = finalist_indices(old, diverse_starts=config.polish_diverse_starts)
    transfers = authority.get('reference_transfers', [])
    if not isinstance(transfers, list):
        raise ValueError('Reference transfers must be an explicit list')
    for transfer in transfers:
        required_transfer = {'old_bank_index', 'mutation_id', 'source_cluster',
                             'destination_cluster', 'expected_score_before', 'expected_score_after'}
        if not isinstance(transfer, dict) or set(transfer) != required_transfer:
            raise ValueError('Reference transfer must bind parent, mutation, labels and both scores')
        index = transfer['old_bank_index']
        if type(index) is not int or not 0 <= index < len(old.candidates):
            raise ValueError('Reference transfer parent index is invalid')
        if index not in chosen:
            chosen.append(index)
    bank = _CandidateBank(model, tree, compact_general=True)
    records = []
    for index in chosen:
        candidate = old.candidates[index]
        fit = candidate.refit
        if fit.selection_objective != config.selection_objective:
            raise ValueError('Imported state has a different scoring objective')
        imported = bank.retain_validated_fixed_fit(fit.labels, fit.centers, fit.clonal_cluster,
            {'route': 'validated_cross_source_import', 'requested_k': len(fit.centers),
             'old_bank_index': index, 'old_checkpoint_sha256': file_hash,
             'old_source_commit': authority['source_commit'], 'old_provenance': candidate.provenance},
            expected_score=candidate.score)
        records.append({'old_bank_index': index, 'occupied_k': len(fit.centers),
                        'old_score': candidate.score, 'recomputed_score': imported.score,
                        'new_bank_index': bank.candidates.index(imported)})
    references = []
    for transfer in transfers:
        index = transfer['old_bank_index']
        fit = old.candidates[index].fit
        q = len(fit.centers)
        source, destination = (transfer[key] for key in ('source_cluster', 'destination_cluster'))
        if (any(type(k) is not int or not 0 <= k < q for k in (source, destination))
                or source == destination or transfer['mutation_id'] not in model.mutation_ids):
            raise ValueError('Reference transfer has invalid mutation/cluster identities')
        before, after = (transfer[key] for key in ('expected_score_before', 'expected_score_after'))
        if (any(isinstance(value, bool) or not isinstance(value, (int, float))
                or not np.isfinite(value) for value in (before, after))
                or not np.isclose(before, fit.score, atol=1e-7, rtol=1e-10)
                or after >= before):
            raise ValueError('Reference transfer must bind an improving independently checkable score')
        row = model.mutation_ids.index(transfer['mutation_id'])
        labels = np.asarray(fit.labels).copy()
        if labels[row] != source:
            raise ValueError('Reference transfer source does not match the saved membership')
        labels[row] = destination
        if not np.array_equal(np.unique(labels), np.arange(q)):
            raise ValueError('Reference transfer would change occupied cluster count')
        # The standard fixed-fit admission independently checks every final
        # cluster's regional support, domains, anchor and raw-input score.
        reference = bank.retain_validated_fixed_fit(labels, fit.centers, fit.clonal_cluster,
            {'route': 'validated_reference_transfer', 'requested_k': q,
             'old_checkpoint_sha256': file_hash, **transfer}, expected_score=after)
        parent_score = next(record['recomputed_score'] for record in records
                            if record['old_bank_index'] == index)
        if reference.score >= parent_score:
            raise ValueError('Reference transfer does not improve its independently recomputed parent')
        references.append({**transfer, 'recomputed_score': reference.score,
                           'new_bank_index': bank.candidates.index(reference),
                           'centers_refitted': False})
    info = {'schema': 'clipp2_validated_candidate_import_v1', 'source_checkpoint_schema': schema,
            'source_checkpoint_sha256': file_hash, 'source_checkpoint': str(Path(checkpoint).resolve()),
            'authority_sha256': hashlib.sha256(authority_blob).hexdigest(),
            'old_source_commit': authority['source_commit'], 'old_source_identity': authority['source_identity'],
            'scientific_config': scientific_settings(config), 'old_candidate_count': len(old.candidates),
            'validated_import_count': len(records), 'full_archive_retained_across_versions': False,
            'original_archive_preserved': True, 'records': records,
            'validated_reference_transfers': references,
            'seconds': perf_counter()-started}
    selected = min(bank.candidates, key=_score_key)
    return TreeFit(selected, bank.candidates, tree, pilot,
        {'import': info, 'capacities': capacities, 'tree_identity': tree.identity,
         'search_policy': {'mode': 'import_and_polish', 'discovery_executed': False},
         'native_continuation_executed': False, 'global_optimality_certified': False}, bank=bank)
