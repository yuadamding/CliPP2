"""Authoritative Version 1.2 cohort replay and prescribed semantic RNG streams.

Semantic hashes bind parsed scientific content and map declared RNG keys to
seeds. Realized tip fractions are consumed exactly, never resampled.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from .biology import History, _Copies, history_content_id
from CliPP2.io.data import content_hash, freeze_record


@dataclass(frozen=True)
class Cohort:
    settings: dict
    path: str
    content_id: str
    trees: dict = field(init=False)
    variants: dict = field(init=False)
    histories: dict = field(init=False)
    _physical_cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        settings = freeze_record(self.settings)
        object.__setattr__(self, 'settings', settings)
        object.__setattr__(self, 'trees', settings['trees'])
        object.__setattr__(self, 'variants', MappingProxyType({row['id']: row for row in settings['variants']}))
        object.__setattr__(self, 'histories', MappingProxyType({row['history_id']: row for row in settings['histories']}))

    @property
    def region_ids(self):
        return tuple(self.settings['region_order'])

    @property
    def master_seed(self):
        return self.settings['master_seed']

    def __getitem__(self, name):
        return self.settings[name]


def semantic_seed(cohort, *keys):
    """The supplied seed rule, independent of row order, scheduling and files."""
    settings = cohort.settings if isinstance(cohort, Cohort) else cohort
    payload = json.dumps([*settings['seed_prefix'], *keys], separators=(',', ':'), ensure_ascii=True)
    return int.from_bytes(hashlib.sha256(payload.encode('utf-8')).digest()[:16], 'big')


def _check_numpy(cohort, enforce):
    wanted = cohort.settings['numpy_version']
    if enforce and np.__version__ != wanted:
        raise RuntimeError(f'Authoritative cohort acquisition/coordinates require NumPy {wanted}; '
                           f'current runtime is {np.__version__}. Use the pinned study environment.')


def cohort_content_id(settings):
    """Bind all parsed biological settings; formatting and object order are inert."""
    return content_hash({"schema": "clipp2.cohort-content.v1", "settings": settings})


def load_cohort(path, *, expected_content_id=None):
    """Load and index the supplied settings without modifying its contents."""
    path = Path(path).resolve()
    with path.open(encoding='utf-8') as stream:
        settings = json.load(stream)
    identity = cohort_content_id(settings)
    if expected_content_id is not None and identity != expected_content_id:
        raise ValueError("Authoritative cohort content differs from its declared identity")
    if (settings.get('schema_version') != 'clipp2.phylogeny.v1'
            or settings.get('simulation_plan_version') != '1.2'
            or settings.get('workflow_protocol') != '2.1'
            or settings.get('rng') != 'NumPy PCG64'):
        raise ValueError('Expected the authoritative phylogeny.v1 / plan1.2 / Protocol2.1 cohort')
    if settings.get('seed_prefix') != ['clipp2-tree-v1', settings['master_seed']]:
        raise ValueError('Unrecognized semantic seed prefix')
    variants = {row['id']: row for row in settings['variants']}
    histories = {row['history_id']: row for row in settings['histories']}
    validation = settings['validation']
    if len(variants) != len(settings['variants']) or len(variants) != validation['variants']:
        raise ValueError('Variant identities are incomplete or duplicated')
    if len(histories) != len(settings['histories']):
        raise ValueError('History identities are duplicated')
    counts = Counter(row['split'] for row in histories.values())
    if counts != {'main': validation['main_histories'], 'development': validation['development_histories']}:
        raise ValueError('Realized split sizes disagree with the supplied validation record')
    for history in histories.values():
        variant = variants[history['variant_id']]
        tree = settings['trees'][variant['tree_id']]
        fractions = np.asarray(history['tip_fractions'], float)
        if (fractions.shape != (len(tree['sampled_tips']), len(settings['region_order']))
                or not np.isfinite(fractions).all() or np.any(fractions < 0)):
            raise ValueError('Invalid realized tip fractions')
        np.testing.assert_allclose(fractions.sum(axis=0), 1., atol=1e-12, rtol=0)
        if not set(history['acquisition_ids']) <= set(settings['acquisitions']):
            raise ValueError('Unknown prescribed acquisition')
    return Cohort(settings, str(path), identity)


def _physical_variant(cohort, variant_id):
    if variant_id in cohort._physical_cache:
        return cohort._physical_cache[variant_id]
    variant = cohort.variants[variant_id]
    tree = cohort.trees[variant['tree_id']]
    logs = {row['segment_id']: row['events_by_birth_node'] for row in variant['event_logs']}
    authoritative = {row['segment_id']: row for row in variant['validated_copy_truth']}
    segments = tuple(sorted(logs))
    if len(segments) != cohort.settings['validation']['segments_per_history'] or set(logs) != set(authoritative):
        raise ValueError('Event logs and validated segment truth differ')
    parents = tree['parent']
    roots = [node for node, parent in parents.items() if parent is None]
    if len(roots) != 1:
        raise ValueError('Physical phylogeny requires one root')
    engine = _Copies(segments)
    root = roots[0]
    engine.states[root] = {segment: [] for segment in segments}
    engine.tree.append(dict(clone_id=root, parent_id=None, node_time=tree['node_time'][root]))
    for segment in segments:
        for homolog in (1, 2):
            engine.copy(root, segment, homolog, kind='initial_copy')
    batch_records, mutation_records, seen_wgd = {}, {}, set()
    remaining = [node for node in parents if node != root]
    while remaining:
        ready = [node for node in remaining if parents[node] in engine.states]
        if not ready:
            raise ValueError('Cyclic or disconnected phylogeny')
        for node in ready:
            engine.fork(parents[node], node)
            engine.tree[-1]['node_time'] = tree['node_time'][node]
            for segment in segments:
                for event in logs[segment].get(node, ()):
                    op = event['op']
                    common = dict(source_event_time=event['event_time'], source_birth_node=node)
                    if op in ('snv_batch', 'duplicate_copy'):
                        homolog = int(event['homolog'][1:])
                        copies = [copy_id for copy_id in engine.states[node][segment]
                                  if engine.copies[copy_id]['homolog'] == homolog]
                        target = copies[event['copy_index']]
                    if op == 'snv_batch':
                        batch = event['batch_id']
                        if (segment, batch) in batch_records:
                            raise ValueError('A mutation batch was acquired more than once')
                        ids = tuple(f'{segment}.{batch}.{within:03}' for within in range(1, event['count']+1))
                        batch_records[segment, batch] = dict(class_id=event['class_id'], ids=ids)
                        for within, mutation in enumerate(ids, 1):
                            engine.event('snv', mutation_id=mutation, clone_id=node, segment_id=segment,
                                         copy_id=target, class_id=event['class_id'], batch_id=batch, **common)
                            engine.copies[target]['mutation_ids'].add(mutation)
                            mutation_records[mutation] = dict(mutation_id=mutation, class_id=event['class_id'],
                                batch_id=batch, within_batch_index=within, segment_id=segment,
                                birth_node=node, acquisition_time=event['event_time'],
                                parental_homolog=event['homolog'], acquisition_copy_index=event['copy_index'])
                    elif op == 'duplicate_copy':
                        engine.copy(node, segment, homolog, target, 'gain_copy')
                        engine.events[-1].update(common, source_copy_index=event['copy_index'])
                    elif op == 'WGD':
                        identifier = node, event['genome_event_id']
                        if identifier not in seen_wgd:
                            engine.event('wgd', clone_id=node, genome_event_id=event['genome_event_id'], **common)
                            seen_wgd.add(identifier)
                        for copy_id in tuple(engine.states[node][segment]):
                            engine.copy(node, segment, engine.copies[copy_id]['homolog'], copy_id, 'wgd_copy')
                            engine.events[-1].update(common, genome_event_id=event['genome_event_id'])
                    else:
                        raise ValueError(f'Unsupported authoritative physical event: {op}')
            remaining.remove(node)
    class_order = {name: i for i, name in enumerate(variant['mutation_class_order'])}
    declared_mutations = tuple(sorted(mutation_records, key=lambda m: (
        mutation_records[m]['segment_id'], class_order[mutation_records[m]['class_id']],
        mutation_records[m]['batch_id'], mutation_records[m]['within_batch_index'])))
    for rank, mutation in enumerate(declared_mutations, 1):
        mutation_records[mutation]['global_rank'] = rank
    # Numerical axes match the canonical input loader; declared rank remains
    # authoritative for coordinates and observation interventions.
    mutations = tuple(sorted(declared_mutations))
    if len(mutations) != cohort.settings['validation']['mutations_per_history']:
        raise ValueError('Physical event replay did not produce the prescribed mutation count')
    classes = Counter(mutation_records[mutation]['class_id'] for mutation in mutations)
    if any(classes[name] != count for name, count in variant['assayed_counts'].items()):
        raise ValueError('Physical event replay disagrees with assayed class allocation')
    tips = tuple(tree['sampled_tips'])
    index = {mutation: i for i, mutation in enumerate(mutations)}
    counts = np.zeros((len(mutations), len(tips)), int)
    cn = np.zeros((len(segments), len(tips), 2), int)
    for s, segment in enumerate(segments):
        for j, tip in enumerate(tips):
            for copy_id in engine.states[tip][segment]:
                copy = engine.copies[copy_id]
                cn[s, j, copy['homolog']-1] += 1
                for mutation in copy['mutation_ids']:
                    counts[index[mutation], j] += 1
            np.testing.assert_array_equal(cn[s, j], authoritative[segment]['cn_by_tip'][tip],
                                          err_msg=f'Final homolog CN differs for {variant_id}/{segment}/{tip}')
        actual_batches = {batch: record for (seg, batch), record in batch_records.items() if seg == segment}
        expected_batches = authoritative[segment]['mutation_batches']
        if set(actual_batches) != set(expected_batches):
            raise ValueError('Replayed and validated mutation batches differ')
        for batch, expected in expected_batches.items():
            record = actual_batches[batch]
            if record['class_id'] != expected['class_id'] or len(record['ids']) != expected['count']:
                raise ValueError('Replayed mutation batch definition differs')
            for mutation in record['ids']:
                np.testing.assert_array_equal(counts[index[mutation]], expected['multiplicity_by_tip'],
                    err_msg=f'Final mutant copies differ for {variant_id}/{mutation}')
    payload = dict(mutations=mutations, declared_mutations=declared_mutations,
        segments=segments, tips=tips, counts=counts, cn=cn,
        mutation_records=mutation_records, tree=tuple(engine.tree), events=tuple(engine.events),
        copies=tuple(dict(record, mutation_ids=tuple(sorted(record['mutation_ids']))) for record in engine.copies.values()))
    cohort._physical_cache[variant_id] = payload
    return payload


def cohort_history(cohort, history_id, *, enforce_numpy_version=True):
    """Replay the supplied physical events and retain the realized mixture exactly."""
    _check_numpy(cohort, enforce_numpy_version)
    realized = cohort.histories[history_id]
    variant = cohort.variants[realized['variant_id']]
    physical = _physical_variant(cohort, realized['variant_id'])
    mutations, segments, tips = (physical[name] for name in ('mutations', 'segments', 'tips'))
    fractions = np.asarray(realized['tip_fractions'], float)
    counts, cn = physical['counts'], physical['cn']
    records = deepcopy(physical['mutation_records'])
    n, r = len(mutations), len(cohort.region_ids)
    phi = (counts > 0) @ fractions
    multiplicity = np.full((n, r), np.nan)
    major, minor = np.empty((n, r), int), np.empty((n, r), int)
    segment_index = {segment: i for i, segment in enumerate(segments)}
    endpoint_normalizations = {}
    for i, mutation in enumerate(mutations):
        s = segment_index[records[mutation]['segment_id']]
        all_carriers = np.zeros(r, bool)
        for j in range(r):
            present = fractions[:, j] > 0
            alleles = np.sort(cn[s, present], axis=1)
            if not np.all(alleles == alleles[0]):
                raise ValueError('Authoritative history is outside single-state regional CN scope')
            minor[i, j], major[i, j] = alleles[0]
            carriers = present & (counts[i] > 0)
            if carriers.any():
                integer = np.unique(counts[i, carriers])
                if len(integer) != 1 or not 1 <= integer[0] <= major[i, j]:
                    raise ValueError('Authoritative carrier multiplicity is outside scalar integer scope')
                multiplicity[i, j] = integer[0]
                if carriers.sum() == present.sum():
                    all_carriers[j] = True
                    phi[i, j] = 1.
        supplied_phi = np.asarray(realized['true_ccfs'][records[mutation]['class_id']], float)
        above_one = supplied_phi > 1
        if (not np.isfinite(supplied_phi).all() or np.any(supplied_phi < 0)
                or np.any(above_one & (~all_carriers | (supplied_phi > 1+1e-12)))):
            raise ValueError('Supplied realized CCF must be finite and in [0,1]')
        for j in np.flatnonzero(above_one):
            key = records[mutation]['class_id'], cohort.region_ids[j]
            normalization = endpoint_normalizations.setdefault(key, dict(history_id=history_id,
                class_id=key[0], region_id=key[1], supplied_value=float(supplied_phi[j]),
                supplied_float_hex=float(supplied_phi[j]).hex(), canonical_value=1.,
                reason='Upper endpoint roundoff; every positive-fraction genotype carries the mutation',
                affected_mutation_ids=[]))
            normalization['affected_mutation_ids'].append(mutation)
            supplied_phi[j] = 1.
        np.testing.assert_allclose(phi[i], supplied_phi, atol=1e-12, rtol=0)
        # Keep the supplied numerical realization after independently proving
        # its physical meaning; its bits define the prescribed read model.
        phi[i] = supplied_phi
    coordinates = cohort.settings['segment_coordinates']
    length = coordinates['length_bp']
    for s, segment in enumerate(segments):
        ids = [mutation for mutation in physical['declared_mutations'] if records[mutation]['segment_id'] == segment]
        generator = np.random.Generator(np.random.PCG64(semantic_seed(cohort,
            realized['split'], realized['variant_id'], realized['replicate'], 'coordinates', segment)))
        offsets = np.sort(generator.choice(length, size=len(ids), replace=False, shuffle=True))
        for mutation, offset in zip(ids, offsets):
            records[mutation].update(chromosome=coordinates['chromosome'], position=int(s*length+1+offset),
                                     segment_start=s*length+1, segment_end=(s+1)*length)
    vectors = sorted(set(tuple(np.round(row, 12)) for row in phi))
    classes = {name: tuple(m for m in mutations if records[m]['class_id'] == name)
               for name in variant['mutation_class_order']}
    batches = {name: tuple(m for m in mutations if records[m]['batch_id'] == name)
               for name in sorted({row['batch_id'] for row in records.values()})}
    reference = variant.get('controlled_reference')
    sources = (classes.get('T', ()) if reference == 'all_T' else batches.get('T_low', ())
               if reference == 'T_low_only' else batches.get('T_G', ()) if reference == 'T_G_only' else ())
    source_set = set(sources)
    metadata = dict(schema='clipp2.simulation.authoritative-history.v1', study_version='1.2',
        cohort_content_id=cohort.content_id,
        cohort_schema=cohort.settings['schema_version'], history_id=history_id, family=variant['family'],
        variant_id=variant['id'], split=realized['split'], replicate=realized['replicate'],
        seed=int(realized['seed128']), master_seed=cohort.master_seed, tree_id=variant['tree_id'],
        recipe=variant['recipe'], mutation_classes=classes, mutation_batches=batches,
        mutation_records=tuple(records[m] for m in mutations), mutation_class_order=variant['mutation_class_order'],
        declared_mutation_order=physical['declared_mutations'],
        original_mutation_order=physical['declared_mutations'],
        prescribed_rank_by_id={m: records[m]['global_rank'] for m in mutations},
        controlled_reference=reference, declared_source_ids=sources,
        declared_remainder_ids=tuple(m for m in mutations if m not in source_set),
        controlled_source_status='declared_diagnostic_source_not_an_expected_fitted_partition',
        observation_policy=variant.get('observation_policy'), acquisition_ids=realized['acquisition_ids'],
        normal_cn=cohort.trees[variant['tree_id']]['normal_cn'],
        ccf_endpoint_normalizations=tuple(endpoint_normalizations.values()),
        supplied_ccf_endpoint_policy='Preserve valid supplied bits; normalize only proven all-carrier upper roundoff',
        numpy_version=np.__version__, declared_numpy_version=cohort.settings['numpy_version'],
        mutation_key_convention='segment.batch.within_batch_3digit', genome_build='synthetic_1Mb_segments',
        physical_replay_validated=True, supplied_final_copy_truth_verified=True,
        interval_map={segment: dict(chromosome=coordinates['chromosome'], start=s*length+1, end=(s+1)*length)
                      for s, segment in enumerate(segments)},
        limitations=['Core cohort is supplied constructed biological truth, not patient prevalence.',
                    'Realized tip fractions were not resampled; controlled source definitions are not fitted labels.'])
    return History(mutations, cohort.region_ids, segments,
        np.array([[records[m]['segment_id']]*r for m in mutations]), phi, multiplicity, major, minor,
        tips, fractions, counts, cn, tuple(records[m]['birth_node'] for m in mutations),
        tuple(f'g{vectors.index(tuple(np.round(row, 12))):03}' for row in phi), deepcopy(physical['tree']),
        tuple(dict(clone_id=tip, sampled=True, fractions=fractions[j].tolist()) for j, tip in enumerate(tips)),
        deepcopy(physical['events']), deepcopy(physical['copies']), metadata)


def acquire_cohort(history, cohort, acquisition_id, *, read_replicate_id=1, matched_R4_mask=False,
                   enforce_numpy_version=True):
    """Generate every pair from its prescribed fresh PCG64 semantic seed."""
    _check_numpy(cohort, enforce_numpy_version)
    metadata = history.metadata
    if metadata.get('cohort_content_id') != cohort.content_id:
        raise ValueError('Physical history and acquisition cohort content identities differ')
    realized = cohort.histories[history.history_id]
    if acquisition_id not in realized['acquisition_ids']:
        raise ValueError('Acquisition is not prescribed for this history split')
    if type(read_replicate_id) is not int or read_replicate_id < 1:
        raise ValueError('Read replicate identity must be a positive integer')
    setting = cohort.settings['acquisitions'][acquisition_id]
    n, r = history.phi.shape
    rho = float(setting['purity'])
    normal = np.full((n, r), metadata['normal_cn'], float)
    burden = history.mutant_copy_counts @ history.clone_fractions
    denominator = (1-rho)*normal + rho*(history.major_cn+history.minor_cn)
    physical_probability = rho*burden/denominator
    reduced = rho*np.nan_to_num(history.multiplicity)*history.phi/denominator
    np.testing.assert_allclose(physical_probability, reduced, atol=1e-12, rtol=0)
    if np.any(reduced < 0) or np.any(reduced > 1):
        raise ValueError('Physical read probability is outside [0,1]')
    theta = reduced
    latent_depth, latent_alt = np.empty((n, r), int), np.empty((n, r), int)
    prefix = (metadata['split'], metadata['variant_id'], metadata['replicate'], 'reads', acquisition_id, read_replicate_id)
    for i, mutation in enumerate(history.mutation_ids):
        for j, region in enumerate(history.region_ids):
            depth_rng = np.random.Generator(np.random.PCG64(semantic_seed(cohort, *prefix, 'depth', mutation, region)))
            alt_rng = np.random.Generator(np.random.PCG64(semantic_seed(cohort, *prefix, 'alt', mutation, region)))
            latent_depth[i, j] = depth_rng.poisson(setting['mean_depth'])
            latent_alt[i, j] = alt_rng.binomial(latent_depth[i, j], theta[i, j])
    observed, excluded, zero_depth = np.ones((n, r), bool), np.zeros((n, r), bool), np.zeros((n, r), bool)
    policy = metadata['observation_policy']
    rank = np.asarray([row['global_rank'] for row in metadata['mutation_records']])
    if policy == 'deterministic_missing_excluded_zero_depth':
        observed[rank % 5 == 0, history.region_ids.index('R2')] = False
        excluded[rank % 10 == 0, history.region_ids.index('R3')] = True
        zero_depth[rank % 10 == 1, history.region_ids.index('R4')] = True
        observed[rank <= 30], excluded[rank <= 30], zero_depth[rank <= 30] = False, False, False
    elif policy == 'mask_nontrunk_R4':
        observed[[row['class_id'] in ('A', 'B', 'C') for row in metadata['mutation_records']], history.region_ids.index('R4')] = False
    elif policy not in (None, 'observed_zero_ALT_for_B_R1', 'no_assayed_trunk', 'monoclonal_empty_remainder_control'):
        raise ValueError('Unsupported authoritative observation policy')
    if matched_R4_mask:
        fixture = cohort.settings['optional_diagnostic_fixtures']['matched_R4_mask']
        if metadata['variant_id'] not in fixture['variants']:
            raise ValueError('Matched R4-mask fixture is not declared for this variant')
        sizes = Counter((row['segment_id'], row['batch_id']) for row in metadata['mutation_records'])
        selected = [row['class_id'] in ('T', 'A') and row['within_batch_index'] <= sizes[row['segment_id'], row['batch_id']]//2
                    for row in metadata['mutation_records']]
        observed[selected, history.region_ids.index('R4')] = False
    depth, alt = np.where(zero_depth, 0, latent_depth), np.where(zero_depth, 0, latent_alt)
    records = [dict(mutation_id=mutation, sample_id=region,
        alt_count=int(alt[i, j]) if observed[i, j] else '.',
        ref_count=int(depth[i, j]-alt[i, j]) if observed[i, j] else '.', count_observed=int(observed[i, j]),
        purity=rho, normal_cn=int(normal[i, j]), segment_id=str(history.segment_ids[i, j]),
        cn_state_id='0', cn_state_fraction=1., allele_a_cn=int(history.major_cn[i, j]),
        allele_b_cn=int(history.minor_cn[i, j]), mapping_genuine=1, normal_cn_supplied=1,
        input_excluded=int(excluded[i, j]), chromosome_index=metadata['mutation_records'][i]['chromosome'],
        position=metadata['mutation_records'][i]['position'], mutation_rank=int(rank[i]))
        for i in np.argsort(rank) for mutation in (history.mutation_ids[i],)
        for j, region in enumerate(history.region_ids)]
    frame = pd.DataFrame(records)
    eligible = (history.major_cn <= 10) & ~excluded
    identity = history_content_id(history)
    frame.attrs['metadata'] = dict(tumor_id=f'{history.history_id}.{acquisition_id}.read{read_replicate_id}',
        history_id=history.history_id, history_content_id=identity,
        cohort_content_id=cohort.content_id,
        source_cohort_protocol_version=cohort.settings['workflow_protocol'],
        genome_build='synthetic_1Mb_segments', coordinate_system='1-based-inclusive',
        mutation_key_convention='segment.batch.within_batch_3digit')
    frame.attrs['acquisition'] = dict(schema='clipp2.simulation.cohort-acquisition.v1', study_version='1.2',
        cohort_content_id=cohort.content_id, history_content_id=identity,
        mutation_ids=history.mutation_ids, region_ids=history.region_ids,
        history_id=history.history_id, variant_id=metadata['variant_id'], split=metadata['split'],
        replicate=metadata['replicate'], acquisition_id=acquisition_id, read_replicate_id=read_replicate_id,
        seed_derivation=cohort.settings['seed_derivation'], numpy_version=np.__version__,
        declared_numpy_version=cohort.settings['numpy_version'], mean_depth=setting['mean_depth'],
        true_purity=np.full((n, r), rho), supplied_purity=np.full((n, r), rho), normal_cn=normal,
        true_probability=theta, read_mean_probability=theta, sampled_probability=theta,
        alt_counts=np.where(observed, alt, 0), depth=np.where(observed, depth, 0),
        total_counts=np.where(observed, depth, 0), latent_alt_counts=latent_alt, latent_depth=latent_depth,
        count_observed=observed, input_excluded=excluded, forced_zero_depth=zero_depth,
        supplied_major_cn=history.major_cn, supplied_minor_cn=history.minor_cn,
        supplied_segment_ids=history.segment_ids, mapping_genuine=np.ones((n, r), bool),
        original_eligible=eligible, input_loss=(~observed)|(depth == 0)|(~eligible),
        biological_truth_available=np.ones((n, r), bool), positive_carrier=history.phi > 0,
        inherited_support_exclusion=np.zeros((n, r), bool), cn_error=np.zeros((n, r), bool),
        segment_mapping_error=np.zeros((n, r), bool), observation_policy=policy,
        optional_matched_R4_mask=bool(matched_R4_mask))
    return frame
