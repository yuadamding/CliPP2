"""Physical-copy histories for the twelve Version 1.1 biological study families.

These are designed mechanisms, not a model of their prevalence in patients.
All inference truth is derived from the final physical copies, never from caps.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from CliPP2.io.data import content_hash

FAMILIES = {
    'S01': 'diploid_single_copy', 'S02': 'shared_gain_valid_cap',
    'S03': 'mixed_clonal_timing', 'S04': 'parallel_gain_false_clonality',
    'S05': 'mixed_clonal_and_new_targets', 'S06': 'whole_genome_duplication',
    'S07': 'regionally_clonal', 'S08': 'near_clonal_selection',
    'S09': 'contributor_density', 'S10': 'loss_and_observation_edges',
    'S11': 'incomplete_reference', 'S12': 'multiplicity_ccf_aliases',
}


def _plain(value):
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def history_content_id(history):
    """Identify exact retained biological content without modifying the history."""
    return content_hash({'schema': 'clipp2.history-content.v1', 'history': history.to_dict()})


@dataclass(frozen=True)
class History:
    mutation_ids: tuple[str, ...]
    region_ids: tuple[str, ...]
    segment_keys: tuple[str, ...]
    segment_ids: np.ndarray
    phi: np.ndarray
    multiplicity: np.ndarray
    major_cn: np.ndarray
    minor_cn: np.ndarray
    clone_ids: tuple[str, ...]
    clone_fractions: np.ndarray
    mutant_copy_counts: np.ndarray
    copy_cn: np.ndarray
    mutation_branches: tuple[str, ...]
    observable_truth_ids: tuple[str, ...]
    tree: tuple[dict, ...]
    clones: tuple[dict, ...]
    events: tuple[dict, ...]
    copies: tuple[dict, ...]
    metadata: dict

    def __post_init__(self):
        for name in ('mutation_ids', 'region_ids', 'segment_keys', 'clone_ids',
                     'mutation_branches', 'observable_truth_ids', 'tree', 'clones', 'events', 'copies'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in ('segment_ids', 'phi', 'multiplicity', 'major_cn', 'minor_cn',
                     'clone_fractions', 'mutant_copy_counts', 'copy_cn'):
            dtype = str if name == 'segment_ids' else float if name in ('phi', 'multiplicity', 'clone_fractions') else int
            value = np.ascontiguousarray(getattr(self, name), dtype=dtype)
            object.__setattr__(self, name, np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape))
        self.validate()

    @property
    def family(self):
        return self.metadata['family']

    @property
    def history_id(self):
        return self.metadata['history_id']

    def validate(self):
        n, r, v, s = len(self.mutation_ids), len(self.region_ids), len(self.clone_ids), len(self.segment_keys)
        assert n and r and s and len(set(self.mutation_ids)) == n
        assert len(set(self.region_ids)) == r and len(set(self.segment_keys)) == s
        assert self.segment_ids.shape == self.phi.shape == self.multiplicity.shape == self.major_cn.shape == self.minor_cn.shape == (n, r)
        assert self.clone_fractions.shape == (v, r) and self.mutant_copy_counts.shape == (n, v)
        assert self.copy_cn.shape == (s, v, 2)
        assert len(self.mutation_branches) == len(self.observable_truth_ids) == n
        assert np.isfinite(self.clone_fractions).all() and (self.clone_fractions >= 0).all()
        np.testing.assert_allclose(self.clone_fractions.sum(axis=0), 1., atol=1e-14, rtol=0)
        assert (self.mutant_copy_counts >= 0).all() and (self.copy_cn >= 0).all()
        copy_lookup = {row['copy_id']: row for row in self.copies}
        assert len(copy_lookup) == len(self.copies)
        acquisition = [row['mutation_id'] for row in self.events if row['event_type'] == 'snv']
        assert len(acquisition) == n and set(acquisition) == set(self.mutation_ids)
        # Reconstruct every final genotype from copy rows, independently of arrays.
        index = {mutation: i for i, mutation in enumerate(self.mutation_ids)}
        segment_index = {segment: i for i, segment in enumerate(self.segment_keys)}
        counts, cn = np.zeros((n, v), int), np.zeros((s, v, 2), int)
        for record in self.copies:
            parent = record['parent_copy_id']
            if parent is not None:
                assert parent in copy_lookup
                assert copy_lookup[parent]['homolog'] == record['homolog']
                assert copy_lookup[parent]['segment_id'] == record['segment_id']
                assert record['creation_index'] > copy_lookup[parent]['creation_index']
            if record['active'] and record['clone_id'] in self.clone_ids:
                j = self.clone_ids.index(record['clone_id'])
                cn[segment_index[record['segment_id']], j, record['homolog']-1] += 1
                for mutation in record['mutation_ids']:
                    counts[index[mutation], j] += 1
        np.testing.assert_array_equal(counts, self.mutant_copy_counts)
        np.testing.assert_array_equal(cn, self.copy_cn)
        # Replay mutation sets in chronological order to verify inheritance/gain.
        state = {}
        for event in self.events:
            kind = event['event_type']
            if kind in ('initial_copy', 'inherit_copy', 'gain_copy', 'wgd_copy', 'loh_copy'):
                parent = event.get('parent_copy_id')
                state[event['copy_id']] = set() if parent is None else set(state[parent])
            elif kind == 'snv':
                state[event['copy_id']].add(event['mutation_id'])
        for copy_id, mutations in state.items():
            assert mutations == set(copy_lookup[copy_id]['mutation_ids'])
        for i in range(n):
            for j in range(r):
                present = self.clone_fractions[:, j] > 0
                segment = segment_index[self.segment_ids[i, j]]
                alleles = np.sort(cn[segment, present], axis=1)
                assert (alleles == alleles[:1]).all(), 'mixed cancer CN is outside the principal study'
                assert (self.minor_cn[i, j], self.major_cn[i, j]) == tuple(alleles[0])
                carriers = present & (counts[i] > 0)
                phi = self.clone_fractions[carriers, j].sum()
                np.testing.assert_allclose(self.phi[i, j], phi, atol=1e-12, rtol=0)
                if carriers.any():
                    multiplicities = np.unique(counts[i, carriers])
                    assert len(multiplicities) == 1, 'carrier multiplicity is not one integer'
                    assert self.multiplicity[i, j] == multiplicities[0]
                    assert 1 <= multiplicities[0] <= self.major_cn[i, j]
                else:
                    assert np.isnan(self.multiplicity[i, j]) and self.phi[i, j] == 0
        burden = self.mutant_copy_counts @ self.clone_fractions
        np.testing.assert_allclose(burden, np.nan_to_num(self.multiplicity)*self.phi, atol=1e-12, rtol=0)
        assert np.all((self.phi >= 0) & (self.phi <= 1))
        return True

    def to_dict(self):
        return _plain(dict(schema='clipp2.simulation.history.v1', **vars(self)))

    @classmethod
    def from_dict(cls, payload):
        payload = dict(payload)
        if payload.pop('schema', None) != 'clipp2.simulation.history.v1':
            raise ValueError('Unsupported physical-history schema')
        return cls(**payload)

    def write_truth(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'history.json').write_text(json.dumps(self.to_dict(), allow_nan=False, separators=(',', ':'))+'\n')
        records = [dict(mutation_id=m, region_id=r, segment_id=self.segment_ids[i, j],
            phi=self.phi[i, j], multiplicity=self.multiplicity[i, j], major_cn=self.major_cn[i, j],
            minor_cn=self.minor_cn[i, j], acquisition_branch=self.mutation_branches[i],
            observable_truth_id=self.observable_truth_ids[i],
            current_carriers=json.dumps([clone for clone, copies in zip(self.clone_ids, self.mutant_copy_counts[i]) if copies > 0]),
            globally_ubiquitous=bool((self.mutant_copy_counts[i] > 0).all()))
            for i, m in enumerate(self.mutation_ids) for j, r in enumerate(self.region_ids)]
        pd.DataFrame(records).to_csv(directory / 'truth_pairs.tsv', sep='\t', index=False, na_rep='NA')
        for name in ('tree', 'clones', 'events', 'copies'):
            records = [{key: json.dumps(_plain(value), separators=(',', ':')) if isinstance(value, (tuple, list, dict)) else value
                        for key, value in row.items()} for row in getattr(self, name)]
            pd.DataFrame(records).to_csv(directory / ('truth_'+name+'.tsv'), sep='\t', index=False, na_rep='NA')
        return directory


class _Copies:
    def __init__(self, segments):
        self.segments, self.states, self.copies, self.events, self.tree = segments, {}, {}, [], []

    def event(self, kind, **values):
        event = dict(event_id=f'e{len(self.events):07}', event_type=kind, **values)
        self.events.append(event)
        return event

    def copy(self, clone, segment, homolog, parent=None, kind='gain_copy'):
        copy_id = f'c{len(self.copies):07}'
        event = self.event(kind, clone_id=clone, segment_id=segment, copy_id=copy_id, parent_copy_id=parent)
        self.copies[copy_id] = dict(copy_id=copy_id, clone_id=clone, segment_id=segment, homolog=homolog,
            parent_copy_id=parent, creation_index=len(self.events)-1, creation_event=event['event_id'],
            active=True, loss_event=None, mutation_ids=set() if parent is None else set(self.copies[parent]['mutation_ids']))
        self.states[clone][segment].append(copy_id)
        return copy_id

    def root(self):
        self.states['trunk'] = {segment: [] for segment in self.segments}
        self.tree.append(dict(clone_id='trunk', parent_id=None))
        for segment in self.segments:
            for homolog in (1, 2):
                self.copy('trunk', segment, homolog, kind='initial_copy')

    def fork(self, parent, clone):
        self.states[clone] = {segment: [] for segment in self.segments}
        self.tree.append(dict(clone_id=clone, parent_id=parent))
        self.event('clone_split', clone_id=clone, parent_clone_id=parent)
        for segment in self.segments:
            for copy_id in self.states[parent][segment]:
                self.copy(clone, segment, self.copies[copy_id]['homolog'], copy_id, 'inherit_copy')

    def first(self, clone, segment, homolog=1):
        return next(c for c in self.states[clone][segment] if self.copies[c]['homolog'] == homolog)

    def snv(self, mutation, clone, segment, homolog=1):
        target = self.first(clone, segment, homolog)
        self.event('snv', mutation_id=mutation, clone_id=clone, segment_id=segment, copy_id=target)
        self.copies[target]['mutation_ids'].add(mutation)

    def gain(self, clone, segment, homolog=1, kind='gain_copy'):
        parent = self.first(clone, segment, homolog)
        self.copy(clone, segment, homolog, parent, kind)

    def wgd(self, clone):
        self.event('wgd', clone_id=clone)
        for segment in self.segments:
            for parent in tuple(self.states[clone][segment]):
                self.copy(clone, segment, self.copies[parent]['homolog'], parent, 'wgd_copy')

    def loh(self, clone, segment):
        self.event('copy_neutral_loh', clone_id=clone, segment_id=segment, lost_homolog=2, retained_homolog=1)
        lost = [c for c in self.states[clone][segment] if self.copies[c]['homolog'] == 2]
        for copy_id in lost:
            event = self.event('deletion', clone_id=clone, segment_id=segment, copy_id=copy_id)
            self.states[clone][segment].remove(copy_id)
            self.copies[copy_id].update(active=False, loss_event=event['event_id'])
            self.gain(clone, segment, 1, 'loh_copy')


def generate_history(family, seed, n_mutations=1500, n_segments=50, n_regions=4, *, major_cn_range=None):
    """Generate event histories first, then derive exact scalar-integer truth.

    Seeds select prespecified within-family mixture/timing variants. Controlled
    source/reference interventions are declared separately from fitted outcomes.
    """
    if family not in FAMILIES:
        raise ValueError(f'Unknown family {family!r}')
    if any(type(v) is not int or v < 1 for v in (n_mutations, n_segments, n_regions)) or n_mutations < 6:
        raise ValueError('Require at least six mutations and positive integer segment/region counts')
    major_cn_range = (3, 6) if major_cn_range is None else tuple(major_cn_range)
    if (len(major_cn_range) != 2 or any(type(value) is not int for value in major_cn_range)
            or not 2 <= major_cn_range[0] < major_cn_range[1]):
        raise ValueError('major_cn_range must be an integer half-open interval at least two')
    rng = np.random.default_rng(seed)
    segments = tuple(f's{i:03}' for i in range(n_segments))
    regions = tuple(f'r{i+1}' for i in range(n_regions))
    mutations = tuple(f'm{i:06}' for i in range(n_mutations))
    variant = int(seed) % 3
    weights = dict(E=.15, R=.20, A=.30, D=.20, B=.15)
    if family == 'S02':
        weights = dict(E=.15, R=.20, A=.35, D=.15, B=.15)
    elif family == 'S04':
        weights = dict(E=0., R=.30, A=.45, D=.15, B=.10)
    elif family == 'S05':
        weights = dict(E=.25, R=.20, A=.35, D=.10, B=.10)
    elif family == 'S08':
        weights = dict(E=0., R=0. if variant == 0 else .05, A=.75 if variant == 0 else .70, D=.15, B=.10)
    elif family == 'S09':
        density = (.02, .15, .45)[variant]
        weights = dict(E=.20, R=density, A=.50-density, D=.15, B=.15)
    elif family == 'S10':
        weights = dict(E=.10, R=.15, A=.30, D=.20, B=.15, L=.10)
    elif family == 'S11':
        weights = dict(E=0., R=0. if variant == 0 else .15, A=.60 if variant == 0 else .45, D=.25, B=.15)
    counts = {name: int(n_mutations*weight) for name, weight in weights.items()}
    for name, weight in weights.items():
        if weight and counts[name] == 0:
            counts[name] = 1
    while sum(counts.values()) > n_mutations:
        largest = max(counts, key=counts.get)
        counts[largest] -= 1
    for i in range(n_mutations-sum(counts.values())):
        counts[list(weights)[i % len(weights)]] += 1
    # Zero-weight classes remain absent, including the no-assayed-trunk panels.
    zero_classes = [key for key, weight in weights.items() if not weight]
    for key in zero_classes:
        counts['A'] += counts[key]
        counts[key] = 0
    classes = [name for name, count in counts.items() for _ in range(count)]
    assignment = rng.permutation(np.arange(n_mutations) % n_segments)
    mutation_segment = {m: segments[0 if label == 'L' else assignment[i]] for i, (m, label) in enumerate(zip(mutations, classes))}
    by_class = {name: [m for m, label in zip(mutations, classes) if label == name] for name in weights}
    engine = _Copies(segments)
    engine.root()
    parallel = family in ('S04', 'S05', 'S12') or (family == 'S06' and int(seed) % 2 == 1)
    shared_wgd = family == 'S06' and not parallel
    major_targets = {segment: int(rng.integers(*major_cn_range)) for segment in segments}
    if family == 'S02':
        for segment in segments:
            for _ in range(major_targets[segment]-2):
                engine.gain('trunk', segment)
    for label in ('E', 'R', 'L'):
        if label == 'R' and shared_wgd:
            continue
        for mutation in by_class.get(label, ()):
            engine.snv(mutation, 'trunk', mutation_segment[mutation], 1 if label == 'E' or family == 'S02' else 2)
    if family == 'S10':
        engine.fork('trunk', 'unsampled')
        engine.loh('trunk', segments[0])
        if n_segments > 1:
            for _ in range(10):
                engine.gain('trunk', segments[1])
    elif family == 'S02':
        for segment in segments:
            engine.gain('trunk', segment)
    elif family in ('S03', 'S09', 'S11'):
        for segment in segments:
            for _ in range(2 if family == 'S09' else 1):
                engine.gain('trunk', segment)
    elif shared_wgd:
        engine.wgd('trunk')
        for mutation in by_class['R']:
            engine.snv(mutation, 'trunk', mutation_segment[mutation], 2)
    engine.fork('trunk', 'persistent')
    engine.fork('trunk', 'ancestor')
    for mutation in by_class['A']:
        engine.snv(mutation, 'ancestor', mutation_segment[mutation])
    if parallel:
        for clone in ('persistent', 'ancestor'):
            if family == 'S06':
                engine.wgd(clone)
            else:
                for segment in segments:
                    engine.gain(clone, segment)
    engine.fork('ancestor', 'descendant')
    for mutation in by_class['D']:
        engine.snv(mutation, 'descendant', mutation_segment[mutation])
    engine.fork('trunk', 'sibling')
    if parallel:
        if family == 'S06':
            engine.wgd('sibling')
        else:
            for segment in segments:
                engine.gain('sibling', segment)
    for mutation in by_class['B']:
        engine.snv(mutation, 'sibling', mutation_segment[mutation])
    if family != 'S10':
        engine.fork('trunk', 'unsampled')
    clones = ('persistent', 'ancestor', 'descendant', 'sibling', 'unsampled')
    # Region-specific streams make adding regions preserve every existing mixture.
    regional_draws = np.array([np.random.default_rng(np.random.SeedSequence([int(seed), 701, r])).random(3)
                              for r in range(n_regions)])
    ancestor = .45 + .35*regional_draws[:, 0]
    if family in ('S04', 'S05', 'S12'):
        ancestor = (np.resize([.35, .50, .65, .80, .40, .55, .70, .90], n_regions)
                    if variant == 1 else np.full(n_regions, .5))
        if variant == 2 and n_regions >= 4:
            ancestor[3] = .65
    elif family == 'S07':
        ancestor = np.resize([1., .65, .25, .45], n_regions)
    elif family == 'S08':
        ancestor = .97 + .02*regional_draws[:, 0]
    descendant = ancestor*(.25 + .30*regional_draws[:, 1])
    sibling = (1-ancestor)*(.25 + .40*regional_draws[:, 2])
    fractions = np.array([1-ancestor-sibling, ancestor-descendant, descendant, sibling, np.zeros(n_regions)])
    # Last nonzero component absorbs floating summation error without changing history.
    fractions[0] = 1-fractions[1:].sum(axis=0)
    fractions[np.abs(fractions) < 1e-15] = 0
    vcounts = np.zeros((n_mutations, len(clones)), int)
    cn = np.zeros((n_segments, len(clones), 2), int)
    mi, si = {m: i for i, m in enumerate(mutations)}, {s: i for i, s in enumerate(segments)}
    for j, clone in enumerate(clones):
        for segment, copies in engine.states[clone].items():
            for copy_id in copies:
                copy = engine.copies[copy_id]
                cn[si[segment], j, copy['homolog']-1] += 1
                for mutation in copy['mutation_ids']:
                    vcounts[mi[mutation], j] += 1
    phi = (vcounts > 0) @ fractions
    multiplicity, major, minor = np.full_like(phi, np.nan), np.empty_like(phi, dtype=int), np.empty_like(phi, dtype=int)
    for i, mutation in enumerate(mutations):
        segment = si[mutation_segment[mutation]]
        for r in range(n_regions):
            present = fractions[:, r] > 0
            carriers = present & (vcounts[i] > 0)
            a, b = sorted(cn[segment, np.flatnonzero(present)[0]], reverse=True)
            major[i, r], minor[i, r] = a, b
            if carriers.any():
                multiplicity[i, r] = vcounts[i, carriers][0]
                if np.all(vcounts[i, present] > 0):
                    phi[i, r] = 1.
    vectors = sorted(set(tuple(np.round(row, 12)) for row in phi))
    truth_ids = tuple(f'g{vectors.index(tuple(np.round(row, 12))):03}' for row in phi)
    branches = {event['mutation_id']: event['clone_id'] for event in engine.events if event['event_type'] == 'snv'}
    sources = by_class['E']+by_class['R'] if family in ('S01', 'S02') else by_class.get('R', [])
    if family == 'S09':
        limit = (1, 5, 20)[variant]
        sources = [m for segment in segments for m in [x for x in sources if mutation_segment[x] == segment][:limit]]
    metadata = dict(schema='clipp2.simulation.biological-design.v1', study_version='1.1', family=family,
        family_name=FAMILIES[family], history_id=f'{family}_{int(seed):010}', seed=int(seed), variant=variant,
        mutation_classes={name: tuple(ids) for name, ids in by_class.items()},
        declared_source_ids=tuple(sources), declared_remainder_ids=tuple(m for m in mutations if m not in set(sources)),
        controlled_source_status='mechanism_intervention_not_an_expected_fitted_partition',
        normal_cn=2, mutation_key_convention='simulation_somatic_snv_id', genome_build='synthetic_atomic_intervals',
        interval_map={segment: dict(chromosome='1', start=1+i*100000, end=(i+1)*100000) for i, segment in enumerate(segments)},
        acquisition_panel=('missing_zero_excluded' if family == 'S10' else 'mask_last_distinguishing_region'
                           if family == 'S12' and variant == 2 else 'complete'),
        distinguishing_region_index=3 if family in ('S04', 'S05', 'S12') and variant == 2 and n_regions >= 4 else None,
        reference_panels=(['complete', 'remove_ancestor', 'duplicate_ancestor', 'distort_ancestor', 'K1']
                          if family in ('S04', 'S05', 'S11', 'S12') else ['complete']),
        limitations=['Constructed family frequencies are not patient prevalence.',
            'Core atomic intervals are shared; segment errors are acquisition perturbations.',
            'Controlled reference/source panels do not force ordinary fitter outcomes.'])
    copy_rows = tuple(dict(record, mutation_ids=tuple(sorted(record['mutation_ids']))) for record in engine.copies.values())
    clone_rows = tuple(dict(clone_id=clone, sampled=bool(fractions[j].any()), fractions=fractions[j].tolist()) for j, clone in enumerate(clones))
    return History(mutations, regions, segments,
        np.asarray([[mutation_segment[m]]*n_regions for m in mutations]), phi, multiplicity, major, minor,
        clones, fractions, vcounts, cn, tuple(branches[m] for m in mutations), truth_ids,
        tuple(engine.tree), clone_rows, tuple(engine.events), copy_rows, metadata)
