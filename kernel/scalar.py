"""Single-region specialization of CliPP2's joint CCF kernel.

The native optimizer returns arrays directly; selection consumes the same
in-memory proposals. No intermediate TSVs or upstream result API are involved.
"""
import json
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from . import ALGORITHM, KERNEL_VERSION, MAX_CLUSTERS
from ..config import DEFAULT_MAX_CLUSTERS, SearchPolicy
from .candidates import CandidateStore
from .initialization import pooled_cp_initialization
from .likelihood import MultiplicityModel
from .native import solve_chain
from .selection import select_chain
from .topology import build_tree, partition_from_cuts


def supports_scalar(model):
    """Single observed region with the exact scalar emission domain."""
    rows = np.asarray(model.active_indices)
    return (model.r == 1 and bool(model.observed.all())
            and bool(np.all(model.data.normal_cn[rows] == 2))
            and bool(np.all(model.lower == 0)) and bool(np.all(model.upper == 1)))


def fit_scalar(model, max_clusters=DEFAULT_MAX_CLUSTERS, *, coordinate_keys=None, search=None):
    if not supports_scalar(model):
        raise ValueError('Scalar kernel requires observed single-state diploid-normal input')
    if type(max_clusters) is not int or not 1 <= max_clusters <= MAX_CLUSTERS:
        raise ValueError(f'max_clusters must be in 1..{MAX_CLUSTERS}')
    search = (SearchPolicy() if search is None else search).resolve(max_clusters)
    model.check_clonal_feasibility(max_clusters)
    arrays = (model.alt[:, 0], model.depth[:, 0], model.major[:, 0], model.total[:, 0])
    purity = float(model.purity[0, 0])
    if not np.all(model.purity[:, 0] == purity):
        raise ValueError('Scalar kernel requires one regional purity')
    scalar = MultiplicityModel(*arrays, purity)
    started = perf_counter()
    pilot_cp, initialization = pooled_cp_initialization(scalar, **(
        {'matrix_cache_bytes': model.memory.initialization_bytes} if model.memory else {}))
    if coordinate_keys is None:
        chrom, position = np.array(model.mutation_ids), np.full(model.n, '')
        tie_policy = 'canonical_mutation_id_v1'
    else:
        chrom, position = [np.asarray(a, dtype=str) for a in coordinate_keys]
        if chrom.shape != (model.n,) or position.shape != (model.n,):
            raise ValueError('Coordinate tie keys must follow the retained mutation order')
        tie_policy = 'chromosome_position_text_v1'
    order = np.lexsort((position, chrom, pilot_cp))
    pilot = (pilot_cp/purity)[:, None]
    tree = build_tree(pilot, model.observed, model.mutation_ids, chain_order=order)
    timings = {'initialization': perf_counter()-started}

    def evaluate(x):
        phi = np.empty((model.n, 1))
        phi[order, 0] = x
        return tuple(term[order, 0] for term in model.node_terms(phi))

    cuda = str(model.device).startswith('cuda')
    capacities = np.arange(1, min(model.n, max_clusters)+1, dtype=np.int32)
    started = perf_counter()
    proposals, native_digest = solve_chain(tuple(a[order] for a in arrays), purity,
        pilot_cp[order], capacities, clonal_eligible=model.clonal_eligible[order],
        evaluator=evaluate if cuda else None)
    for proposal in proposals:
        raw = proposal['diagnostics']
        index = int(order[raw['anchor_witness_index']])
        raw['anchor_witness_canonical_index'] = index
        raw['anchor_witness_mutation_id'] = model.mutation_ids[index]
    timings['continuation'] = perf_counter()-started
    if search.mode == 'bounded':
        # The high-capacity bank is deliberately nonexhaustive. Keep scalar
        # initialization/native CP continuation, but reuse the joint conditional
        # refit/scoring bank. Equal-center adjacent merges are budgeted search
        # proposals, not an implicit q-changing step inside a scored variant.
        from ..tree.search import fit_bounded
        seeds = []
        for proposal in proposals:
            labels = np.empty(model.n, dtype=int)
            labels[order] = proposal['labels']
            cuts = tuple(int(e) for e, (i, j) in enumerate(tree.edges) if labels[i] != labels[j])
            seeds.append(dict(requested_k=proposal['requested_k'], cuts=cuts,
                              labels=partition_from_cuts(tree, cuts), diagnostics=proposal['diagnostics']))
        result = fit_bounded(model, tree, pilot, seeds, tuple(map(int, capacities)), search,
                             pilot_diagnostics={model.region_ids[0]: initialization},
                             initialization_seconds=timings['initialization'],
                             continuation_seconds=timings['continuation'])
        result.diagnostics.update(native_source_sha256=native_digest, tie_policy=tie_policy,
            backend='cuda_likelihood_host_chain' if cuda else 'cpu_native_chain',
            scalar_refit_policy='scalar_CP_refits_joint_bank_budgeted_equal_center_merges',
            continuation_records=[dict(requested_k=p['requested_k'], **p['diagnostics']) for p in proposals])
        return result
    started = perf_counter()
    with CandidateStore() as bank:
        selected = select_chain(scalar, order, proposals, capacities, candidate_store=bank,
                                clonal_eligible=model.clonal_eligible, memory=model.memory,
                                refit_store=model.refit_store())
        records = tuple(bank.iter_rows())
        winner = selected['selection'].loc[selected['selection'].selected].iloc[0]
        record = next(f for f in selected['fits'] if f['requested_k'] == winner.requested_k
                      and f['candidate_id'] == winner.candidate_id)
        parameters = record['partition_parameters']
        if isinstance(parameters, str):
            parameters = json.loads(parameters)
        labels = np.empty(model.n, dtype=np.int64)
        labels[order] = np.asarray(parameters['block_labels'], dtype=int)[
            np.searchsorted(parameters['cuts'], np.arange(model.n), side='right')]
        centers = np.asarray(parameters['centers'])[:, None]/purity
        clonal = int(parameters['clonal_cluster'])
        centers[clonal] = 1.
        weights = np.asarray(parameters['weights'])
        fit = SimpleNamespace(labels=labels, centers=centers, weights=weights,
            clonal_cluster=clonal,
            conditional_log_likelihood=float(winner.conditional_log_likelihood),
            mixture_log_likelihood=float(winner.log_likelihood),
            complexity_penalty=2*(len(centers)-1)*np.log(model.n), score=float(winner.bic),
            eligible=True, reason=None,
            diagnostics={'weight_optimality_gap': float(winner.weight_optimality_gap)})
        telemetry = dict(selected['telemetry'])
        continuation_records = selected['selection'].to_dict(orient='records')
    timings['refit_weights_refinement'] = perf_counter()-started
    cuts = tuple(int(e) for e, (i, j) in enumerate(tree.edges) if labels[i] != labels[j])
    candidate = SimpleNamespace(fit=fit, cuts=cuts, requested_k=int(winner.requested_k),
                               kind=str(winner.candidate_kind), diagnostics={})
    return SimpleNamespace(selected=candidate, labels=labels, centers=centers, weights=weights,
        clonal_cluster=clonal,
        score=fit.score, candidate_bank=records, tree=tree, pilot=pilot, diagnostics={
            'algorithm': ALGORITHM,
            'kernel_version': KERNEL_VERSION, 'native_source_sha256': native_digest,
            'tie_policy': tie_policy, 'tree_identity': tree.identity,
            'continuation_records': continuation_records,
            'backend': 'cuda_likelihood_host_chain' if cuda else 'cpu_native_chain',
            'initialization': initialization, 'phase_seconds': timings, 'cache': telemetry,
            'global_optimality_proven': False, 'joint_mixture_center_mle': False,
            'qualification': 'numerical_multimode_refit_not_global_certificate'})
