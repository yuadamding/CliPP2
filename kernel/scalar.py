"""Single-region specialization of CliPP2's joint CCF kernel.

The native optimizer returns arrays directly; selection consumes the same
in-memory proposals. No intermediate TSVs or upstream result API are involved.
"""
import json
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from . import ALGORITHM, KERNEL_VERSION
from .candidates import CandidateStore
from .initialization import pooled_cp_initialization
from .likelihood import MultiplicityModel
from .native import solve_chain
from .selection import select_chain
from .topology import build_tree


def supports_scalar(model):
    """Single observed region with the exact scalar emission domain."""
    rows = np.asarray(model.active_indices)
    return (model.r == 1 and bool(model.observed.all())
            and bool(np.all(model.data.cn_state_count[rows] == 1))
            and bool(np.all(model.data.normal_cn[rows] == 2)))


def fit_scalar(model, max_clusters=10, *, coordinate_keys=None):
    if not supports_scalar(model):
        raise ValueError('Scalar kernel requires observed single-state diploid-normal input')
    if type(max_clusters) is not int or not 1 <= max_clusters <= 10:
        raise ValueError('max_clusters must be in 1..10')
    arrays = (model.alt[:, 0], model.depth[:, 0], model.major[:, 0], model.total[:, 0])
    purity = float(model.purity[0, 0])
    if not np.all(model.purity[:, 0] == purity):
        raise ValueError('Scalar kernel requires one regional purity')
    scalar = MultiplicityModel(*arrays, purity)
    started = perf_counter()
    pilot_cp, initialization = pooled_cp_initialization(scalar)
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
        pilot_cp[order], capacities, evaluator=evaluate if cuda else None)
    timings['continuation'] = perf_counter()-started
    started = perf_counter()
    with CandidateStore() as bank:
        selected = select_chain(scalar, order, proposals, capacities, candidate_store=bank)
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
        weights = np.asarray(parameters['weights'])
        fit = SimpleNamespace(labels=labels, centers=centers, weights=weights,
            conditional_log_likelihood=float(winner.conditional_log_likelihood),
            mixture_log_likelihood=float(winner.log_likelihood),
            complexity_penalty=(2*len(centers)-1)*np.log(model.n), score=float(winner.bic),
            eligible=True, reason=None,
            diagnostics={'weight_optimality_gap': float(winner.weight_optimality_gap)})
        telemetry = dict(selected['telemetry'])
        continuation_records = selected['selection'].to_dict(orient='records')
    timings['refit_weights_refinement'] = perf_counter()-started
    cuts = tuple(int(e) for e, (i, j) in enumerate(tree.edges) if labels[i] != labels[j])
    candidate = SimpleNamespace(fit=fit, cuts=cuts, requested_k=int(winner.requested_k),
                               kind=str(winner.candidate_kind), diagnostics={})
    return SimpleNamespace(selected=candidate, labels=labels, centers=centers, weights=weights,
        score=fit.score, candidate_bank=records, tree=tree, pilot=pilot, diagnostics={
            'algorithm': ALGORITHM,
            'kernel_version': KERNEL_VERSION, 'native_source_sha256': native_digest,
            'tie_policy': tie_policy, 'tree_identity': tree.identity,
            'continuation_records': continuation_records,
            'backend': 'cuda_likelihood_host_chain' if cuda else 'cpu_native_chain',
            'initialization': initialization, 'phase_seconds': timings, 'cache': telemetry,
            'global_optimality_proven': False, 'joint_mixture_center_mle': False,
            'qualification': 'numerical_multimode_refit_not_global_certificate'})
