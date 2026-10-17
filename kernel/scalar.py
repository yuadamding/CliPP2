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
from .topology import FrozenTree, build_tree, partition_from_cuts


def supports_scalar(model):
    """Single observed region without unsupported native hard-zero bounds."""
    support = getattr(model, 'regional_support', None)
    if support is not None and support.hard_zero.any():
        return False
    rows = np.asarray(model.active_indices)
    return (model.r == 1 and bool(model.observed.all())
            and bool(np.all(model.data.normal_cn[rows] == 2))
            and bool(np.all(model.lower == 0)) and bool(np.all(model.upper == 1)))


def fit_scalar(model, max_clusters=DEFAULT_MAX_CLUSTERS, *, coordinate_keys=None, search=None,
               checkpoint_path=None, resume=False, checkpoint_stop_after=None):
    if not supports_scalar(model):
        raise ValueError('Scalar kernel requires observed diploid-normal input without strict '
                         'regional zeros; use the shared support-aware fitting path')
    if type(max_clusters) is not int or not 1 <= max_clusters <= MAX_CLUSTERS:
        raise ValueError(f'max_clusters must be in 1..{MAX_CLUSTERS}')
    search = (SearchPolicy() if search is None else search).resolve(max_clusters)
    if (checkpoint_path is not None or resume or checkpoint_stop_after is not None) and search.mode != 'bounded':
        raise ValueError('Search checkpoints require explicit bounded search; no route is changed silently')
    if resume and checkpoint_path is None:
        raise ValueError('resume requires a checkpoint_path')
    model.check_clonal_feasibility(max_clusters)
    arrays = (model.alt[:, 0], model.depth[:, 0], model.major[:, 0], model.total[:, 0])
    purity = float(model.purity[0, 0])
    if not np.all(model.purity[:, 0] == purity):
        raise ValueError('Scalar kernel requires one regional purity')
    scalar = MultiplicityModel(*arrays, purity)
    # A scientific objective must not change merely because dispatch selected
    # the specialized one-region chain route.
    scalar.selection_objective = model.selection_objective
    if coordinate_keys is None:
        chrom, position = np.array(model.mutation_ids), np.full(model.n, '')
        tie_policy = 'canonical_mutation_id_v1'
    else:
        chrom, position = [np.asarray(a, dtype=str) for a in coordinate_keys]
        if chrom.shape != (model.n,) or position.shape != (model.n,):
            raise ValueError('Coordinate tie keys must follow the retained mutation order')
        tie_policy = 'chromosome_position_text_v1'
    saved = None
    if resume:
        from ..tree.checkpoint import read_checkpoint
        saved = read_checkpoint(checkpoint_path, model)
        context = saved.get('checkpoint_context')
        if not context or 'scalar_pilot_cp' not in context:
            raise ValueError('Scalar resume requires its exact native CP pilot and chain-order context')
        pilot_cp, order = context['scalar_pilot_cp'], context['scalar_chain_order']
        if (context['tie_policy'] != tie_policy
                or not np.array_equal(order, np.lexsort((position, chrom, pilot_cp)))):
            raise ValueError('Scalar checkpoint coordinate tie order changed')
        pilot = saved['pilot']
        tree = FrozenTree(**saved['tree'])
        if tree.identity != build_tree(pilot, model.observed, model.mutation_ids, chain_order=order).identity:
            raise ValueError('Scalar checkpoint chain does not match its pilot/order')
        initialization = saved['pilot_diagnostics'][model.region_ids[0]]
        timings = {'initialization': saved['timings']['initialization_seconds'],
                   'continuation': saved['timings']['continuation_seconds']}
        native_digest = context['native_source_sha256']
    else:
        started = perf_counter()
        pilot_cp, initialization = pooled_cp_initialization(scalar, **(
            {'matrix_cache_bytes': model.memory.initialization_bytes} if model.memory else {}))
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
    old_seeds = list(saved['seeds']) if saved is not None else []
    old_capacities = {seed['requested_k'] for seed in old_seeds}
    pending = np.asarray([k for k in capacities if k not in old_capacities], dtype=np.int32)
    if len(pending):
        proposals, native_digest = solve_chain(tuple(a[order] for a in arrays), purity,
            pilot_cp[order], pending, clonal_eligible=model.clonal_eligible[order],
            evaluator=evaluate if cuda else None,
            numerical_diagnostics_dir=getattr(model, 'numerical_diagnostics_dir', None),
            numerical_diagnostics_max_bytes=getattr(model, 'numerical_diagnostics_max_bytes', 16_777_216))
    else:
        proposals = []
    for proposal in proposals:
        raw = proposal['diagnostics']
        index = int(order[raw['anchor_witness_index']])
        raw['anchor_witness_canonical_index'] = index
        raw['anchor_witness_mutation_id'] = model.mutation_ids[index]
    timings['continuation'] = timings.get('continuation', 0.)+perf_counter()-started
    if search.mode == 'bounded':
        # The high-capacity bank is deliberately nonexhaustive. Keep scalar
        # initialization/native CP continuation, but reuse the joint conditional
        # refit/scoring bank. Equal-center adjacent merges are budgeted search
        # proposals, not an implicit q-changing step inside a scored variant.
        from ..tree.search import fit_bounded
        seeds = old_seeds
        for proposal in proposals:
            labels = np.empty(model.n, dtype=int)
            labels[order] = proposal['labels']
            cuts = tuple(int(e) for e, (i, j) in enumerate(tree.edges) if labels[i] != labels[j])
            seeds.append(dict(requested_k=proposal['requested_k'], cuts=cuts,
                              labels=partition_from_cuts(tree, cuts), diagnostics=proposal['diagnostics']))
        result = fit_bounded(model, tree, pilot, seeds, tuple(map(int, capacities)), search,
                             pilot_diagnostics={model.region_ids[0]: initialization},
                             initialization_seconds=timings['initialization'],
                             continuation_seconds=timings['continuation'],
                             checkpoint_path=checkpoint_path, resume=resume,
                             checkpoint_stop_after=checkpoint_stop_after,
                             checkpoint_context={'scalar_pilot_cp': pilot_cp,
                                 'scalar_chain_order': order, 'tie_policy': tie_policy,
                                 'native_source_sha256': native_digest})
        result.diagnostics.update(native_source_sha256=native_digest, tie_policy=tie_policy,
            backend='cuda_likelihood_host_chain' if cuda else 'cpu_native_chain',
            scalar_refit_policy='scalar_CP_refits_joint_bank_budgeted_equal_center_merges',
            continuation_records=[dict(requested_k=p['requested_k'], **p['diagnostics']) for p in seeds])
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
            mixture_log_likelihood=float(winner.mixture_log_likelihood),
            selection_log_likelihood=float(winner.selection_log_likelihood),
            classification_count_term=float(winner.classification_count_term),
            selection_objective=model.selection_objective,
            complexity_penalty=2*(len(centers)-1)*np.log(model.n), score=float(winner.bic),
            eligible=True, reason=None,
            diagnostics={key: winner[key] for key in (
                'weight_optimality_gap', 'weight_active_score_gap', 'selection_objective',
                'score_definition', 'selection_log_likelihood', 'classification_count_term',
                'anchoring_loss', 'anchoring_loss_available', 'anchoring_loss_unavailable_reason',
                'proportions_definition', 'mixture_weight_optimization')})
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
