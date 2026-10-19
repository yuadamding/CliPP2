"""Read-only frozen numerical replay; no fitting, scheduler, or precision fix.

The independent Decimal elimination solves the saved active-set free systems,
including springs to bound neighbors. It is not an independent active-set QP
optimizer: its full KKT audit says whether the saved active set is consistent.
H and b already contain the anchor augmentation and are never augmented here.
"""
import argparse
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .numerical import QP_TOLERANCE, _json_safe, qp_residual


def read_snapshot(path, manifest_path=None):
    path = Path(path)
    manifest = json.loads(Path(manifest_path or path.parent/'manifest.json').read_text())
    entry = manifest['snapshots'][path.name]
    payload = path.read_bytes()
    if len(payload) != entry['bytes'] or hashlib.sha256(payload).hexdigest() != entry['sha256']:
        raise ValueError('Snapshot integrity mismatch')
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    metadata = json.loads(str(arrays['metadata']))
    if metadata.get('schema') not in (1, 2):
        raise ValueError('Unsupported numerical snapshot schema')
    required = {'H', 'b', 'L', 'U', 'rho', 'edges', 'result', 'active_state'}
    if required - arrays.keys():
        raise ValueError('Incomplete quadratic snapshot')
    return arrays


def _decimal(value):
    return Decimal.from_float(float(value))


def decimal_free_system(curvature, rhs, edges, penalty, lower, upper, active, *, digits=80):
    """Direct Schur elimination retaining Decimal parents through reconstruction.

    Native uses positive messages and rounded-parent reconstruction instead.
    Preserve exact binary64 inputs and every bound-neighbor diagonal/RHS term.
    """
    h, b, lo, hi = [np.asarray(v, dtype=float) for v in (curvature, rhs, lower, upper)]
    active, edges = np.asarray(active), np.asarray(edges)
    if (h.ndim != 2 or not all(h.shape) or any(a.shape != h.shape for a in (b, lo, hi, active))
            or not np.isin(active, [-1, 0, 1]).all() or digits < 40):
        raise ValueError('Invalid free-system arrays, active state, or precision')
    if (not all(np.isfinite(a).all() for a in (h, b, lo, hi))
            or np.any(h <= 0) or np.any(lo > hi) or not np.isfinite(penalty) or penalty < 0):
        raise ValueError('Invalid free-system quadratic or bounds')
    if edges.ndim != 2 or edges.shape[1] != 2 or edges.dtype.kind not in 'iu':
        raise ValueError('Invalid forest edges')
    adjacency = [[] for _ in range(len(h))]
    roots = np.arange(len(h))
    for i, j in edges:
        if min(i, j) < 0 or max(i, j) >= len(h):
            raise ValueError('Forest endpoint out of range')
        a, c = int(i), int(j)
        while roots[a] != a:
            a = roots[a]
        while roots[c] != c:
            c = roots[c]
        if a == c:
            raise ValueError('Free-system graph must be a forest')
        roots[c] = a
        adjacency[i].append(int(j))
        adjacency[j].append(int(i))
    with localcontext() as context:
        context.prec = digits
        rho = _decimal(penalty)
        answer = np.empty(h.shape, dtype=object)
        for region in range(h.shape[1]):
            state = active[:, region].copy()
            state[lo[:, region] == hi[:, region]] = -1
            x = [_decimal(lo[i, region] if state[i] < 0 else hi[i, region])
                 for i in range(len(h))]
            diagonal = [_decimal(h[i, region]) + rho*len(adjacency[i]) for i in range(len(h))]
            linear = [_decimal(b[i, region]) for i in range(len(h))]
            parent = np.full(len(h), -2, dtype=int)
            order = []
            for root in range(len(h)):
                if state[root] or parent[root] != -2:
                    continue
                parent[root] = -1
                stack = [root]
                while stack:
                    i = stack.pop()
                    order.append(i)
                    for j in adjacency[i]:
                        if state[j]:
                            linear[i] += rho*x[j]
                        elif parent[j] == -2:
                            parent[j] = i
                            stack.append(j)
            for i in reversed(order):
                if diagonal[i] <= 0:
                    raise ArithmeticError('Reference Schur complement lost positive definiteness')
                j = parent[i]
                if j >= 0:
                    diagonal[j] -= rho*rho/diagonal[i]
                    linear[j] += rho*linear[i]/diagonal[i]
            for i in order:
                numerator = linear[i] + (rho*x[parent[i]] if parent[i] >= 0 else 0)
                x[i] = numerator/diagonal[i]
            answer[:, region] = x
        audit = decimal_residual(h, b, edges, penalty, lo, hi, answer, digits=digits)
    return answer, audit


def decimal_residual(curvature, rhs, edges, penalty, lower, upper, result, *, digits=80):
    """Original-input componentwise KKT check before the FP64 return boundary."""
    h, b, lo, hi = [np.asarray(v) for v in (curvature, rhs, lower, upper)]
    with localcontext() as context:
        context.prec = digits
        rho = _decimal(penalty)
        errors, worst, maximum, violation = [], None, Decimal(-1), Decimal(0)
        for region in range(h.shape[1]):
            x = list(result[:, region])
            base = [_decimal(h[i, region])*x[i] for i in range(len(h))]
            g = [base[i]-_decimal(b[i, region]) for i in range(len(h))]
            scale = [1+abs(base[i])+abs(_decimal(b[i, region])) for i in range(len(h))]
            for i, j in edges:
                jump = rho*(x[i]-x[j])
                g[i] += jump
                g[j] -= jump
                scale[i] += abs(jump)
                scale[j] += abs(jump)
            regional = Decimal(0)
            for i in range(len(h)):
                low, high = _decimal(lo[i, region]), _decimal(hi[i, region])
                violation = max(violation, low-x[i], x[i]-high)
                residual = g[i]
                if low == high:
                    residual = Decimal(0)
                elif x[i] == low:
                    residual = min(residual, Decimal(0))
                elif x[i] == high:
                    residual = max(residual, Decimal(0))
                relative = abs(residual)/scale[i]
                regional = max(regional, relative)
                if relative > maximum:
                    maximum, worst = relative, [i, region]
            errors.append(float(regional))
        return dict(backward_error=float(maximum), backward_error_decimal=str(maximum),
                    regional_backward_error=errors, bound_violation=float(violation),
                    worst_coordinate=worst, decimal_digits=digits,
                    solved=bool(maximum <= _decimal(QP_TOLERANCE) and violation == 0))


def replay_qp(arrays, *, digits=80):
    from .native import native_library
    from .solver import boxed_forest_quadratic
    h, b, lo, hi, edges = [arrays[name] for name in ('H', 'b', 'L', 'U', 'edges')]
    rho = float(arrays['rho'])
    frozen = qp_residual(h, b, edges, rho, lo, hi, arrays['result'])
    result, audit = boxed_forest_quadratic(h, b, edges, rho, lo, hi)
    active = audit.pop('active_state')
    exact, high_precision = decimal_free_system(h, b, edges, rho, lo, hi,
                                               arrays['active_state'], digits=digits)
    rounded = np.asarray(exact, dtype=float)
    rounded_audit = qp_residual(h, b, edges, rho, lo, hi, rounded)
    library, source_sha = native_library()
    binary_sha = hashlib.sha256(Path(library._name).read_bytes()).hexdigest()
    metadata = json.loads(str(arrays['metadata'])) if 'metadata' in arrays else {}
    frozen_identity = metadata.get('source', {})
    if high_precision['solved'] and rounded_audit['failure_reasons']:
        interpretation = ('Independent high-precision free-system coordinates qualify, but their FP64 '
                          'rounding does not. Returning high-precision elimination as FP64 alone '
                          'does not repair this frozen subproblem.')
    elif not high_precision['solved']:
        interpretation = ('Saved active set does not qualify under the independent free-system solve; '
                          'inspect bound activity and reference residuals before attributing rounding.')
    else:
        interpretation = ('High-precision reference and its FP64 return qualify; differences from native '
                          'require elimination, reconstruction, and active-set investigation.')
    return dict(frozen=frozen, native=audit, independent_reference=rounded_audit,
        high_precision=high_precision, native_internal_coordinates='not exposed by native ABI',
        native_vs_frozen_max_abs=float(np.max(abs(result-arrays['result']))),
        native_vs_reference_max_abs=float(np.max(abs(result-rounded))),
        frozen_vs_reference_max_abs=float(np.max(abs(arrays['result']-rounded))),
        native_vs_frozen_active_state_changes=int(np.count_nonzero(active != arrays['active_state'])),
        native_source_matches_snapshot=source_sha == frozen_identity.get('native_source_sha256'),
        native_binary_matches_snapshot=(binary_sha == frozen_identity['native_binary_sha256']
                                       if 'native_binary_sha256' in frozen_identity else None),
        replay_native_source_sha256=source_sha, replay_native_binary_sha256=binary_sha,
        native_abi=library.CliPP2KernelABI(), anchor_augmented_twice=False,
        qualification_tolerance=QP_TOLERANCE, interpretation=interpretation)


def replay_likelihood(arrays, cuda=False):
    from .model import RegionalModel
    data = SimpleNamespace(tumor_id='frozen_numerical_replay',
        mutation_ids=tuple(arrays['model_mutation_ids'].tolist()),
        region_ids=tuple(arrays['model_region_ids'].tolist()),
        alt_counts=arrays['model_alt'], total_counts=arrays['model_depth'],
        major_cn=arrays['model_major'], normal_cn=arrays['model_normal'],
        total_cn=arrays['model_total'], purity=arrays['model_purity'],
        count_observed=arrays['model_observed'])
    cpu = RegionalModel(data, device='cpu', regional_zero_policy='likelihood')
    x = arrays['X']
    if cpu.mutation_ids != data.mutation_ids or cpu.region_ids != data.region_ids:
        raise ValueError('Frozen likelihood row/region order is not canonical; refusing silent reorder')
    ordinary = cpu.node_terms(x)
    batched = cpu.node_terms_batch(np.stack((x, x)))
    result = dict(cpu_vs_cpu_batch_max_abs=[float(np.max(abs(v-b[0]))) for v, b in zip(ordinary, batched)],
                  cuda='not_requested', shape=list(x.shape),
                  curvature_meaning='surrogate posterior-mean complete-data curvature, not true Hessian')
    for key, val in zip(('likelihood_gradient', 'likelihood_curvature'), ordinary[1:]):
        if key in arrays:
            result['frozen_vs_cpu_' + key] = float(np.max(abs(arrays[key]-val)))
    if {'H', 'edges', 'rho', 'metadata', 'likelihood_gradient'} <= arrays.keys():
        result['stagnation_guard_probe'] = stagnation_guard_probe(arrays, cpu, ordinary[0])
    if cuda:
        gpu = RegionalModel(data, device='cuda', regional_zero_policy='likelihood')
        direct = gpu.node_terms(x)
        batch = gpu.node_terms_batch(np.stack((x, x)))
        result.update(cuda='executed_on_explicitly_available_device',
            cpu_vs_cuda_max_abs=[float(np.max(abs(a-b))) for a, b in zip(ordinary, direct)],
            cpu_vs_cuda_batch_max_abs=[float(np.max(abs(a-b[0]))) for a, b in zip(ordinary, batch)])
    return result


def stagnation_guard_probe(arrays, model, values):
    """One frozen fallback step, not a replay of the historical trajectory."""
    from .anchor import anchor_penalty
    from .solver import _spring_gradient
    metadata = json.loads(str(arrays['metadata']))
    if 'witness' not in metadata:
        return {'available': False, 'reason': 'snapshot_has_no_anchor_witness'}
    x, forest, rho = arrays['X'], arrays['edges'], float(arrays['rho'])
    witness = metadata['witness']
    gradient = arrays['likelihood_gradient']+_spring_gradient(x, forest, rho)
    gradient[witness] += rho*(x[witness]-1.)
    degree = np.bincount(forest.ravel(), minlength=len(x))[:, None]
    proposal = np.clip(x-gradient/(arrays['H']+rho*degree), arrays['L'], arrays['U'])
    direction = proposal-x
    slope = float(np.sum(gradient*direction))
    def objective(point, loss):
        jumps = point[forest[:, 0]]-point[forest[:, 1]]
        return float(np.sum(loss)+.5*rho*np.sum(jumps*jumps))+anchor_penalty(point, witness, rho)
    old = objective(x, arrays.get('likelihood_values', values))
    threshold = 1e-12*max(1., abs(old))
    accepted, decrease, backtracks, step_max = False, 0., 0, 0.
    if np.isfinite(slope) and slope < 0:
        for backtracks in range(40):
            trial = np.clip(x+.5**backtracks*direction, arrays['L'], arrays['U'])
            value = objective(trial, model.node_terms(trial)[0])
            if np.isfinite(value) and value <= old+1e-4*.5**backtracks*slope+threshold:
                accepted, decrease = True, old-value
                step_max = float(np.max(abs(trial-x)))
                break
    return dict(available=True, scope='one frozen CPU-evaluated fallback step, not historical trajectory',
        original_values_saved='likelihood_values' in arrays, accepted=accepted, backtracks=backtracks,
        old_objective=old, accepted_decrease=decrease, negligible_decrease_threshold=threshold,
        negligible_decrease=decrease <= threshold, max_step_CCF=step_max,
        same_support_and_witness=None, prior_consecutive_stagnation=None,
        history_unavailable='prior support/witness history was not stored in this single QP snapshot')


def analytic():
    from .solver import boxed_forest_quadratic
    rows = []
    h, b = np.ones((2, 1)), np.array([[.25], [.75]])
    edges, lo, hi = np.array([[0, 1]]), np.zeros_like(h), np.ones_like(h)
    for rho in (1e8, 1e10, 1e12):
        exact, high_precision = decimal_free_system(h, b, edges, rho, lo, hi, h*0)
        rounded = np.asarray(exact, dtype=float)
        result, audit = boxed_forest_quadratic(h, b, edges, rho, lo, hi)
        independent = qp_residual(h, b, edges, rho, lo, hi, rounded)
        rows.append(dict(rho=rho, decimal_coordinates=[str(v) for v in exact[:, 0]],
                         rounded_analytic_backward_error=independent['backward_error'],
                         high_precision=high_precision, native_backward_error=audit['backward_error'],
                         native_qualified=audit['solved'], precision_risk=audit['precision_risk'],
                         native_equals_rounded_analytic=bool(np.array_equal(result, rounded))))
    return dict(case='two_node_float64_representability', rows=rows,
                interpretation='A synthetic rounding control, not attribution of patient failures.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('analytic', 'qp', 'likelihood', 'all'), required=True)
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--digits', type=int, default=80)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.mode == 'analytic':
        result = analytic()
    else:
        if args.snapshot is None:
            parser.error('--snapshot is required for qp/likelihood/all')
        arrays = read_snapshot(args.snapshot, args.manifest)
        if args.mode == 'qp':
            result = replay_qp(arrays, digits=args.digits)
        elif args.mode == 'likelihood':
            result = replay_likelihood(arrays, args.cuda)
        else:
            result = dict(qp=replay_qp(arrays, digits=args.digits),
                          likelihood=replay_likelihood(arrays, args.cuda))
        result['frozen_metadata'] = json.loads(str(arrays['metadata']))
        result['snapshot_sha256'] = hashlib.sha256(args.snapshot.read_bytes()).hexdigest()
    document = json.dumps(_json_safe(result), sort_keys=True, indent=2, allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as handle:
            handle.write(document)
    print(document)


if __name__ == '__main__':
    main()
