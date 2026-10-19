"""Original-QP attribution and bounded, opt-in reproducible failure evidence.

Diagnostics never relax admission, change the fixed-forest objective, or turn
an observed precision risk into proof of a biological run's root cause.
"""
import hashlib
import io
import json
from pathlib import Path
from threading import Lock

import numpy as np


QP_TOLERANCE = 1e-7


def qp_residual(curvature, rhs, edges, penalty, lower, upper, result):
    """Independent componentwise KKT audit in extended accumulation precision."""
    h, b, lo, hi, x = [np.asarray(a) for a in (curvature, rhs, lower, upper, result)]
    for name, values in zip(('curvature', 'rhs', 'lower', 'upper', 'result'), (h, b, lo, hi, x)):
        indices = np.argwhere(~np.isfinite(values))
        if len(indices):
            return dict(backward_error=float('inf'), bound_violation=float('inf'),
                        first_nonfinite={'array': name, 'coordinate': indices[0].tolist()},
                        failure_reasons=['nonfinite_result'])
    xx, hh, bb = [a.astype(np.longdouble) for a in (x, h, b)]
    base = hh * xx
    gradient = base - bb
    scale = 1 + np.abs(base) + np.abs(bb)
    jumps = np.longdouble(penalty) * (xx[edges[:, 0]] - xx[edges[:, 1]])
    np.add.at(gradient, edges[:, 0], jumps)
    np.add.at(gradient, edges[:, 1], -jumps)
    np.add.at(scale, edges[:, 0], np.abs(jumps))
    np.add.at(scale, edges[:, 1], np.abs(jumps))
    residual = np.where(x == lo, np.minimum(gradient, 0), gradient)
    residual = np.where(x == hi, np.maximum(residual, 0), residual)
    residual[lo == hi] = 0
    relative = np.abs(residual) / scale
    worst = np.unravel_index(np.argmax(relative), x.shape)
    error = float(relative[worst])
    violation = float(max(np.max(lo - x), np.max(x - hi), 0.0))
    # ULP sensitivity is an upper bound on coordinate rounding's contribution,
    # not proof that this particular solve has attained a representation limit.
    spacing = np.abs(np.spacing(x)).astype(np.longdouble)
    rounding = np.abs(hh) * spacing / 2
    edge_rounding = np.longdouble(penalty) * (spacing[edges[:, 0]] + spacing[edges[:, 1]]) / 2
    np.add.at(rounding, edges[:, 0], edge_rounding)
    np.add.at(rounding, edges[:, 1], edge_rounding)
    rounding[lo == hi] = 0
    reasons = []
    if error > QP_TOLERANCE:
        reasons.append('kkt_backward_error_exceeded')
    if violation:
        reasons.append('bound_violation')
    return dict(backward_error=error, bound_violation=violation,
                projected_qp_stationarity_CCF=float(np.max(np.abs(xx-np.clip(xx-gradient, lo, hi)))),
                regional_backward_error=np.max(relative, axis=0).astype(float).tolist(),
                worst_coordinate=list(map(int, worst)), worst_residual=float(residual[worst]),
                worst_gradient=float(gradient[worst]), worst_scale=float(scale[worst]),
                worst_diagonal_term=float(base[worst]), worst_rhs=float(bb[worst]),
                worst_coordinate_spacing=float(spacing[worst]),
                worst_rounding_relative_bound=float((rounding / scale)[worst]),
                precision_risk=bool(error > QP_TOLERANCE and (rounding / scale)[worst] >= QP_TOLERANCE),
                first_nonfinite=None, failure_reasons=reasons)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


class FailureSnapshots:
    """One run-local first/worst capture with a strict total on-disk byte cap.

    The configured directory must be absent or empty. Full input/model arrays
    and returned active sets are captured, not sampled rows. If two complete
    captures do not fit, the skipped event is reported without partial arrays.
    """
    def __init__(self, directory, max_bytes):
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, (int, np.integer)) or max_bytes < 1:
            raise ValueError('Numerical diagnostics byte budget must be a positive integer')
        self.directory = Path(directory)
        if self.directory.exists() and any(self.directory.iterdir()):
            raise FileExistsError('Numerical diagnostics directory must be absent or empty')
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_bytes)
        self.lock = Lock()
        self.files = {}
        self.events = self.skipped = 0
        self.worst = -np.inf
        from .._source import git_source_identity, source_fingerprint
        from .native import native_library
        root = Path(__file__).resolve().parents[1]
        commit, dirty = git_source_identity(root)
        library, native_sha = native_library()
        self.identity = dict(commit=commit, dirty=dirty, python_source_sha256=source_fingerprint(root),
                             native_source_sha256=native_sha, native_abi=library.CliPP2KernelABI(),
                             native_binary_sha256=hashlib.sha256(Path(library._name).read_bytes()).hexdigest(),
                             longdouble_mantissa_bits=int(np.finfo(np.longdouble).nmant))

    def record(self, arrays, metadata, severity):
        with self.lock:
            self.events += 1
            if self.files and severity <= self.worst:
                return
            # Account before allocating the zip buffer. The conservative ZIP
            # overhead allowance also bounds observer memory for rejected cases.
            if sum(np.asarray(v).nbytes for v in arrays.values()) + 32_768 > self.max_bytes // 2:
                self.skipped += 1
                return
            document = _json_safe(dict(metadata, source=self.identity, schema=2,
                quadratic_definition='diag(H) + rho * forest_laplacian; linear term -b',
                anchor_augmentation_already_in_H_and_b=True,
                returned_coordinate_dtype=str(np.asarray(arrays['result']).dtype)))
            buffer = io.BytesIO()
            np.savez(buffer, **arrays, metadata=np.asarray(json.dumps(document, sort_keys=True)))
            payload = buffer.getvalue()
            names = ['first.npz', 'worst.npz'] if not self.files else ['worst.npz']
            proposed = dict(self.files)
            for name in names:
                proposed[name] = dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
            manifest = json.dumps(dict(schema=1, byte_budget=self.max_bytes, snapshots=proposed,
                                       meaning='first and worst observed failure; not a convergence certificate'),
                                  sort_keys=True).encode() + b'\n'
            if sum(v['bytes'] for v in proposed.values()) + len(manifest) > self.max_bytes:
                self.skipped += 1
                return
            for name in names:
                (self.directory / name).write_bytes(payload)
            (self.directory / 'manifest.json').write_bytes(manifest)
            self.files, self.worst = proposed, severity

    def summary(self):
        return dict(directory=str(self.directory), max_bytes=self.max_bytes,
                    snapshot_bytes=sum(v['bytes'] for v in self.files.values()),
                    events=self.events, skipped_for_budget=self.skipped,
                    snapshots=dict(self.files))


def likelihood_snapshot_arrays(model, x):
    """Freeze only likelihood inputs, not mutable cache or credential state."""
    result = {'X': np.asarray(x)}
    for name in ('alt', 'depth', 'major', 'normal', 'total', 'purity', 'observed'):
        if hasattr(model, name):
            result['model_' + name] = np.asarray(getattr(model, name))
    for name in ('mutation_ids', 'region_ids'):
        if hasattr(model, name):
            result['model_' + name] = np.asarray(getattr(model, name), dtype=str)
    return result
