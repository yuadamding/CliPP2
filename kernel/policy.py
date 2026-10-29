"""One public execution policy; scientific math and search remain elsewhere.

No qualification is inferred from version labels or a passing small benchmark.
Absent, stale, incomplete or hardware-ambiguous attestations keep auto on the
reference route. Explicit CUDA never substitutes CPU likelihood arithmetic.
"""
from dataclasses import dataclass
import importlib.metadata
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

from .._source import file_sha256
from .build_identity import backend_identity, native_identity
from .execution import resolve_cpu_threads

CANDIDATE_POLICY_ID = 'cuda_candidate_resident_weights_v1'
QUALIFICATION_METHOD = 'median_ratio_with_log_mean_t_upper95_v1'
REQUIRED_SPEED_ANCHORS = ((500, 2), (1000, 3), (2000, 4),
                          (5000, 2), (10000, 4), (20000, 8))
MODEL_OPTIONS = dict(resident_weights=True, probability_reuse=True)
SEARCH_OPTIONS = dict(weight_batch_size=32, coarsening_batch_size=32,
    continuation_batch_size=4, weight_batch_state_bytes=128*1024**2, device_proposals=True)
_CACHE_FIELDS = ('grid_cache_mb', 'device_cache_mb', 'workspace_mb',
                 'native_host_memory_mb', 'device_memory_mb', 'cpu_threads')


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def paired_ratio_statistics(ratios):
    """The predeclared median and one-sided 95% log-mean Student-t bound."""
    _check(len(ratios) >= 5 and all(type(value) in (int, float) and math.isfinite(value) and value > 0
                                  for value in ratios), 'invalid paired timing ratios')
    from scipy.stats import t
    logs = [math.log(value) for value in ratios]
    upper_log = statistics.mean(logs)+float(t.ppf(.95, len(ratios)-1))*statistics.stdev(logs)/math.sqrt(len(ratios))
    _check(upper_log <= math.log(sys.float_info.max), 'uncertainty bound is unbounded')
    return statistics.median(ratios), math.exp(upper_log)


def validate_release_panel(manifest):
    """Require the full declared speed panel before any automatic promotion.

    Case names are unrestricted identifiers, not a substitute for qualified
    dimensions. Capacity-only controls cannot enlarge the promotion envelope.
    These limits apply to attestations, never to an explicitly requested fit.
    """
    cases = manifest['cases']
    _check(isinstance(cases, (list, tuple)) and cases, 'release panel is empty')
    identities, speed_shapes = [], []
    for case in cases:
        identity = case['case_id']
        _check(isinstance(identity, str) and identity and identity == identity.strip()
               and identity not in ('.', '..') and not any(char in identity for char in '/\\\t\r\n'),
               'release case IDs must be nonempty, safe identifiers')
        _check(all(type(case[key]) is int and case[key] > 0 for key in ('n', 'r')),
               'release case dimensions must be positive integers')
        _check(type(case['requires_speed']) is bool, 'release speed designation must be boolean')
        identities.append(identity)
        if case['requires_speed']:
            speed_shapes.append((case['n'], case['r']))
    _check(len(set(identities)) == len(identities), 'release case IDs are duplicated')
    _check(set(REQUIRED_SPEED_ANCHORS).issubset(speed_shapes),
           'required moderate/large speed anchors are missing or relabeled')
    bounds = manifest['attestation']['workload_envelope']
    _check(all(type(bounds[key]) is int and bounds[key] > 0 for key in
               ('n_min', 'n_max', 'r_min', 'r_max', 'max_clusters', 'max_major_cn')),
           'invalid qualified workload envelope')
    for axis, index in (('n', 0), ('r', 1)):
        minimum, maximum = min(shape[index] for shape in speed_shapes), max(shape[index] for shape in speed_shapes)
        _check(minimum <= bounds[axis+'_min'] <= bounds[axis+'_max'] <= maximum,
               'workload envelope exceeds speed-qualified dimensions')
    for key in ('max_clusters', 'max_major_cn'):
        values = [manifest['fit_settings'][device][key] for device in ('cpu', 'cuda')]
        _check(all(type(value) is int and value > 0 for value in values) and values[0] == values[1],
               'qualified CPU/CUDA scientific limits differ')
        _check(bounds[key] <= values[0], 'workload envelope exceeds qualified '+key)


def qualification_passes(record):
    """Validate the complete declared release report, never an aggregate mean."""
    _check(record['schema'] == 'clipp2.gpu-release.v1' and record['passed'] is True
           and record['predeclared'] is True, 'release gate is absent or failed')
    _check(record['method'] == QUALIFICATION_METHOD, 'unqualified statistical method')
    rounds = record['rounds']
    _check(type(rounds) is int and rounds >= 5, 'at least five fixed paired rounds are required')
    identifiers = record['required_case_ids']
    cases = record['cases']
    _check(identifiers and all(isinstance(value, str) and value for value in identifiers)
           and len(set(identifiers)) == len(identifiers), 'declared cases are empty or duplicated')
    _check(isinstance(record['manifest_sha256'], str) and len(record['manifest_sha256']) == 64,
           'qualified manifest identity is absent')
    manifest = record['declared_manifest']
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':'),
                                      allow_nan=False).encode()).hexdigest()
    _check(digest == record['manifest_sha256'], 'declared release manifest was changed')
    _check(manifest['method'] == QUALIFICATION_METHOD and manifest['rounds'] == rounds,
           'release method or fixed repetitions differ from declaration')
    _check(manifest['options']['cuda']['model'] == MODEL_OPTIONS
           and manifest['options']['cuda']['search'] == SEARCH_OPTIONS,
           'release panel did not use this candidate policy')
    validate_release_panel(manifest)
    declarations = manifest['cases']
    _check(len(declarations) == len(identifiers)
           and {row['case_id'] for row in declarations} == set(identifiers),
           'declared case membership differs from release results')
    _check(len(cases) == len(identifiers) and {row['case_id'] for row in cases} == set(identifiers),
           'required cases were omitted or replaced')
    required_speed = 0
    for case in cases:
        _check(case['scientific_passed'] is True and case['measurement_accepted'] is True,
               'a required case failed scientific or measurement validation')
        ratios = case['ratios']
        _check(len(ratios) == rounds and all(type(x) in (int, float) and math.isfinite(x) and x > 0
                                           for x in ratios), 'incomplete or invalid paired ratios')
        _check(case['order_counts'] in ([rounds//2, rounds-rounds//2],
                                       [rounds-rounds//2, rounds//2]), 'unbalanced paired order')
        _check(case['median_ratio'] == statistics.median(ratios), 'reported median disagrees with pairs')
        upper = case['log_mean_upper95_ratio']
        _check(type(upper) in (int, float) and math.isfinite(upper) and upper > 0,
               'invalid uncertainty bound')
        declaration, = [row for row in declarations if row['case_id'] == case['case_id']]
        _check(type(case['requires_speed']) is bool
               and case['requires_speed'] == declaration['requires_speed'], 'speed-case designation is invalid')
        _, expected_upper = paired_ratio_statistics(ratios)
        _check(math.isclose(upper, expected_upper, rel_tol=1e-12, abs_tol=1e-15),
               'reported uncertainty bound disagrees with paired timings')
        if case['requires_speed']:
            required_speed += 1
            _check(case['median_ratio'] < 1 and expected_upper < 1,
                   'a required case has no qualified CUDA advantage')
    _check(required_speed > 0, 'no moderate/large speed cases were qualified')


def config_identity(config):
    return {name: getattr(config, name) for name in _CACHE_FIELDS}


def _runtime_identity(config):
    """Metadata only; promotion requires unambiguous single-GPU UUID visibility."""
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    _check(visible.startswith('GPU-') and ',' not in visible,
           'unproven_gpu_identity: auto promotion requires one explicit GPU UUID')
    output = subprocess.check_output(['nvidia-smi', '-i', visible,
        '--query-gpu=uuid,name,driver_version', '--format=csv,noheader'], text=True)
    rows = [row.strip() for row in output.splitlines() if row.strip()]
    _check(len(rows) == 1, 'GPU identity is ambiguous')
    uuid, name, driver = [part.strip() for part in rows[0].split(',')]
    _check(uuid == visible, 'visible GPU UUID disagrees with hardware')
    import scipy.optimize._slsqp_py as wrapper
    import scipy.optimize._slsqplib as optimizer
    from threadpoolctl import threadpool_info
    pools = threadpool_info()
    _check(pools, 'loaded BLAS identity is unavailable')
    return dict(gpu=dict(uuid=uuid, name=name, driver_version=driver),
        python_sha256=file_sha256(sys.executable), python_version=sys.version,
        numpy=importlib.metadata.version('numpy'), scipy=importlib.metadata.version('scipy'),
        optimizer={Path(module.__file__).name: file_sha256(module.__file__) for module in (wrapper, optimizer)},
        threadpools=sorted([dict(api=row['user_api'], implementation=row['internal_api'],
            prefix=row['prefix'], version=row.get('version'), threads=row['num_threads'],
            binary_sha256=file_sha256(row['filepath'])) for row in pools], key=lambda row: row['prefix']),
        cpu_threads=resolve_cpu_threads(config.cpu_threads),
        selectors={key: os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER',
            'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'CLIPP2_CPU_THREADS')})


def _native_binding(root):
    kernel = root/'kernel'
    paths = sorted(kernel.glob('_native*.so'))
    _check(len(paths) == 3 and not any(path.is_symlink() for path in paths),
           'exact native artifact inventory is unavailable')
    return dict(build_ids=dict(forest=native_identity(root), cpu=backend_identity('cpu'),
                               cuda=backend_identity('cuda')),
                binaries={path.name: file_sha256(path) for path in paths})


def _eligible_document(record, config, data, root, source_hash):
    _check(record['schema'] == 'clipp2.gpu-policy.v1'
           and record['candidate_policy_id'] == CANDIDATE_POLICY_ID, 'unknown policy attestation')
    _check(record['source_sha256'] == source_hash, 'attestation source is stale')
    qualification_passes(record['qualification'])
    declaration = record['qualification']['declared_manifest']
    _check(declaration['source']['fingerprint'] == source_hash,
           'release panel source differs from policy source')
    _check(all(record[name] == declaration['attestation'][name] for name in
               ('native', 'runtime', 'config', 'workload_envelope')),
           'policy identities differ from declared release contract')
    _check(record['native'] == _native_binding(root), 'native binaries or source identities differ')
    _check(record['config'] == config_identity(config), 'cache or execution budgets differ')
    bounds = record['workload_envelope']
    _check(all(type(bounds[key]) is int and bounds[key] > 0 for key in
               ('n_min', 'n_max', 'r_min', 'r_max', 'max_clusters', 'max_major_cn')),
           'invalid workload envelope')
    n, r = len(data.mutation_ids), len(data.region_ids)
    _check(bounds['n_min'] <= n <= bounds['n_max'] and bounds['r_min'] <= r <= bounds['r_max']
           and config.max_clusters <= bounds['max_clusters']
           and config.max_major_cn <= bounds['max_major_cn'], 'workload is outside the declared envelope')
    _check(record['runtime'] == _runtime_identity(config), 'GPU or runtime environment differs')


@dataclass(frozen=True)
class ExecutionPolicy:
    requested: str
    resolved: str
    reason: str
    attestation_sha256: str | None = None

    @property
    def model_options(self):
        return dict(MODEL_OPTIONS) if self.resolved == 'candidate' else {}

    @property
    def search_options(self):
        return dict(SEARCH_OPTIONS) if self.resolved == 'candidate' else {}

    def verify_attestation(self, source_root):
        """Publication must retain the JSON authority used to select auto."""
        if self.attestation_sha256 is None:
            return
        path = Path(source_root)/'_gpu_policy.json'
        try:
            valid = not path.is_symlink() and path.is_file() and file_sha256(path) == self.attestation_sha256
        except OSError as error:
            raise RuntimeError('GPU execution attestation changed during fitting; no successful publication') from error
        if not valid:
            raise RuntimeError('GPU execution attestation changed during fitting; no successful publication')

    def record(self, config, numerical_policy, model=None):
        from .scoring import WEIGHT_CONTROLLER_ID
        resident = self.resolved == 'candidate' if model is None else bool(model._resident_weights)
        return dict(requested=self.requested, resolved=self.resolved, reason=self.reason,
            candidate_policy_id=CANDIDATE_POLICY_ID if self.resolved == 'candidate' else None,
            controller='resident_portable_blas_v1' if resident else 'reference_scipy_slsqp_brent',
            controller_identity=(WEIGHT_CONTROLLER_ID, 'resident_portable_blas_v1') if resident else WEIGHT_CONTROLLER_ID,
            numerical_policy=numerical_policy,
            initialization='reference',
            external_override=resident != (self.resolved == 'candidate'),
            model_options=self.model_options,
            batch_limits=self.search_options or dict(weight_batch_size=1, coarsening_batch_size=1,
                continuation_batch_size=1, weight_batch_state_bytes=8*1024**2,
                device_proposals=False), cache_budgets=config_identity(config),
            attestation_sha256=self.attestation_sha256)


def resolve_execution_policy(config, data, source_root, source_hash):
    """Resolve without constructing another model or touching public outputs."""
    requested = config.execution_policy
    if requested == 'reference' or config.device == 'cpu':
        return ExecutionPolicy(requested, 'reference', 'explicit reference' if requested == 'reference' else 'CPU reference')
    if requested == 'candidate':
        return ExecutionPolicy(requested, 'candidate', 'explicit unpromoted CUDA research policy')
    path = Path(source_root)/'_gpu_policy.json'
    if not path.is_file():
        return ExecutionPolicy(requested, 'reference', 'no source-bound complete-fit GPU attestation')
    try:
        _check(not path.is_symlink(), 'policy attestation may not be a symlink')
        digest = file_sha256(path)
        _eligible_document(json.loads(path.read_text()), config, data, Path(source_root), source_hash)
        _check(file_sha256(path) == digest, 'policy attestation changed while validating')
        return ExecutionPolicy(requested, 'candidate', 'complete declared GPU panel qualified', digest)
    except (OSError, KeyError, TypeError, ValueError, ImportError, subprocess.SubprocessError) as error:
        return ExecutionPolicy(requested, 'reference', 'GPU attestation not applicable: '+str(error))
