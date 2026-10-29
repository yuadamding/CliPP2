"""Resolve CPU parallelism independently of scientific search limits."""
import os
from pathlib import Path


def _cgroup_paths(controller):
    """Yield the current hierarchy and ancestors (nested limits also apply)."""
    try:
        records = Path('/proc/self/cgroup').read_text().splitlines()
    except OSError:
        return
    for record in records:
        _, controllers, relative = record.split(':', 2)
        if controllers and controller not in controllers.split(','):
            continue
        base = Path('/sys/fs/cgroup') / (controller if controllers else '')
        path = base / relative.lstrip('/')
        while path == base or base in path.parents:
            yield path, not controllers
            if path == base:
                break
            path = path.parent


def resolve_cpu_threads(value=None):
    if value is None:
        raw = os.environ.get('CLIPP2_CPU_THREADS', os.environ.get('OMP_NUM_THREADS'))
        if raw is not None:
            try:
                value = int(raw)
            except ValueError as error:
                raise ValueError('CLIPP2_CPU_THREADS/OMP_NUM_THREADS must be a positive integer') from error
    if value is not None:
        if type(value) is not int or not 1 <= value <= 2**31 - 1:
            raise ValueError('cpu_threads must be a positive int32 or None (auto)')
        return value
    count = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    for directory, v2 in _cgroup_paths('cpu'):
        try:
            if v2:
                quota, period = (directory / 'cpu.max').read_text().split()
                if quota == 'max':
                    continue
            else:
                quota = (directory / 'cpu.cfs_quota_us').read_text()
                period = (directory / 'cpu.cfs_period_us').read_text()
            if int(quota) > 0:
                count = min(count, max(1, int(quota) // int(period)))
        except (OSError, ValueError):
            continue
    return max(1, count)
