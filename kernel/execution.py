"""Finite execution budgets, independent of scientific search limits.

Caps cover tracked native allocations, not Python, BLAS, allocator metadata or
other processes. Auto headroom uses both OS availability and cgroup allowances;
the resolved limits are immutable for one backend context.
"""
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


def host_headroom():
    """Available bytes now; no process-lifetime or elapsed-time restriction."""
    try:
        entries = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        available = int(entries['MemAvailable'].split()[0]) * 1024
    except (OSError, KeyError, ValueError):
        available = int(os.sysconf('SC_AVPHYS_PAGES')) * int(os.sysconf('SC_PAGE_SIZE'))
    for directory, v2 in _cgroup_paths('memory'):
        try:
            limit = (directory / ('memory.max' if v2 else 'memory.limit_in_bytes')).read_text().strip()
            if limit == 'max':
                continue
            used = int((directory / ('memory.current' if v2 else 'memory.usage_in_bytes')).read_text())
            available = min(available, max(0, int(limit) - used))
        except (OSError, ValueError):
            continue
    return max(0, available)


def resolve_memory_budget(value, *, tracked=0, free=None, device=False):
    if value is not None:
        if type(value) is not int or not 0 < value < 2**63:
            raise ValueError('Native memory budget must be a positive int64 byte count or None (auto)')
        return value
    if free is None:
        if device:
            raise ValueError('Auto device memory requires measured device headroom')
        free = host_headroom()
    return min(2**63 - 1, int(tracked) + int(free) * (80 if device else 75) // 100)


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


def preflight_host_export(required):
    """Caller-owned exports are separate from the native cap, but preflighted."""
    available = host_headroom()
    if required > available:
        from .backend import NativeExecutionError
        raise NativeExecutionError(f'Host export requires {required} bytes; available headroom {available} bytes')
