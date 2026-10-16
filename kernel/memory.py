"""Per-fit acceleration allowances, independent of numerical blocking policy.

These are additional-workspace ceilings, not process RSS/VRAM limits. Resolve
against current host/container headroom once; never change the search to fit.
"""
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

from ..config import validate_budget


GIB = 1024**3


def process_memory_snapshot(torch_module=None, device='cpu'):
    """Process/context high water, explicitly not an isolated fit attribution."""
    result = {'scope': 'process_lifetime_high_water_not_isolated_fit'}
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result['peak_host_rss_bytes'] = int(rss if sys.platform == 'darwin' else rss*1024)
    except (ImportError, OSError):
        result['peak_host_rss_bytes'] = None
    if torch_module is not None:
        result.update(device_allocated_bytes=torch_module.cuda.memory_allocated(device),
                      device_reserved_bytes=torch_module.cuda.memory_reserved(device),
                      peak_device_allocated_bytes=torch_module.cuda.max_memory_allocated(device),
                      peak_device_reserved_bytes=torch_module.cuda.max_memory_reserved(device))
    return result


def _host_headroom():
    """Best-effort Linux limits; missing probes never imply unlimited memory."""
    limits = []
    for root, maximum_name, current_name in (
            (Path('/sys/fs/cgroup'), 'memory.max', 'memory.current'),
            (Path('/sys/fs/cgroup/memory'), 'memory.limit_in_bytes', 'memory.usage_in_bytes')):
        try:
            maximum = (root/maximum_name).read_text().strip()
            if maximum != 'max':
                limits.append(max(0, int(maximum)-int((root/current_name).read_text())))
        except (OSError, ValueError):
            pass
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                limits.append(int(line.split()[1])*1024)
        for line in Path('/proc/self/cgroup').read_text().splitlines():
            hierarchy, controllers, group = line.split(':', 2)
            version2 = hierarchy == '0' and not controllers
            if version2 or 'memory' in controllers.split(','):
                root = Path('/sys/fs/cgroup') if version2 else Path('/sys/fs/cgroup/memory')
                maximum_name, current_name = (('memory.max', 'memory.current') if version2 else
                                              ('memory.limit_in_bytes', 'memory.usage_in_bytes'))
                current = root / group.lstrip('/')
                if '..' in current.parts:
                    continue
                while current.is_relative_to(root):
                    try:
                        maximum = (current/maximum_name).read_text().strip()
                        if maximum != 'max':
                            limits.append(max(0, int(maximum)-int((current/current_name).read_text())))
                    except (OSError, ValueError):
                        pass
                    if current == root:
                        break
                    current = current.parent
    except (OSError, ValueError):
        pass
    return min(limits) if limits else 0


@dataclass(frozen=True)
class MemoryPolicy:
    requested_ram_bytes: int
    requested_vram_bytes: int
    ram_bytes: int
    vram_bytes: int
    initialization_bytes: int
    refit_grid_bytes: int
    refit_bytes: int
    column_bytes: int
    weight_bytes: int
    partition_bytes: int
    workspace_bytes: int
    device_column_bytes: int
    device_workspace_bytes: int
    qualify_device_weights: bool = False

    @classmethod
    def resolve(cls, ram_budget_gib=1., vram_budget_gib=0., *, device='cpu', qualify_device_weights=False):
        def as_bytes(value, name):
            value = validate_budget(value, name)
            return int(value)*GIB + int((value % 1)*GIB)
        ram = as_bytes(ram_budget_gib, 'ram_budget_gib')
        vram = as_bytes(vram_budget_gib, 'vram_budget_gib')
        host = min(ram, _host_headroom()//2)
        gpu = 0
        if vram and str(device).startswith('cuda'):
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA was requested but is unavailable; no CPU fallback')
            free, _ = torch.cuda.mem_get_info(device)
            gpu = min(vram, int(free)//2)
        # Initialization is a separate phase; its table is released before the
        # search caches are populated. Leave 25% for cache construction/batches.
        usable = host*3//4
        grid, refits, columns, weights, partitions = (usable*p//100 for p in (35, 10, 25, 10, 20))
        return cls(ram, vram, host, gpu, usable, grid, refits, columns,
                   weights, partitions, host-usable, gpu//2, gpu-gpu//2, qualify_device_weights)

    def as_dict(self):
        return asdict(self)


class ByteLRU:
    """Explicitly charged LRU; immutable/copy-on-read values are caller-owned."""
    def __init__(self, budget_bytes):
        self.budget_bytes = max(0, int(budget_bytes))
        self.entries = OrderedDict()
        self.bytes = self.peak_bytes = self.hits = self.misses = self.evictions = 0

    def get(self, key):
        entry = self.entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self.hits += 1
        self.entries.move_to_end(key)
        return entry[0]

    def put(self, key, value, size):
        if size > self.budget_bytes:
            return
        old = self.entries.pop(key, None)
        if old is not None:
            self.bytes -= old[1]
        while self.entries and self.bytes+size > self.budget_bytes:
            _, (_, removed) = self.entries.popitem(last=False)
            self.bytes -= removed
            self.evictions += 1
        self.entries[key] = (value, size)
        self.bytes += size
        self.peak_bytes = max(self.peak_bytes, self.bytes)

    def stats(self):
        return dict(budget_bytes=self.budget_bytes, bytes=self.bytes,
                    peak_bytes=self.peak_bytes, hits=self.hits,
                    misses=self.misses, evictions=self.evictions)
