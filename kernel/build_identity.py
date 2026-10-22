"""Stdlib-only build identity, shared by packaging and runtime validation."""
import hashlib
from pathlib import Path


def backend_sources(kind):
    root = Path(__file__).parent / "csrc"
    return sorted([*root.glob("common/*.h"), *root.glob(f"{kind}/*")])


def backend_identity(kind):
    root = Path(__file__).parent / "csrc"
    manifest = "".join(f"{path.relative_to(root).as_posix()}:{hashlib.sha256(path.read_bytes()).hexdigest()}\n"
                       for path in backend_sources(kind) if path.is_file())
    return hashlib.sha256(manifest.encode()).hexdigest()
