"""Stdlib-only build identity, shared by packaging and runtime validation."""
import hashlib
from pathlib import Path


NATIVE_SOURCES = ('kernel/csrc/forest.cpp', 'kernel/csrc/bindings.cpp',
                  'kernel/csrc/continuation.cpp', 'kernel/csrc/forest.h',
                  'kernel/csrc/continuation.h', *tuple(
                      path.relative_to(Path(__file__).resolve().parent.parent).as_posix()
                      for path in sorted((Path(__file__).resolve().parent/'csrc/common').glob('*.h'))))


def native_identity(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parent.parent
    return hashlib.sha256(b''.join(name.encode()+b'\0'+(root/name).read_bytes()
                                   for name in NATIVE_SOURCES)).hexdigest()


def backend_sources(kind):
    root = Path(__file__).parent / "csrc"
    return sorted([*root.glob("common/*.h"), *root.glob(f"{kind}/*")])


def backend_identity(kind):
    root = Path(__file__).parent / "csrc"
    manifest = "".join(f"{path.relative_to(root).as_posix()}:{hashlib.sha256(path.read_bytes()).hexdigest()}\n"
                       for path in backend_sources(kind) if path.is_file())
    return hashlib.sha256(manifest.encode()).hexdigest()
