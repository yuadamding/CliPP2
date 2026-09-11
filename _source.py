"""Dependency-free, OS-independent Python source identity for build and runtime.

The inventory excludes development/build trees and setup.py. Hash order is
case-sensitive POSIX-relative filename order, never the host Path ordering.
Each UTF-8 filename plus NUL is followed by its source-byte SHA-256 digest.
"""
from collections.abc import Iterable
import hashlib
from pathlib import Path, PurePath
import subprocess


def git_source_identity(root: str | Path) -> tuple[str | None, bool | None]:
    """Probe only a repository rooted here, never an enclosing checkout."""
    root = Path(root)
    if not (root / ".git").exists():
        return None, None
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        ).strip())
        return commit, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


_EXCLUDED_DIRECTORIES = frozenset({"tests", "tools", "build", "__pycache__", ".git"})


def canonical_source_name(relative: PurePath) -> str:
    """Preserve case and Unicode while normalizing native path separators."""
    if relative.anchor or ".." in relative.parts or not relative.parts:
        raise ValueError("Source identities require nonempty relative paths without '..'.")
    return relative.as_posix()


def source_inventory(root: str | Path) -> tuple[tuple[str, Path], ...]:
    """Return the same canonical, sorted Python inventory on every platform."""
    root = Path(root)
    entries = []
    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        if (not path.is_file() or path.suffix != ".py" or relative.name == "setup.py"
            or _EXCLUDED_DIRECTORIES.intersection(relative.parts)):
            continue
        entries.append((canonical_source_name(relative), path))
    return tuple(sorted(entries, key=lambda entry: entry[0]))


def fingerprint_source_entries(entries: Iterable[tuple[PurePath, bytes]]) -> str:
    """Hash relative names and byte contents independently of input ordering."""
    files = [(canonical_source_name(path), hashlib.sha256(content).digest())
             for path, content in entries]
    if len({name for name, _ in files}) != len(files):
        raise ValueError("Source inventory contains duplicate canonical filenames.")
    digest = hashlib.sha256()
    for name, content_hash in sorted(files, key=lambda entry: entry[0]):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(content_hash)
    return digest.hexdigest()


def source_fingerprint(root: str | Path) -> str:
    """Fingerprint exactly the canonical source inventory used by both callers."""
    root = Path(root)
    return fingerprint_source_entries(
        (path.relative_to(root), path.read_bytes()) for _, path in source_inventory(root)
    )
