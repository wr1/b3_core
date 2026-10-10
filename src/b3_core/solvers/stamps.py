"""Per-backend numerics stamps.

Bump the string for a backend in the same change that alters its numbers.
Library versions (dolfinx, mfem, ccx) stay off the cache key; they belong on
the run record. ``SOURCE_DIGEST`` is the hash of every other file in this
package. A solver edit fails ``tests/test_cache_key.py`` until that digest is
updated here, next to the stamp bump.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

STAMPS: dict[str, str] = {
    "fenicsx": "fenicsx/2026-10-10-energy",
    "mfem": "mfem/2026-10-10-uniform-master",
    "numpy": "numpy/2026-10-10-uniform-master",
    "ccx": "ccx/2026-10-10-uniform-master",
}

# sha256 of solvers/**/*.py except this file. Update in the same commit as a
# numerics change. Placeholder replaced after the first hash is computed.
SOURCE_DIGEST = "554b0b2d43958a275fd113d4c1cb2abc4a48eaaf90767c1a10d29865aa7d679a"


def stamp_for(backend: str) -> str:
    """Stamp for ``backend``. Unregistered names get a stable ``unstamped`` tag."""
    return STAMPS.get(backend, f"{backend}/unstamped")


def source_digest() -> str:
    """Hash of solver sources, excluding this file so the constant can live here."""
    root = Path(__file__).resolve().parent
    hasher = hashlib.sha256()
    paths = sorted(
        path
        for path in root.rglob("*.py")
        if path.name != "stamps.py" and "__pycache__" not in path.parts
    )
    for path in paths:
        rel = path.relative_to(root).as_posix()
        hasher.update(rel.encode())
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()
