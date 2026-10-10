"""Who produced a run record, and who is serving it.

``solved_with`` stays on the numbers. ``served_by`` is the process that
returned them, which differs after a cache hit. Commit comes from ``git``
in a checkout, otherwise from the installed distribution's ``direct_url.json``.
"""

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from b3_core import _version as _version_mod


class CodeIdentity(BaseModel):
    """One process: package version, commit, and the solver libraries it saw."""

    model_config = ConfigDict(extra="forbid")

    version: str
    commit: str | None = None
    dirty: bool | None = None
    generated_utc: str
    solver_stamp: str = ""
    backend_versions: dict[str, str | None] = Field(default_factory=dict)


class Provenance(BaseModel):
    """Block stored on a run record as ``b3_core``.

    The flat fields repeat ``solved_with``, so a reader that only looks at
    the top of the block still sees the code that produced the numbers.
    """

    model_config = ConfigDict(extra="forbid")

    version: str
    commit: str | None = None
    dirty: bool | None = None
    generated_utc: str
    solver_stamp: str = ""
    backend_versions: dict[str, str | None] = Field(default_factory=dict)
    solved_with: CodeIdentity
    served_by: CodeIdentity


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git_root() -> Path | None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".git").exists():
            return parent
    return None


def _git_identity() -> tuple[str | None, bool | None]:
    root = _git_root()
    if root is None:
        return None, None
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, None
    if head.returncode != 0:
        return None, None
    commit = head.stdout.strip() or None
    if status.returncode != 0:
        return commit, None
    return commit, bool(status.stdout.strip())


def _commit_from_metadata() -> str | None:
    try:
        from importlib.metadata import distribution

        text = distribution("b3_core").read_text("direct_url.json")
    except Exception:
        return None
    if not text:
        return None
    try:
        payload: dict[str, Any] = json.loads(text)
    except json.JSONDecodeError:
        return None
    vcs = payload.get("vcs_info")
    if not isinstance(vcs, dict):
        return None
    commit = vcs.get("commit_id")
    if isinstance(commit, str) and commit:
        return commit[:12]
    return None


def code_revision() -> tuple[str | None, bool | None]:
    """``(commit, dirty)``. A checkout wins; otherwise the installed VCS commit."""
    if _git_root() is not None:
        return _git_identity()
    return _commit_from_metadata(), None


def _module_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    version = getattr(module, "__version__", None)
    if version:
        return str(version)
    try:
        from importlib.metadata import version as dist_version

        return dist_version(name)
    except Exception:
        return "imported"


def _ccx_version() -> str | None:
    exe = shutil.which("ccx")
    if exe is None:
        return None
    try:
        proc = subprocess.run(
            [exe, "-v"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "present"
    text = f"{proc.stdout or ''}\n{proc.stderr or ''}"
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and "version" in stripped.lower():
            return stripped[:120]
    return "present"


@lru_cache(maxsize=1)
def backend_versions() -> dict[str, str | None]:
    """Versions of the solver libraries. Missing ones are ``None``."""
    return {
        "dolfinx": _module_version("dolfinx"),
        "dolfinx_mpc": _module_version("dolfinx_mpc"),
        "mfem": _module_version("mfem"),
        "ccx": _ccx_version(),
    }


def identity(*, solver_stamp: str) -> CodeIdentity:
    """Identity of this process, stamped for the backend that solved."""
    commit, dirty = code_revision()
    return CodeIdentity(
        version=_version_mod.__version__,
        commit=commit,
        dirty=dirty,
        generated_utc=_utc_now(),
        solver_stamp=solver_stamp,
        backend_versions=dict(backend_versions()),
    )


def make_provenance(*, solver_stamp: str) -> Provenance:
    """Fresh solve: the same identity solved and is serving."""
    current = identity(solver_stamp=solver_stamp)
    return Provenance(
        version=current.version,
        commit=current.commit,
        dirty=current.dirty,
        generated_utc=current.generated_utc,
        solver_stamp=current.solver_stamp,
        backend_versions=dict(current.backend_versions),
        solved_with=current,
        served_by=current,
    )


def with_server(prov: Provenance) -> Provenance:
    """Cache hit: keep ``solved_with``, replace ``served_by`` with this process."""
    served = identity(solver_stamp=prov.solver_stamp)
    return prov.model_copy(
        update={
            "served_by": served,
            "version": prov.solved_with.version,
            "commit": prov.solved_with.commit,
            "dirty": prov.solved_with.dirty,
            "generated_utc": prov.solved_with.generated_utc,
            "solver_stamp": prov.solved_with.solver_stamp,
            "backend_versions": dict(prov.solved_with.backend_versions),
        }
    )


def comment_line(prov: Provenance | None) -> str | None:
    """One CalculiX ``**`` comment. ``None`` when there is no provenance."""
    if prov is None:
        return None
    commit = prov.commit or "unknown"
    dirty = " dirty" if prov.dirty is True else ""
    return (
        f"** b3_core {prov.version} {commit}{dirty} "
        f"{prov.solver_stamp} {prov.generated_utc}\n"
    )
