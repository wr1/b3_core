"""Result cache. ``cache=None`` on the public API is a :class:`NullCache`."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from b3_core.result import RunRecord
from b3_core.solvers.stamps import stamp_for


class Cache(Protocol):
    def get(self, key: str) -> RunRecord | None: ...

    def put(self, key: str, record: RunRecord) -> None: ...


class NullCache:
    """Never hits. The default when the caller passes ``cache=None``."""

    def get(self, key: str) -> RunRecord | None:
        return None

    def put(self, key: str, record: RunRecord) -> None:
        return None


class MemoryCache:
    """Process-local cache. ``maxsize`` drops the least recently used key."""

    def __init__(self, maxsize: int | None = None) -> None:
        self.maxsize = maxsize
        self._data: dict[str, RunRecord] = {}
        self._order: list[str] = []

    def get(self, key: str) -> RunRecord | None:
        record = self._data.get(key)
        if record is None:
            return None
        if self.maxsize is not None and key in self._order:
            self._order.remove(key)
            self._order.append(key)
        return record

    def put(self, key: str, record: RunRecord) -> None:
        if key not in self._data:
            self._order.append(key)
        self._data[key] = record
        if self.maxsize is not None:
            while len(self._order) > self.maxsize:
                old = self._order.pop(0)
                self._data.pop(old, None)


class DiskCache:
    """One JSON file per key, written with a temp file and ``os.replace``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> RunRecord | None:
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            record = RunRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, json.JSONDecodeError, ValueError):
            return None
        if record.solver_stamp != stamp_for(record.result.backend):
            return None
        return record

    def put(self, key: str, record: RunRecord) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        payload = record.model_dump(mode="json")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, path)


def iter_cache_payloads(root: str | Path):
    """Yield ``(path, payload)`` for every JSON object under ``root``."""
    folder = Path(root)
    if not folder.is_dir():
        return
    for path in sorted(folder.rglob("*.json")):
        if path.name.endswith(".tmp") or "__pycache__" in path.parts:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError):
            continue
        if isinstance(payload, dict):
            yield path, payload


def cache_row(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """One inspect line. ``stale`` when the stored stamp is not the current one."""
    result = payload.get("result")
    result = result if isinstance(result, dict) else {}
    curv = payload.get("input")
    curv = curv.get("curvature") if isinstance(curv, dict) else None
    curv = curv if isinstance(curv, dict) else {}
    backend = str(result.get("backend") or "")
    stored = payload.get("solver_stamp")
    stored_s = stored if isinstance(stored, str) else ""
    expected = stamp_for(backend) if backend else ""
    stale = stored_s != expected
    return {
        "path": str(path),
        "case_hash": payload.get("case_hash"),
        "backend": backend,
        "solver_stamp": stored_s,
        "kx": curv.get("kx"),
        "ky": curv.get("ky"),
        "stale": stale,
    }


def cache_rows(root: str | Path, *, backend: str = "") -> list[dict[str, Any]]:
    """Inspect rows. ``backend`` keeps one resolved backend name."""
    rows = [cache_row(path, payload) for path, payload in iter_cache_payloads(root)]
    if backend:
        rows = [row for row in rows if row["backend"] == backend]
    return rows


def cache_stats(root: str | Path) -> dict[str, Any]:
    """Counts by backend, including how many stamps are no longer current."""
    rows = cache_rows(root)
    by_backend: dict[str, dict[str, int]] = {}
    for row in rows:
        name = row["backend"] or "?"
        bucket = by_backend.setdefault(name, {"n": 0, "n_stale": 0})
        bucket["n"] += 1
        if row["stale"]:
            bucket["n_stale"] += 1
    return {
        "dir": str(root),
        "n": len(rows),
        "n_stale": sum(1 for row in rows if row["stale"]),
        "by_backend": by_backend,
    }


def cache_purge(
    root: str | Path,
    *,
    stale: bool = False,
    backend: str = "",
) -> list[str]:
    """Delete matching entries. Requires ``stale`` or ``backend`` so a bare call keeps the cache."""
    if not stale and not backend:
        raise ValueError(
            "pass stale=True and/or backend=; refusing to delete every entry"
        )
    removed: list[str] = []
    for row in cache_rows(root):
        if backend and row["backend"] != backend:
            continue
        if stale and not row["stale"]:
            continue
        path = Path(row["path"])
        path.unlink(missing_ok=True)
        removed.append(str(path))
    return removed


def cache_export(root: str | Path, dest: str | Path) -> dict[str, Any]:
    """Zip the cache JSON files plus a manifest of stamps. The key is the path."""
    import zipfile

    folder = Path(root)
    target = Path(dest)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = cache_rows(folder)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for row in rows:
            path = Path(row["path"])
            archive.write(path, path.relative_to(folder).as_posix())
        archive.writestr(
            "manifest.json",
            json.dumps(
                {
                    "n": len(rows),
                    "n_stale": sum(1 for row in rows if row["stale"]),
                    "entries": [
                        {
                            "path": str(
                                Path(row["path"]).relative_to(folder).as_posix()
                            ),
                            "backend": row["backend"],
                            "solver_stamp": row["solver_stamp"],
                            "stale": row["stale"],
                        }
                        for row in rows
                    ],
                }
            ),
        )
    return {"zip": str(target), "n": len(rows)}


def cache_import(
    bundle: str | Path, root: str | Path, *, overwrite: bool = False
) -> dict[str, Any]:
    """Copy bundled entries into ``root``. Existing keys stay unless ``overwrite``."""
    import zipfile

    folder = Path(root)
    folder.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    skipped: list[str] = []
    with zipfile.ZipFile(bundle) as archive:
        for name in archive.namelist():
            if (
                name.endswith("/")
                or name == "manifest.json"
                or not name.endswith(".json")
            ):
                continue
            dest = folder / name
            if dest.exists() and not overwrite:
                skipped.append(name)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(archive.read(name))
            copied.append(name)
    return {"copied": copied, "skipped": skipped}
