"""Result cache. ``cache=None`` on the public API is a :class:`NullCache`."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol

from b3_core.result import RunRecord


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
        return RunRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def put(self, key: str, record: RunRecord) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        payload = record.model_dump(mode="json")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, path)
