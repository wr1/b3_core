"""Cache identity: schema, solver stamp, and the cache CLI."""

import json

import pytest
from tests.fakes import fake_backend, unregister

from b3_core.api import homogenize
from b3_core.cache import DiskCache, cache_purge, cache_rows, cache_stats
from b3_core.cases import plain
from b3_core.core import run as run_mod
from b3_core.hashing import CACHE_SCHEMA, cache_key, case_hash
from b3_core.solvers.stamps import SOURCE_DIGEST, STAMPS, source_digest


def test_cache_key_carries_schema_and_stamp():
    key = cache_key(plain().input, backend="mfem")
    assert key.schema == CACHE_SCHEMA == 2
    assert key.solver_stamp == STAMPS["mfem"]
    assert key.backend == "mfem"
    assert case_hash(plain().input, backend="mfem") == key.hexdigest()


def test_stamp_changes_the_key():
    case = plain().input
    first = case_hash(case, backend="numpy")
    second = case_hash(case, backend="numpy", solver_stamp="numpy/other")
    assert first != second


def test_schema_changes_the_key(monkeypatch):
    case = plain().input
    current = case_hash(case, backend="numpy")
    monkeypatch.setattr("b3_core.hashing.CACHE_SCHEMA", 1)
    assert case_hash(case, backend="numpy") != current


def _solve_once(tmp_path):
    register, cls = fake_backend("fake")
    register(cls)
    cache = DiskCache(tmp_path)
    try:
        homogenize(plain(), backend="fake", cache=cache)
    finally:
        unregister("fake")
    path = next(tmp_path.rglob("*.json"))
    return cache, path, json.loads(path.read_text())


def test_disk_get_misses_a_stale_stamp(tmp_path):
    cache, path, payload = _solve_once(tmp_path)
    assert payload["solver_stamp"] == "fake/unstamped"
    assert cache.get(payload["case_hash"]) is not None
    payload["solver_stamp"] = "fake/old"
    path.write_text(json.dumps(payload))
    assert cache.get(payload["case_hash"]) is None


def test_schema1_file_without_a_stamp_is_a_miss(tmp_path):
    cache, path, payload = _solve_once(tmp_path)
    payload["schema"] = "b3_core.run/1"
    payload.pop("solver_stamp")
    path.write_text(json.dumps(payload))
    assert cache.get(payload["case_hash"]) is None


def test_purge_stale_removes_only_the_old_stamp(tmp_path):
    _cache, path, payload = _solve_once(tmp_path)
    stale = path.with_name("old.json")
    old = dict(payload)
    old["solver_stamp"] = "fake/before-stamp"
    stale.write_text(json.dumps(old))
    removed = cache_purge(tmp_path, stale=True)
    assert removed == [str(stale)]
    assert path.is_file()
    assert not stale.exists()
    rows = cache_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["stale"] is False


def test_purge_requires_a_filter(tmp_path):
    with pytest.raises(ValueError, match="refusing"):
        cache_purge(tmp_path)


def test_cache_stats_counts_stale(tmp_path):
    _cache, path, payload = _solve_once(tmp_path)
    old = dict(payload)
    old["solver_stamp"] = "mfem/old"
    old["result"] = dict(old["result"])
    old["result"]["backend"] = "mfem"
    path.with_name("old.json").write_text(json.dumps(old))
    stats = cache_stats(tmp_path)
    assert stats["n"] == 2
    assert stats["n_stale"] == 1
    assert stats["by_backend"]["fake"]["n_stale"] == 0
    assert stats["by_backend"]["mfem"]["n_stale"] == 1


def test_cache_cli_stats_json(tmp_path, capsys):
    _solve_once(tmp_path)
    run_mod.cmd_cache_stats(directory=str(tmp_path), as_json=True)
    payload = json.loads(capsys.readouterr().out)
    assert payload["n"] == 1
    assert payload["n_stale"] == 0


def test_cache_cli_purge_refuses(capsys):
    with pytest.raises(SystemExit) as exc:
        run_mod.cmd_cache_purge(directory=".b3cache")
    assert exc.value.code == 2
    assert "refusing" in capsys.readouterr().err


def test_solver_source_digest_is_current():
    assert source_digest() == SOURCE_DIGEST
