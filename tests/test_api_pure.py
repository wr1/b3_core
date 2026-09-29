"""homogenize writes nothing unless asked, and the cache key is stable."""

import json
import logging
import subprocess
import sys

import numpy as np
import pytest
from tests.fakes import fake_backend, unregister

from b3_core.api import homogenize
from b3_core.cache import MemoryCache
from b3_core.cases import plain
from b3_core.hashing import case_hash
from b3_core.loaders import normalize_case

# plain() isotropic case, resolved backend "mfem". Bump CACHE_SCHEMA when this moves.
GOLDEN_PLAIN_MFEM = "f82dc7940c4a9b4b42300e0575a6cc0744a5af015f323eb36d499c894a829b27"


def test_homogenize_is_pure(monkeypatch, tmp_path):
    register, cls = fake_backend("fake")
    register(cls)
    monkeypatch.chdir(tmp_path)
    try:
        first = homogenize(plain(), backend="fake", name="a")
        second = homogenize(plain(), backend="fake", name="a")
    finally:
        unregister("fake")
    assert list(tmp_path.iterdir()) == []
    assert first.engineering_constants == second.engineering_constants
    assert first.resin_volume_fraction == second.resin_volume_fraction
    assert np.array_equal(first.stiffness, second.stiffness)


def test_memory_cache_skips_the_second_solve(caplog):
    solves: list[str] = []
    register, cls = fake_backend("fake", solves=solves)
    register(cls)
    cache = MemoryCache()
    try:
        with caplog.at_level(logging.INFO, logger="b3_core.cache"):
            homogenize(plain(), backend="fake", cache=cache)
            homogenize(plain(), backend="fake", cache=cache)
    finally:
        unregister("fake")
    assert solves == ["fake"]
    assert "cache hit" in caplog.text


def test_disk_cache_survives_a_fresh_process(tmp_path):
    cache = tmp_path / "cache"
    marker = tmp_path / "solved"
    script = r"""
import sys
from pathlib import Path
import numpy as np
from b3_core.api import homogenize
from b3_core.cache import DiskCache
from b3_core.cases import plain
from b3_core.solvers import register
from b3_core.solvers.protocol import Capabilities, SolveResult
from b3_core import solvers

class Fake:
    name = "fake"
    capabilities = Capabilities(True, True, True, False)
    def is_available(self):
        return True
    def solve(self, req):
        Path(sys.argv[2]).write_text("solved")
        stiffness = np.eye(6)
        properties = {
            "Ex": 1.0e9, "Ey": 1.0e9, "Ez": 1.0e9,
            "Gxy": 4.0e8, "Gxz": 4.0e8, "Gyz": 4.0e8,
            "nuxy": 0.3, "nuxz": 0.3, "nuyz": 0.3,
        }
        return SolveResult(stiffness=stiffness, properties=properties, compliance=stiffness)

register(Fake)
try:
    result = homogenize(plain(), backend="fake", cache=DiskCache(sys.argv[1]))
    print(result.material.Ex)
finally:
    solvers._REGISTRY.pop("fake", None)
"""
    first = subprocess.run(
        [sys.executable, "-c", script, str(cache), str(marker)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert first.returncode == 0, first.stderr
    assert marker.is_file()
    marker.unlink()
    second = subprocess.run(
        [sys.executable, "-c", script, str(cache), str(marker)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert second.returncode == 0, second.stderr
    assert not marker.exists()
    assert float(second.stdout.strip()) == pytest.approx(1.0e9)


def test_hash_stable_across_legacy_shapes():
    legacy = {
        "dy": 50,
        "dx": 50.0,
        "thickness": 30,
        "ygr": [],
        "xgr": [[10, 10, -8.0, 2]],
        "core": {"rho": 100, "nu": 0.3, "E": 1.0e9},
        "resin": {"E": 3e9, "nu": 0.3, "rho": 1100},
        "_note": "ignored",
    }
    typed = {
        "dx": 50,
        "dy": 50.0,
        "thickness": 30.0,
        "xgr": [{"offset": 10, "pitch": 10, "depth": 8, "width": 2, "mouth": "top"}],
        "ygr": [],
        "core": {"E": 1e9, "nu": 0.3, "rho": 100},
        "resin": {"E": 3.0e9, "nu": 0.3, "rho": 1100.0},
    }
    left, _workdir = normalize_case(legacy)
    right, _workdir = normalize_case(typed)
    assert case_hash(left, backend="numpy") == case_hash(right, backend="numpy")


def test_golden_hash_and_backend_split():
    case = plain().input
    assert case_hash(case, backend="mfem") == GOLDEN_PLAIN_MFEM
    assert case_hash(case, backend="mfem") != case_hash(case, backend="numpy")


def test_write_true_overwrites_one_file(tmp_path):
    register, cls = fake_backend("fake")
    register(cls)
    try:
        with pytest.raises(ValueError, match="workdir"):
            homogenize(plain(), backend="fake", write=True)
        homogenize(plain(), backend="fake", write=True, workdir=tmp_path)
        path = next(tmp_path.glob("run*.json"))
        path.write_text("{}\n")
        homogenize(plain(), backend="fake", write=True, workdir=tmp_path)
    finally:
        unregister("fake")
    files = list(tmp_path.glob("run*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text())
    assert payload["schema"] == "b3_core.run/1"
    assert payload["result"]["backend"] == "fake"


def test_validate_skips_when_ccx_is_missing(monkeypatch, caplog):
    from b3_core.api import run_case
    from b3_core.solvers.calculix.backend import CalculixBackend

    register, cls = fake_backend("fake")
    register(cls)
    monkeypatch.setattr(CalculixBackend, "is_available", lambda self: False)
    try:
        with caplog.at_level(logging.WARNING, logger="b3_core.cache"):
            record = run_case(plain().with_backend("fake", validate_with_ccx=True))
    finally:
        unregister("fake")
    assert record.validation is None
    assert "skipping the cross-check" in caplog.text


def test_ccx_solve_raises_when_binary_is_missing(monkeypatch):
    from b3_core.solvers.calculix.backend import CalculixBackend

    monkeypatch.setattr(CalculixBackend, "is_available", lambda self: False)
    with pytest.raises(RuntimeError, match="ccx is not on PATH"):
        CalculixBackend().solve(None)
