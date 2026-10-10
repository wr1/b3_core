"""Provenance on run records: solved_with stays, served_by follows the process."""

import json

import numpy as np
from tests.fakes import fake_backend, unregister

from b3_core import __version__
from b3_core.api import run_case
from b3_core.cache import MemoryCache
from b3_core.cases import plain
from b3_core.export.ccx import ccx_ortho_card
from b3_core.provenance import comment_line, make_provenance
from b3_core.result import CoreResult, RunRecord


def test_fresh_record_carries_provenance():
    register, cls = fake_backend("fake")
    register(cls)
    try:
        record = run_case(plain(), backend="fake")
    finally:
        unregister("fake")
    assert record.schema_name == "b3_core.run/2"
    block = record.provenance
    assert block is not None
    assert block.version == __version__
    assert block.solved_with.version == __version__
    assert block.served_by.version == __version__
    assert block.solver_stamp == "fake/unstamped"
    assert block.commit
    assert isinstance(block.dirty, bool)
    assert "mfem" in block.backend_versions
    dumped = record.model_dump(mode="json")
    assert dumped["schema"] == "b3_core.run/2"
    assert dumped["b3_core"]["solved_with"]["solver_stamp"] == "fake/unstamped"


def test_cache_hit_refreshes_served_by(monkeypatch):
    register, cls = fake_backend("fake")
    register(cls)
    cache = MemoryCache()
    try:
        first = run_case(plain(), backend="fake", cache=cache)
        monkeypatch.setattr(
            "b3_core.provenance._version_mod.__version__",
            "9.9.9",
        )
        second = run_case(plain(), backend="fake", cache=cache)
    finally:
        unregister("fake")
    assert first.provenance is not None
    assert second.provenance is not None
    assert second.provenance.solved_with.version == first.provenance.solved_with.version
    assert second.provenance.version == first.provenance.version
    assert second.provenance.served_by.version == "9.9.9"
    assert cache.get(first.case_hash).provenance.served_by.version != "9.9.9"


def test_run1_record_loads_without_provenance():
    record = RunRecord.model_validate(
        {
            "schema": "b3_core.run/1",
            "case_hash": "abc",
            "b3_core_version": "0.3.0",
            "input": {},
            "geometry": {},
            "result": {
                "backend": "numpy",
                "stiffness": [[0.0] * 6 for _ in range(6)],
                "properties": {"Ex": 1.0},
            },
        }
    )
    assert record.schema_name == "b3_core.run/1"
    assert record.provenance is None
    assert record.solver_stamp == ""


def test_legacy_flat_includes_the_block():
    record = run_case_or_skip()
    flat = record.flat()
    assert flat["b3_core"]["commit"] == record.provenance.commit


def run_case_or_skip():
    register, cls = fake_backend("fake2")
    register(cls)
    try:
        return run_case(plain(), backend="fake2")
    finally:
        unregister("fake2")


def test_ccx_comment_does_not_change_the_elastic_body():
    C = np.eye(6)
    plain_card = ccx_ortho_card(C, name="core_hom", rho=180.0)
    prov = make_provenance(solver_stamp="numpy/test")
    commented = ccx_ortho_card(
        C,
        name="core_hom",
        rho=180.0,
        provenance_comment=comment_line(prov),
    )
    assert commented.startswith(plain_card)
    assert "*elastic,type=ortho" in commented
    assert "** b3_core" in commented
    result = CoreResult.from_record(
        RunRecord(
            case_hash="abc",
            b3_core_version=__version__,
            solver_stamp="numpy/test",
            provenance=prov,
            input={},
            geometry={
                "rho_infused": 180.0,
                "resin_vf": 0.0,
                "area_increase": 1.0,
            },
            result={
                "backend": "numpy",
                "stiffness": np.eye(6).tolist(),
                "properties": {
                    "Ex": 1e9,
                    "Ey": 1e9,
                    "Ez": 1e9,
                    "Gxy": 4e8,
                    "Gxz": 4e8,
                    "Gyz": 4e8,
                    "nuxy": 0.3,
                    "nuxz": 0.3,
                    "nuyz": 0.3,
                },
            },
        )
    )
    card = result.ccx_ortho()
    assert card.startswith("*material,name=")
    assert "** b3_core" in card
    json.dumps(result.provenance.model_dump(mode="json"))
