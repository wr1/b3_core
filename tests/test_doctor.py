"""FEniCSx doctor: missing stack in this env, and a stubbed success path."""

import json

import pytest

from b3_core.core import run as run_mod
from b3_core.doctor import _cube_mesh, _periodic_cube, _version, doctor
from b3_core.solvers import fenicsx


def test_doctor_reports_a_missing_stack():
    report = doctor()
    if fenicsx.is_fenicsx_available():
        pytest.skip("FEniCSx imports in this environment")
    assert report["ok"] is False
    assert report["imports"]["dolfinx"] is None
    assert report["solve"] is None
    assert report["error"]
    assert report["factorisation"] == "lu/mumps"


def test_cmd_doctor_json_exits_when_stack_is_missing(capsys):
    if fenicsx.is_fenicsx_available():
        pytest.skip("FEniCSx imports in this environment")
    with pytest.raises(SystemExit) as ei:
        run_mod.cmd_doctor(as_json=True)
    assert ei.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "dolfinx" in payload["imports"]


def test_version_falls_back_when_the_distribution_is_absent():
    assert _version(object(), ("not-a-real-distribution",)) == "unknown"


def test_cube_mesh_is_two_by_two_by_two():
    grid = _cube_mesh()
    assert grid.n_cells == 8
    assert len(grid.cell_data["resin"]) == 8
    assert len(grid.cell_data["face"]) == 8


def test_periodic_cube_uses_the_solver(monkeypatch):
    class _Grid:
        n_cells = 8

    monkeypatch.setattr("b3_core.doctor._cube_mesh", lambda: _Grid())
    monkeypatch.setattr(
        "b3_core.solvers.fenicsx.runfenicsx",
        lambda *_args, **_kwargs: {"Ex": 4.0e9},
    )
    out = _periodic_cube()
    assert out["ok"] is True
    assert out["cells"] == 8
    assert out["rel_error"] == pytest.approx(0.0)


def test_doctor_accepts_a_working_stack(monkeypatch):
    names = ("dolfinx", "dolfinx_mpc", "petsc4py", "mpi4py", "ufl", "basix")
    monkeypatch.setattr(
        "b3_core.doctor.probe_imports",
        lambda: ({name: "1" for name in names}, None),
    )
    monkeypatch.setattr("b3_core.doctor._mpi_and_mumps", lambda: (1, True))
    monkeypatch.setattr(
        "b3_core.doctor._periodic_cube",
        lambda: {"ok": True, "cells": 8, "Ex": 4.0e9, "rel_error": 0.0},
    )
    report = doctor()
    assert report["ok"] is True
    assert report["mpi_size"] == 1
    assert report["mumps"] is True
    assert report["error"] is None


def test_doctor_rejects_a_bad_modulus(monkeypatch):
    names = ("dolfinx", "dolfinx_mpc", "petsc4py", "mpi4py", "ufl", "basix")
    monkeypatch.setattr(
        "b3_core.doctor.probe_imports",
        lambda: ({name: "1" for name in names}, None),
    )
    monkeypatch.setattr("b3_core.doctor._mpi_and_mumps", lambda: (1, True))
    monkeypatch.setattr(
        "b3_core.doctor._periodic_cube",
        lambda: {"ok": False, "cells": 8, "Ex": 1.0, "rel_error": 1.0},
    )
    report = doctor()
    assert report["ok"] is False
    assert "rel_error" in report["error"]


def test_is_fenicsx_available_caches_the_import(monkeypatch):
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise ImportError("missing")

    fenicsx.is_fenicsx_available.cache_clear()
    monkeypatch.setattr(fenicsx, "_import_fenicsx_stack", boom)
    try:
        assert fenicsx.is_fenicsx_available() is False
        assert fenicsx.is_fenicsx_available() is False
        assert calls["n"] == 1
    finally:
        fenicsx.is_fenicsx_available.cache_clear()


@pytest.mark.fenicsx
@pytest.mark.skipif(
    not fenicsx.is_fenicsx_available(), reason="FEniCSx is not installed"
)
def test_doctor_solves_when_the_stack_imports():
    report = doctor()
    assert report["ok"] is True
    assert report["mumps"] is True
    assert report["solve"]["cells"] == 8
    assert report["solve"]["rel_error"] < 1e-4
