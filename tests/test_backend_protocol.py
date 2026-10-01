"""Solver registry, capability fallback, and the datasheet backend path."""

import logging
from pathlib import Path

import pytest
from tests.fakes import fake_backend, unregister

from b3_core.api import homogenize
from b3_core.cases import grid_scored, plain
from b3_core.solvers import get_backend
from b3_core.solvers.protocol import SolverBackend, UnknownBackendError


def test_builtin_backends_implement_the_protocol():
    for name in ("mfem", "numpy", "fenicsx", "ccx"):
        backend = get_backend(name)
        assert isinstance(backend, SolverBackend)
        assert backend.is_available() in (True, False)


def test_fake_backend_runs_through_homogenize():
    register, cls = fake_backend("fake")
    register(cls)
    try:
        result = homogenize(plain(), backend="fake")
    finally:
        unregister("fake")
    assert result.material.Ex == pytest.approx(1.0e9)
    assert result.stiffness.shape == (6, 6)


def test_unknown_backend_raises():
    with pytest.raises(UnknownBackendError):
        homogenize(plain(), backend="not_a_backend")


def test_incapable_backend_warns_and_falls_back(caplog):
    with pytest.warns(DeprecationWarning, match="orthotropic"):
        with caplog.at_level(logging.WARNING, logger="b3_core.pipeline"):
            result = homogenize(grid_scored(), backend="ccx")
    assert "halo" in caplog.text
    assert result.stiffness.shape == (6, 6)


def test_auto_prefers_mfem_for_orthotropic_halo(caplog):
    with caplog.at_level(logging.INFO, logger="b3_core.pipeline"):
        homogenize(grid_scored())
    assert "backend auto → mfem (orthotropic, halo)" in caplog.text


def test_datasheet_uses_registered_ccx(monkeypatch, tmp_path):
    from b3_core import datasheet

    calls: list[str] = []
    register, cls = fake_backend("ccx", solves=calls)
    register(cls)

    class _Fig:
        def savefig(self, path, **kwargs):
            Path(path).write_bytes(b"png")

    class _Scene:
        def __init__(self, model):
            self.model = model

        def add_phases(self):
            return self

        def add_axes(self):
            return self

        def add_modulus_surface(self):
            return self

        def isometric(self):
            return self

        def screenshot(self, path):
            Path(path).write_bytes(b"png")

        def close(self):
            return None

    monkeypatch.setattr(
        datasheet.slices, "plot_orthogonal_cuts", lambda model: (_Fig(), 1.0)
    )
    monkeypatch.setattr(datasheet.plt, "close", lambda *args, **kwargs: None)
    monkeypatch.setattr(datasheet, "CoreScene", _Scene)
    monkeypatch.setattr(datasheet, "_png_aspect", lambda path: 1.0)
    case = tmp_path / "case.json"
    try:
        plain(backend="ccx").to_json(case)
        spec = datasheet.generate(case, skip_compile=True, workdir=tmp_path)
    finally:
        unregister("ccx")
    assert calls == ["ccx"]
    assert any(
        row[0] == "backend" and row[1].startswith("ccx") for row in spec.analysis_rows
    )
