"""Cprop validation / helpers without full backend sweeps."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from b3_core.core import cprop as cprop_mod
from b3_core.core.cprop import CpropInput, halo_reach, homogenize, load_case


def _base(**kw):
    d = {
        "dx": 30.0,
        "dy": 30.0,
        "thickness": 20.0,
        "xgr": [],
        "ygr": [],
        "core": {"E": 1e9, "nu": 0.3, "rho": 100.0},
        "resin": {"E": 3e9, "nu": 0.3, "rho": 1100.0},
    }
    d.update(kw)
    return d


def test_cprop_input_validators():
    with pytest.raises(ValidationError):
        CpropInput(**_base(element_type="C3D99"))
    with pytest.raises(ValidationError):
        CpropInput(**_base(backend="abaqus-solver"))
    with pytest.raises(ValidationError):
        CpropInput(**_base(backend=""))
    assert CpropInput(**_base(backend="abaqus")).backend == "abaqus"
    with pytest.raises(ValidationError):
        CpropInput(**_base(xgr=[[1, 2, 3]]))  # not 4-tuple
    with pytest.raises(ValidationError):
        CpropInput(**_base(curvature={"kz": 0.1}))
    with pytest.raises(ValidationError):
        CpropInput(**_base(curvature={"kx": "nope"}))
    ok = CpropInput(**_base(backend="numpy", curvature={"kx": 0.0, "ky": 0.0}))
    assert ok.backend == "numpy"


def test_load_case_dict_and_type_error(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps(_base()))
    dct, dirname = load_case(str(path))
    assert dct["dx"] == 30.0
    assert dirname == str(tmp_path)
    with pytest.raises(TypeError, match="Textile"):
        cprop_mod.cprop(123)  # type: ignore[arg-type]


def test_halo_reach_and_needs_numpy():
    plain = _base()
    assert halo_reach(plain) == 0.0
    assert cprop_mod._needs_numpy(plain) is False

    scored = _base(
        xgr=[[0, 30, 10.0, 1.0]],
        core={
            "E1": 32e6,
            "E2": 32e6,
            "E3": 70e6,
            "G12": 19e6,
            "G13": 19e6,
            "G23": 19e6,
            "nu12": 0.3,
            "nu13": 0.3,
            "nu23": 0.3,
            "rho": 60,
            "cell_size": 0.6,
        },
    )
    assert halo_reach(scored) > 0.0
    assert cprop_mod._needs_numpy(scored) is True
    assert cprop_mod._is_orthotropic(scored) is True
    assert cprop_mod._score_field(plain) is None
    assert cprop_mod._score_field(scored) is not None


def test_cprop_overwrites_existing_run(tmp_path):
    from tests.fakes import fake_backend, unregister

    register, cls = fake_backend("fake")
    register(cls)
    case = _base(backend="fake")
    path = tmp_path / "case.json"
    path.write_text(json.dumps(case))
    try:
        first = cprop_mod.cprop(str(path))
        written = list(tmp_path.glob("run*.json"))
        assert len(written) == 1
        written[0].write_text("{}\n")
        second = cprop_mod.cprop(str(path))
    finally:
        unregister("fake")
    assert second["Exx"] == first["Exx"]
    assert json.loads(written[0].read_text())["Exx"] == first["Exx"]
    assert len(list(tmp_path.glob("run*.json"))) == 1


def test_homogenize_uses_registered_backend():
    from tests.fakes import fake_backend, unregister

    from b3_core.cases import plain

    register, cls = fake_backend("fake")
    register(cls)
    try:
        result = homogenize(plain(), name="wrap", backend="fake")
    finally:
        unregister("fake")
    assert result.material.name == "wrap"
    assert result.material.Ex == 1e9
