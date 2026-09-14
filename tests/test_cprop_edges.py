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
        CpropInput(**_base(backend="abaqus"))
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


def test_cprop_file_exists_error(tmp_path, monkeypatch):
    case = _base(backend="numpy")
    path = tmp_path / "case.json"
    path.write_text(json.dumps(case))

    # Pre-create the hashed output path that cprop will refuse to overwrite.
    validated = CpropInput(**case).model_dump()
    import hashlib

    h = hashlib.md5(str(validated).encode()).hexdigest()
    out = tmp_path / f"run{h}.json"
    out.write_text("{}")

    with pytest.raises(FileExistsError):
        cprop_mod.cprop(str(path))


def test_homogenize_wraps_cprop(monkeypatch):
    eng = {
        "Exx": 1e9,
        "Eyy": 1e9,
        "Ezz": 1e9,
        "Gxy": 0.4e9,
        "Gxz": 0.4e9,
        "Gyz": 0.4e9,
        "nuxy": 0.3,
        "nuxz": 0.3,
        "nuyz": 0.3,
        "rho_infused": 200.0,
        "resin_vf": 0.1,
        "area_increase": 1.05,
    }
    monkeypatch.setattr(cprop_mod, "cprop", lambda *_a, **_k: eng)
    r = homogenize("ignored.json", name="wrap")
    assert r.material.name == "wrap"
    assert r.material.Ex == 1e9
