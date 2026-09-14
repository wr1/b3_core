"""Textile-as-code factories and normalize_case."""

from __future__ import annotations

import pytest

from b3_core import (
    CpropInput,
    Textile,
    crossed,
    curved_panel,
    grid_scored,
    normalize_case,
    plain,
    two_sided,
    uniaxial,
)
from b3_core.cases import from_dict, from_path
from b3_core.core import cprop as cprop_mod


def test_plain_and_normalize_cprop_input():
    t = plain()
    assert isinstance(t, Textile)
    assert t.input.xgr == []
    inp, wd = normalize_case(t)
    assert isinstance(inp, CpropInput)
    assert wd == "."
    inp2, _ = normalize_case(t.input)
    assert inp2.dx == t.input.dx


def test_normalize_dict_strips_comment():
    d = plain().model_dump()
    d["_comment"] = "ignore me"
    inp, _ = normalize_case(d)
    assert inp.thickness == 30.0


def test_normalize_path_json(tmp_path):
    p = plain().to_json(tmp_path / "plain.json")
    inp, wd = normalize_case(str(p))
    assert inp.xgr == []
    assert wd == str(tmp_path)


def test_pattern_factories_match_example_geometry():
    u = uniaxial()
    assert u.input.xgr[0][1] == 10.0  # pitch
    assert u.input.ygr == []
    c = crossed()
    assert c.input.xgr and c.input.ygr
    ts = two_sided()
    assert len(ts.input.ygr) == 2


def test_curved_panel_ligament_and_curvature():
    t = curved_panel(thickness=30, ligament=3, kx=0.012)
    assert t.input.xgr[0][2] == pytest.approx(-27.0)
    assert t.input.curvature["kx"] == pytest.approx(0.012)
    t2 = t.with_thickness(25, ligament=3)
    assert t2.input.thickness == 25
    assert t2.input.xgr[0][2] == pytest.approx(-22.0)


def test_grid_scored_halo_routes_numpy():
    g = grid_scored(cell_size=0.6)
    assert g.input.backend == "numpy"
    assert g.input.core.cell_size == 0.6
    assert g.input.scoring
    sharp = grid_scored(with_halo=False, cell_size=None)
    assert sharp.input.core.cell_size is None
    assert not sharp.input.scoring


def test_fluent_modifiers():
    t = uniaxial().with_backend("numpy").with_curvature(kx=-0.004, ky=0.0)
    assert t.input.backend == "numpy"
    assert t.input.curvature["kx"] == pytest.approx(-0.004)
    h = uniaxial().with_halo(0.5, face_enabled=False)
    assert h.input.core.cell_size == 0.5
    assert h.input.scoring["surfaces"]["face"]["enabled"] is False


def test_from_dict_and_path_roundtrip(tmp_path):
    t0 = crossed()
    path = t0.to_json(tmp_path / "c.json")
    t1 = from_path(path)
    assert t1.input.model_dump() == t0.input.model_dump()
    t2 = from_dict(t0.model_dump())
    assert t2.input.dx == t0.input.dx


def test_homogenize_accepts_textile_via_mock(monkeypatch):
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

    def fake_cprop(case):
        # ensure normalize works inside cprop path
        inp, _ = normalize_case(case)
        assert isinstance(inp, CpropInput)
        return eng

    monkeypatch.setattr(cprop_mod, "cprop", fake_cprop)
    # call homogenize from module under test path
    from b3_core.core.cprop import homogenize as homog

    r = homog(plain(), name="t")
    assert r.material.name == "t"


def test_cprop_rejects_unknown_type():
    with pytest.raises(TypeError, match="Textile"):
        normalize_case(123)
