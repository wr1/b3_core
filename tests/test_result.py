"""CoreResult construction and error paths (no FEA)."""

import pytest

from b3_core.result import CoreResult


def _eng(**overrides):
    base = {
        "Exx": 1e9,
        "Eyy": 2e9,
        "Ezz": 3e9,
        "Gxy": 0.5e9,
        "Gxz": 0.4e9,
        "Gyz": 0.3e9,
        "nuxy": 0.3,
        "nuxz": 0.25,
        "nuyz": 0.2,
    }
    base.update(overrides)
    return base


def test_from_engineering_constants():
    eng = _eng()
    r = CoreResult.from_engineering_constants(
        eng,
        rho=120.0,
        resin_volume_fraction=0.1,
        surface_area_factor=1.2,
        name="demo",
    )
    assert r.material.name == "demo"
    assert r.material.Ex == eng["Exx"]
    assert r.material.Ey == eng["Eyy"]
    assert r.material.Ez == eng["Ezz"]
    assert r.material.Gxy == eng["Gxy"]
    assert r.material.rho == 120.0
    assert r.resin_volume_fraction == 0.1
    assert r.surface_area_factor == 1.2
    assert r.engineering_constants == eng


def test_from_cprop_output():
    out = {
        **_eng(),
        "rho_infused": 150.0,
        "resin_vf": 0.05,
        "area_increase": 1.1,
    }
    r = CoreResult.from_cprop_output(out, name="from_cprop")
    assert r.material.Ex == out["Exx"]
    assert r.material.rho == 150.0
    assert r.resin_volume_fraction == 0.05
    assert r.surface_area_factor == 1.1


def test_from_cprop_output_missing_keys():
    with pytest.raises(KeyError, match="missing engineering constants"):
        CoreResult.from_cprop_output(
            {"Exx": 1.0, "rho_infused": 1.0, "resin_vf": 0.0, "area_increase": 1.0}
        )
