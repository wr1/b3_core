import numpy as np
import pytest

import b3_core.core.cprop as cprop_module
from b3_core.io import fenicsx


def test_fenicsx_reports_missing_dependency():
    if fenicsx.is_fenicsx_available():
        pytest.skip("FEniCSx is installed in this environment")

    with pytest.raises(fenicsx.FenicsxUnavailableError):
        fenicsx.runfenicsx(None, {"E": 4e9, "nu": 0.3}, {"E": 4e9, "nu": 0.3})


def test_validate_against_ccx_pass_and_fail():
    ccx = {"Exx": 100.0, "Gxy": 50.0}
    close = {"Exx": 103.0, "Gxy": 49.0}
    far = {"Exx": 120.0, "Gxy": 49.0}

    passing = fenicsx.validate_against_ccx(ccx, close, rtol=0.05)
    failing = fenicsx.validate_against_ccx(ccx, far, rtol=0.05)

    assert passing["passed"] is True
    assert passing["properties"]["Exx"]["ok"] is True
    assert failing["passed"] is False
    assert failing["properties"]["Exx"]["ok"] is False


def test_properties_from_isotropic_stiffness():
    e = 4.0e9
    nu = 0.3
    g = e / (2.0 * (1.0 + nu))
    lam = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    stiffness = np.array(
        [
            [lam + 2.0 * g, lam, lam, 0.0, 0.0, 0.0],
            [lam, lam + 2.0 * g, lam, 0.0, 0.0, 0.0],
            [lam, lam, lam + 2.0 * g, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, g, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, g, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, g],
        ]
    )

    props, _ = fenicsx._properties_from_stiffness(stiffness)

    assert props["Exx"] == pytest.approx(e)
    assert props["Eyy"] == pytest.approx(e)
    assert props["Ezz"] == pytest.approx(e)
    assert props["Gxy"] == pytest.approx(g)
    assert props["Gxz"] == pytest.approx(g)
    assert props["Gyz"] == pytest.approx(g)
    assert props["nuxy"] == pytest.approx(nu)


def test_cprop_fenicsx_validation_dispatch(monkeypatch, tmp_path):
    calls = []

    def fake_fenicsx(mesh, dct, status=None):
        calls.append("fenicsx")
        return {"Exx": 100.0, "Gxy": 50.0}

    def fake_ccx(mesh, name, dct, status=None):
        calls.append("ccx")
        return {"Exx": 101.0, "Gxy": 49.0}

    monkeypatch.setattr(cprop_module, "create_grooved_mesh", lambda *a, **k: object())
    monkeypatch.setattr(
        cprop_module,
        "geom_analysis",
        lambda mesh: {"area_increase": 1.0, "resin_vf": 0.0},
    )
    monkeypatch.setattr(cprop_module, "_run_fenicsx_backend", fake_fenicsx)
    monkeypatch.setattr(cprop_module, "_run_ccx_backend", fake_ccx)

    cfg = tmp_path / "case.json"
    cfg.write_text(
        """{
            "dx": 50.0,
            "dy": 50.0,
            "thickness": 30.0,
            "xgr": [],
            "ygr": [],
            "core": {"E": 4000000000.0, "nu": 0.3, "rho": 100.0},
            "resin": {"E": 4000000000.0, "nu": 0.3, "rho": 1100.0},
            "backend": "fenicsx",
            "validate_with_ccx": true
        }"""
    )

    out = cprop_module.cprop(str(cfg))

    assert calls == ["fenicsx", "ccx"]
    assert out["ccx_validation"]["passed"] is True
