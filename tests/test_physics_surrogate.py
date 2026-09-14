"""Physics surrogate: base trends, fit, and mass curvature lookup."""

from __future__ import annotations

import numpy as np
import pandas as pd

from b3_core.physics_surrogate import (
    GeometrySpec,
    fit_from_homogenization,
    physics_base,
)


def test_physics_base_open_raises_resin_and_eyy():
    """Top-mouth: kx>0 opens → higher resin_vf and Eyy base."""
    g = GeometrySpec()
    feat = {
        "kx": np.array([-0.008, 0.0, 0.008]),
        "cell_size": np.array([0.0, 0.0, 0.0]),
    }
    b = physics_base(feat, g)
    assert b["resin_vf"][2] > b["resin_vf"][1] > b["resin_vf"][0]
    assert b["Eyy"][2] > b["Eyy"][1] > b["Eyy"][0]
    assert b["mass_per_m2"][2] > b["mass_per_m2"][0]


def test_physics_base_halo_raises_eff_vf():
    feat = {
        "kx": np.array([0.0, 0.0]),
        "cell_size": np.array([0.0, 1.0]),
    }
    b = physics_base(feat, GeometrySpec())
    assert b["halo_vf"][1] > b["halo_vf"][0]
    assert b["effective_resin_vf"][1] > b["effective_resin_vf"][0]
    assert b["Eyy"][1] > b["Eyy"][0]


def test_fit_and_lookup_vector_of_curvatures():
    """Mass lookup: vector of kx → property table; halo beats sharp at fixed open."""
    surr = fit_from_homogenization(
        kx_values=[-0.008, -0.004, 0.0, 0.004, 0.008],
        cell_sizes=[0.0, 0.6],
    )
    kx = np.linspace(-0.008, 0.008, 17)
    out = surr.lookup(kx, cell_size=0.6)
    assert len(out) == 17
    assert "Eyy" in out.columns and "mass_per_m2" in out.columns
    assert np.all(np.isfinite(out["Eyy"]))
    assert np.all(out["Eyy"] > 0)
    # open end stiffer / heavier than closed end
    assert out["Eyy"].iloc[-1] > out["Eyy"].iloc[0]
    assert out["mass_per_m2"].iloc[-1] > out["mass_per_m2"].iloc[0]

    sharp = surr.lookup([0.008], cell_size=0.0)
    halo = surr.lookup([0.008], cell_size=0.6)
    assert float(halo["Eyy"].iloc[0]) > float(sharp["Eyy"].iloc[0])


def test_json_roundtrip(tmp_path):
    surr = fit_from_homogenization(
        kx_values=[-0.008, 0.0, 0.008],
        cell_sizes=[0.0, 0.6],
    )
    path = tmp_path / "surr.json"
    surr.to_json(path)
    s2 = type(surr).from_json(path)
    a = surr.lookup([0.0, 0.005], cell_size=0.6)
    b = s2.lookup([0.0, 0.005], cell_size=0.6)
    assert np.allclose(a["Eyy"], b["Eyy"], rtol=1e-9)


def test_predict_accepts_ndarray_and_dataframe():
    surr = fit_from_homogenization(
        kx_values=[-0.008, 0.0, 0.008],
        cell_sizes=[0.0, 0.6],
    )
    X = np.array([[0.0, 0.6], [0.008, 0.6]])
    a = surr.predict(X, targets=["Eyy", "rho_infused"])
    df = pd.DataFrame({"kx": [0.0, 0.008], "cell_size": [0.6, 0.6]})
    b = surr.predict(df, targets=["Eyy", "rho_infused"])
    assert np.allclose(a["Eyy"], b["Eyy"])


def test_geometry_from_case_and_feat_helpers():
    from b3_core.physics_surrogate import GeometrySpec, _as_feat, _col, physics_base

    # DataFrame / dict / default paths in _col
    df = pd.DataFrame({"kx": [0.0, 0.01]})
    assert _col(df, "kx").tolist() == [0.0, 0.01]
    assert _col(df, "cell_size", 0.5).tolist() == [0.5, 0.5]
    assert _col({"kx": 0.002}, "kx").tolist() == [0.002]
    try:
        _col([1, 2], "kx")
        raise AssertionError("expected TypeError")
    except TypeError:
        pass

    feat = _as_feat(df, ["kx", "cell_size"])
    assert "kx" in feat and "cell_size" in feat
    feat2 = _as_feat({"kx": [0.0], "cell_size": [0.6]}, ["kx", "cell_size"])
    assert feat2["cell_size"][0] == 0.6
    feat3 = _as_feat(np.array([[0.0, 0.6]]), ["kx", "cell_size"])
    assert feat3["kx"][0] == 0.0
    try:
        _as_feat(np.array([[1.0]]), ["kx", "cell_size"])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

    # orthotropic + isotropic material parsing
    g_ortho = GeometrySpec.from_case(
        {
            "dx": 40,
            "dy": 15,
            "thickness": 25,
            "xgr": [[0, 8, -18, 1.5]],
            "core": {
                "E1": 30e6,
                "E2": 30e6,
                "E3": 80e6,
                "G12": 15e6,
                "rho": 55,
            },
            "resin": {"E1": 3.2e9, "G12": 1.2e9, "rho": 1200},
        }
    )
    assert g_ortho.E_core_z == 80e6
    g_iso = GeometrySpec.from_case(
        {
            "core": {"E": 40e6, "nu": 0.3, "rho": 70},
            "resin": {"E": 2.5e9, "nu": 0.35, "rho": 1100},
            "xgr": [[1, 10, 15, 2]],
        }
    )
    assert g_iso.E_core == 40e6

    # broadcast scalar features
    b = physics_base({"kx": 0.0, "cell_size": [0.0, 0.5]}, g_iso)
    assert len(b["Eyy"]) == 2
