"""Physics surrogate: base trends, fit, and mass curvature lookup."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from b3_core.physics_surrogate import (
    GeometrySpec,
    build_training_frame,
    fit_from_homogenization,
    fit_physics_surrogate,
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
