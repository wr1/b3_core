"""Exact multi-knob calibration on a stand-in solve. No finite-element run."""

from __future__ import annotations

import json

import numpy as np
import pytest

from b3_core.fit import exact as X

CASE = {
    "dx": 30.0,
    "dy": 30.0,
    "thickness": 20.0,
    "xgr": [[15.0, 30.0, -18.0, 2.0]],
    "ygr": [[15.0, 30.0, -18.0, 2.0]],
    "core": {"E": 85e6, "nu": 0.3, "rho": 80.0, "cell_size": 0.5},
    "resin": {"E": 3.2e9, "nu": 0.35, "rho": 1150.0},
}
KNOBS = ["halo_reach", "core_E", "core_G", "resin_E", "resin_rho"]
ALL = list(X.SYM_TARGETS)  # explicit averaged set (CASE is symmetric)
TRUTH = {
    "halo_reach": 0.72,
    "core_E": 92e6,
    "core_G": 29.5e6,
    "resin_E": 3.05e9,
    "resin_rho": 1180.0,
}


def fake(case, kx=0.0, ky=0.0):
    """Smooth, mildly nonlinear stand-in with the FE model's structure."""
    c, r = case["core"], case["resin"]
    e_c = c.get("Ex", c.get("E"))
    g_c = c.get("Gxz", (c.get("E") or 0) / (2 * (1 + c.get("nu", 0.3))))
    h = float(c.get("cell_size") or 0.0)
    er, rr = r["E"], r["rho"]
    vf = 0.058 + 0.03 * h
    e_in = vf * er * (1 + 0.1 * h) + 0.95 * e_c
    return {
        "Exx": e_in,
        "Eyy": e_in,
        "Ezz": 1 / (0.1 / e_c + 0.9 / (2 * vf * er + e_c)),
        "Gxz": vf * er / 2.7 * (1 + 0.2 * h**2) + 0.9 * g_c,
        "Gyz": vf * er / 2.7 * (1 + 0.2 * h**2) + 0.9 * g_c,
        "Gxy": 1.1 * g_c + 0.004 * er * (1 + h),
        "rho_infused": c["rho"] * (1 - 2 * vf) + 2 * vf * rr,
    }


def test_cubic_core_matches_isotropic_g():
    core = X.cubic_core(CASE["core"])
    assert core["Gxz"] == pytest.approx(85e6 / 2.6)
    assert core["Ex"] == core["Ez"] == 85e6 and "E" not in core


def test_apply_and_read_knobs_roundtrip():
    case = X.apply_knobs(CASE, TRUTH)
    assert X.read_knobs(case, KNOBS) == pytest.approx(TRUTH)
    assert CASE["core"]["E"] == 85e6  # input untouched
    with pytest.raises(KeyError):
        X.apply_knobs(CASE, {"nope": 1.0})
    resin = X.apply_knobs(CASE, {"resin_nu": 0.33, "core_rho": 75.0})
    assert resin["resin"]["nu"] == 0.33 and resin["core"]["rho"] == 75.0


def test_targets_average_symmetric_pairs():
    t = X.targets_of(
        {
            "Exx": 1.0,
            "Eyy": 3.0,
            "Ezz": 2,
            "Gxz": 4,
            "Gyz": 6,
            "Gxy": 1,
            "rho_infused": 9,
        }
    )
    assert t == {
        "Exx": 1.0,
        "Eyy": 3.0,
        "Ezz": 2.0,
        "Gxz": 4.0,
        "Gyz": 6.0,
        "Gxy": 1.0,
        "rho": 9.0,
    }  # raw by default: nothing averaged
    raw = {
        "Exx": 1.0,
        "Eyy": 3.0,
        "Ezz": 2,
        "Gxz": 4,
        "Gyz": 6,
        "Gxy": 1,
        "rho_infused": 9,
    }
    t = X.targets_of(raw, list(X.SYM_TARGETS))
    assert t == {"E_in": 2.0, "Ezz": 2.0, "G_tr": 5.0, "Gxy": 1.0, "rho": 9.0}


def test_ccd_shapes_and_half_fraction():
    assert X.ccd(4).shape == (1 + 16 + 8, 4)
    d5 = X.ccd(5)
    assert d5.shape == (1 + 16 + 10, 5)
    corners = d5[1:17]
    assert np.allclose(corners[:, 4], np.prod(corners[:, :4], axis=1))


def test_quadratic_surface_is_exact_on_a_quadratic():
    rng = np.random.default_rng(1)
    u = X.ccd(3)
    coef = rng.normal(size=(10, 2))
    y = X._phi(u) @ coef
    s = X.QuadSurface.fit(u, y)
    p = rng.uniform(-1, 1, 3)
    assert s.predict(p) == pytest.approx((X._phi(p) @ coef)[0])
    fd = (s.predict(p + [1e-6, 0, 0]) - s.predict(p - [1e-6, 0, 0])) / 2e-6
    assert s.jac(p)[:, 0] == pytest.approx(fd, rel=1e-5)
    assert s.fit_rms_pct < 1e-10
    with pytest.raises(ValueError):
        X.QuadSurface.fit(u[:5], y[:5])


def test_identifiability_rank_and_subsets():
    j = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    rep = X.identifiability(j[:, :2])
    assert rep["rank"] == 2 and rep["condition"] == pytest.approx(1.0)
    dup = X.identifiability(np.array([[1.0, 2.0], [1.0, 2.0]]))
    assert dup["rank"] == 1 and dup["condition"] > 1e12
    ranked = X.rank_subsets(j, ["a", "b", "c"], 2)
    assert ranked[0][1] == ("a", "b")


def test_exact_fit_clean_recovers_truth(tmp_path):
    meas = X.mock_test(fake, CASE, TRUTH)
    store = tmp_path / "s.jsonl"
    fit = X.exact_fit(
        fake, CASE, meas, KNOBS, targets=ALL, store=store, true=list(TRUTH.values())
    )
    assert fit.max_abs_residual_pct < 0.01
    assert fit.fitted == pytest.approx(list(TRUTH.values()), rel=1e-3)
    assert fit.n_design == 27 and np.isfinite(fit.condition)
    assert fit.history[0]["stage"] == "surface"
    assert not fit.at_bound
    # Resume: every solve comes back from the store.
    again = X.exact_fit(fake, CASE, meas, KNOBS, targets=ALL, store=store)
    assert again.n_solves == 0 and again.fitted == pytest.approx(fit.fitted)
    back = X.ExactFit.from_dict(json.loads(json.dumps(fit.to_dict())))
    assert back.values() == pytest.approx(fit.values())
    assert back.apply(CASE)["core"]["cell_size"] == pytest.approx(0.72, rel=1e-3)


def test_exact_fit_noisy_is_still_exact_on_targets():
    noise = {p: 0.03 for p in ("Exx", "Eyy", "Ezz", "Gxz", "Gyz", "Gxy")} | {
        "rho_infused": 0.01
    }
    meas = X.mock_test(fake, CASE, TRUTH, noise=noise, seed=3)
    assert meas["Exx"] != meas["Eyy"]
    fit = X.exact_fit(fake, CASE, meas, KNOBS, targets=ALL)
    if not fit.at_bound:
        assert fit.max_abs_residual_pct < 0.01


def test_exact_fit_needs_a_square_system():
    with pytest.raises(ValueError, match="as many knobs"):
        X.exact_fit(fake, CASE, {}, KNOBS[:3])


def test_default_fit_is_four_by_four_with_gxy_held_out():
    truth = {k: TRUTH[k] for k in X.FIT_KNOBS}
    meas = X.mock_test(fake, CASE, truth)
    fit = X.exact_fit(fake, CASE, meas, X.FIT_KNOBS, average_xy=True)
    assert list(fit.measured) == list(X.FIT_TARGETS_SYM)
    chk = fit.extra["xy_check"]
    assert chk["symmetric_input"] and chk["ok"] and chk["averaged"]
    assert "consistent with symmetric input" in chk["message"]
    assert list(fit.check_measured) == ["Gxy"]
    assert fit.check_model["Gxy"] == pytest.approx(meas["Gxy"], rel=1e-3)
    assert fit.max_abs_residual_pct < 0.01 and fit.n_design == 25
    with pytest.raises(KeyError, match="lacks"):
        X.exact_fit(fake, CASE, {"Exx": 1.0}, X.FIT_KNOBS, average_xy=True)
    # Without the opt-in the default targets are the raw six: not square here.
    with pytest.raises(ValueError, match="as many knobs"):
        X.exact_fit(fake, CASE, meas, X.FIT_KNOBS)


# --------------------------------------------------------------------------- #
# x/y symmetry is detected, never imposed
# --------------------------------------------------------------------------- #
ORTHO = {
    "Ex": 100e6,
    "Ey": 70e6,
    "Ez": 120e6,
    "Gxy": 30e6,
    "Gxz": 36e6,
    "Gyz": 26e6,
    "nuxy": 0.3,
    "nuxz": 0.28,
    "nuyz": 0.32,
    "rho": 80.0,
    "cell_size": 0.5,
}
ACASE = CASE | {"core": ORTHO}


def fake_xy(case, kx=0.0, ky=0.0):
    """Stand-in that keeps x and y apart: Exx from Ex, Gyz from Gyz, ..."""
    c, r = X.orthotropic_core(case["core"]), case["resin"]
    h = float(c.get("cell_size") or 0.0)
    er, rr = r["E"], r["rho"]
    vf = 0.058 + 0.03 * h
    return {
        "Exx": vf * er * (1 + 0.1 * h) + 0.95 * c["Ex"],
        "Eyy": vf * er * (1 + 0.1 * h) + 0.95 * c["Ey"],
        "Ezz": 1 / (0.1 / c["Ez"] + 0.9 / (2 * vf * er + c["Ez"])),
        "Gxz": vf * er / 2.7 * (1 + 0.2 * h**2) + 0.9 * c["Gxz"],
        "Gyz": vf * er / 2.7 * (1 + 0.2 * h**2) + 0.9 * c["Gyz"],
        "Gxy": 1.1 * c["Gxy"] + 0.004 * er * (1 + h),
        "nuxz": 0.2 + c["nuxz"] / 10,
        "nuyz": 0.2 + c["nuyz"] / 10,
        "rho_infused": c["rho"] * (1 - 2 * vf) + 2 * vf * rr,
    }


def test_core_knobs_scale_and_keep_anisotropy():
    d = X.apply_knobs(ACASE, {"core_E": 150e6, "core_G": 54e6})["core"]
    assert d["Ex"] == pytest.approx(150e6)
    assert d["Ey"] / d["Ex"] == pytest.approx(0.7)
    assert d["Ez"] / d["Ex"] == pytest.approx(1.2)
    assert d["Gxz"] == pytest.approx(54e6)
    assert d["Gyz"] / d["Gxz"] == pytest.approx(26 / 36)
    assert d["Gxy"] / d["Gxz"] == pytest.approx(30 / 36)
    assert (d["nuxz"], d["nuyz"]) == (0.28, 0.32)
    # Isotropic generic case: identical to the old cubic overwrite.
    iso = X.apply_knobs(CASE, {"core_E": 92e6, "core_G": 29.5e6})["core"]
    for k in ("Ex", "Ey", "Ez"):
        assert iso[k] == pytest.approx(92e6, rel=1e-15)
    for k in ("Gxy", "Gxz", "Gyz"):
        assert iso[k] == pytest.approx(29.5e6, rel=1e-15)


def test_symmetry_detection_covers_geometry_and_materials():
    from b3_core.symmetry import xy_asymmetry

    assert xy_asymmetry(CASE) == []
    assert "core Ex≠Ey" in xy_asymmetry(ACASE)
    assert "core Gxz≠Gyz" in xy_asymmetry(ACASE)
    nu_only = CASE | {"core": ORTHO | {"Ey": 100e6, "Gyz": 36e6}}
    assert xy_asymmetry(nu_only) == ["core nuxz≠nuyz"]
    shallow = CASE | {"ygr": [[15.0, 30.0, -9.0, 2.0]]}
    assert xy_asymmetry(shallow) == ["x and y groove rows differ"]
    assert xy_asymmetry(CASE | {"dy": 25.0}) == ["dx≠dy"]


def test_asymmetric_fit_uses_separate_targets():
    knobs = ["halo_reach", "core_Ex", "core_Ey", "core_Gxz", "core_Gyz", "resin_E"]
    truth = {
        "halo_reach": 0.6,
        "core_Ex": 105e6,
        "core_Ey": 66e6,
        "core_Gxz": 38e6,
        "core_Gyz": 24e6,
        "resin_E": 3.1e9,
    }
    meas = X.mock_test(fake_xy, ACASE, truth)
    assert meas["Exx"] != pytest.approx(meas["Eyy"], rel=1e-3)
    assert meas["Gxz"] != pytest.approx(meas["Gyz"], rel=1e-3)
    # Opt-in averaging is ignored for an asymmetric case: raw targets.
    fit = X.exact_fit(fake_xy, ACASE, meas, knobs, average_xy=True)
    assert list(fit.measured) == list(X.FIT_TARGETS)
    assert {"Exx", "Eyy", "Gxz", "Gyz"} <= set(fit.measured)
    assert fit.max_abs_residual_pct < 0.01
    assert fit.fitted == pytest.approx(list(truth.values()), rel=1e-3)
    chk = fit.extra["xy_check"]
    assert not chk["symmetric_input"] and not chk["averaged"]
    assert "not x/y symmetric" in chk["message"]
    # Averaged targets on an asymmetric case are refused outright.
    with pytest.raises(ValueError, match="not x/y symmetric"):
        X.exact_fit(fake_xy, ACASE, meas, X.FIT_KNOBS, targets=X.FIT_TARGETS_SYM)


def test_symmetry_check_flags_broken_symmetric_solve():
    from b3_core.symmetry import symmetry_check

    good = symmetry_check(CASE, {"Exx": 1.0, "Eyy": 1.0, "Gxz": 2.0, "Gyz": 2.0})
    assert good["ok"] and "consistent with symmetric input" in good["message"]
    bad = symmetry_check(CASE, {"Exx": 1.0, "Eyy": 1.1, "Gxz": 2.0, "Gyz": 2.0})
    assert not bad["ok"] and "check the solve" in bad["message"]
