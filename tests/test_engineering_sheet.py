"""Engineering datasheet: formatting, closure, calibration, near-zero, full PDF."""

from __future__ import annotations

import json
import re
import shutil
import warnings

import pytest

from b3_core.report import engineering as eng

CASE = {
    "dx": 30.0,
    "dy": 30.0,
    "thickness": 20.0,
    "xgr": [[15.0, 30.0, -18.0, 2.0]],
    "ygr": [[15.0, 30.0, -18.0, 2.0]],
    "core": {"E": 85e6, "nu": 0.3, "rho": 80.0, "cell_size": 0.5},
    "resin": {"E": 3.2e9, "nu": 0.35, "rho": 1150.0},
    "mesh_h_max": 3.0,
}


def fake_solver(case, kx=0.0, ky=0.0):
    """Smooth closed form: grooves stiffen, cell_size adds halo, k opens/closes."""
    grooved = 1.0 if case.get("xgr") else 0.0
    halo = float(case["core"].get("cell_size") or 0.0)
    base = 85e6 * (1 + grooved * (3.5 + 0.8 * halo))
    out = {
        "Exx": base * (1 + 40 * ky),
        "Eyy": base * (1 + 40 * kx),
        "Ezz": base * (1 - 30 * abs(kx) - 30 * abs(ky)),
        "Gxz": 0.25 * base * (1 + 10 * ky),
        "Gyz": 0.25 * base * (1 + 10 * kx),
        "Gxy": 0.12 * base,
        "nuxy": 0.12,
        "nuxz": 0.24,
        "rho_infused": 80 + grooved * (160 + 30 * halo),
        "resin_vf": 0.116 * grooved,
        "halo_vf": 0.06 * halo * grooved,
        "_backend": "fake",
    }
    return out


def test_fmt_modulus_never_one_sig_fig_gpa():
    assert eng.fmt_modulus(4.0e8) == "400 MPa"
    assert eng.fmt_modulus(3.2e9) == "3.2 GPa"
    assert eng.fmt_modulus(8.5e7) == "85 MPa"


def test_closure_curvature_and_clamp():
    close = eng.closure_curvature(CASE)
    assert close["kx"] == pytest.approx(-2.0 / (30 * 18))
    assert eng.is_clamped(CASE, -1 / 200, 0.0)
    assert not eng.is_clamped(CASE, +1 / 200, 0.0)
    assert not eng.is_clamped(CASE, -1 / 2000, 0.0)


def test_collect_warns_on_clamp_and_marks_points():
    with pytest.warns(RuntimeWarning, match="past groove closure"):
        data = eng.collect(CASE, radii_m=(0.2, 1.0), solve=fake_solver)
    assert data.clamped_points > 0
    assert any(r["clamped"] for r in data.sweep)
    assert data.plain["Exx"] < data.flat["Exx"]


def _cal_spec(halo=0.7):
    """1 × 1 exact fit on the stand-in: density pins the halo reach."""
    truth = fake_solver(
        json.loads(json.dumps(CASE)) | {"core": CASE["core"] | {"cell_size": halo}}
    )
    return {
        "measured": {p: truth[p] for p in eng.CAL_PROPS},
        "knobs": ["halo_reach"],
        "targets": ["rho"],
        "true": [halo],
        "label": "mock test",
        "note": "synthetic",
    }


def test_calibration_recovers_parameter():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        data = eng.collect(
            CASE,
            radii_m=(1.0,),
            biaxial=False,
            calibration=_cal_spec(),
            solve=fake_solver,
        )
    fit = data.calibration
    assert fit.fitted[0] == pytest.approx(0.7, abs=1e-4)
    assert data.case["core"]["cell_size"] == pytest.approx(0.7, abs=1e-4)
    assert fit.max_abs_residual_pct < 0.01


def test_calibration_accepts_saved_fit_and_rejects_old_spec(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        data = eng.collect(
            CASE, radii_m=(1.0,), calibration=_cal_spec(), solve=fake_solver
        )
    saved = tmp_path / "fit.json"
    saved.write_text(json.dumps(data.calibration.to_dict()))
    calls = []

    def counting(case, kx=0.0, ky=0.0):
        calls.append((kx, ky))
        return fake_solver(case, kx, ky)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        again = eng.collect(
            CASE, radii_m=(1.0,), biaxial=False, calibration=saved, solve=counting
        )
    assert again.case["core"]["cell_size"] == pytest.approx(0.7, abs=1e-4)
    assert len(calls) == 3 + 2  # flat, k→0, plain + 2 kx points (ky mirrored): no fit
    with pytest.raises(ValueError, match="exact fit"):
        eng.load_calibration(
            {"measured": {}, "param": "core.cell_size"}, CASE, fake_solver
        )


def test_typst_has_required_content():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        data = eng.collect(CASE, radii_m=(0.2, 1.0), solve=fake_solver)
    src = eng.build_typst(
        data,
        title="T",
        subtitle="S",
        figs=dict.fromkeys(["iso", "cuts", "grooving", "heatmaps", "line"], "x.png"),
    )
    assert "Plain foam" in src and "Ratio / ρ" in src
    assert "Provenance" in src and "mesh\\_h\\_max" in src
    assert "drives $E_(y y)$ and $G_(y z)$" in src
    assert "past groove closure" in src
    assert " GPa" not in src.split("Key numbers")[1].split("Inputs")[0]


@pytest.mark.skipif(shutil.which("typst") is None, reason="typst not installed")
def test_symmetric_sweep_mirrors_half_the_solves():
    calls = []

    def counting(case, kx=0.0, ky=0.0):
        calls.append((kx, ky))
        return fake_solver(case, kx, ky)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        data = eng.collect(CASE, radii_m=(1.0, 2.0), solve=counting)
        full = eng.collect(
            CASE | {"ygr": [[15.0, 30.0, -18.0, 2.0001]]},
            radii_m=(1.0, 2.0),
            solve=fake_solver,
        )
    assert eng.xy_symmetric(CASE) and not eng.xy_symmetric(full.case)
    n = len(data.ks)
    assert (
        len(calls) == 3 + (n * n - 1 + n - 1) // 2
    )  # flat, k→0, plain + upper triangle
    by = {(r["kx"], r["ky"]): r for r in data.sweep}
    for (kx, ky), r in by.items():
        ref = fake_solver(CASE, kx, ky)
        for p in eng.PROPS:
            assert r[p] == pytest.approx(ref[p]), (kx, ky, p)


@pytest.mark.skipif(shutil.which("typst") is None, reason="typst not installed")
def test_full_pdf_smoke(tmp_path):
    cal = _cal_spec(0.65)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pdf = eng.generate_engineering(
            CASE,
            tmp_path,
            title="Smoke",
            radii_m=(0.5, 2.0),
            calibration=cal,
            solve=fake_solver,
        )
    assert pdf.is_file() and pdf.stat().st_size > 20_000
    numbers = json.loads((tmp_path / "numbers.json").read_text())
    assert numbers["calibration"]["fitted"][0] == pytest.approx(0.65, abs=1e-3)
    src = (tmp_path / "sheet.typ").read_text()
    assert "exact fit" in src and "cond(J)" in src and "Truth" in src
    for name in (
        "fig_rve_iso_halo.png",
        "fig_rve_cuts.png",
        "fig_curvature_map.png",
        "fig_kx_cut.png",
        "fig_grooving.png",
        "fig_calibration.png",
    ):
        assert (tmp_path / name).is_file()
    if shutil.which("pdfinfo"):
        import subprocess

        info = subprocess.run(
            ["pdfinfo", str(pdf)], capture_output=True, text=True
        ).stdout
        assert re.search(r"Pages:\s+3\b", info)


@pytest.mark.skipif(
    pytest.importorskip("b3_core.pipeline").fenicsx_installed() is False,
    reason="FEniCSx not installed",
)
def test_fenicsx_near_zero_curvature_matches_flat():
    """A k → 0 curved FEniCSx solve is the flat card (no step at k = 0)."""
    solve = eng.fenicsx_solver()
    case = json.loads(json.dumps(CASE)) | {"mesh_h_max": 3.0}
    flat = solve(case, 0.0, 0.0)
    near = solve(case, 1e-9, 1e-9)
    for p in eng.PROPS:
        assert near[p] == pytest.approx(flat[p], rel=5e-3), p
