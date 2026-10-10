"""Library tests for the datasheet path. No finite-element solve."""

import json
from types import SimpleNamespace

import pytest

from b3_core.cases import uniaxial
from b3_core.fit import (
    default_bounds,
    design,
    evaluate,
    fit_quadratic,
    run_fit,
    solve_rsm,
)
from b3_core.fit.spec import FitSpec, Target, TargetSet
from b3_core.report.prose import Prose
from b3_core.report.sheet import DatasheetRejected, assemble_data, generate_full
from b3_core.solvers.stamps import stamp_for
from b3_core.viz import tensor


def _record(case, *, backend, cache, ex=None):
    modulus = float(case.core.E if ex is None else ex)
    return SimpleNamespace(
        result=SimpleNamespace(
            backend=backend,
            properties={
                "Ex": modulus,
                "Ey": modulus,
                "Ez": modulus,
                "Gxy": modulus / 2.6,
                "Gxz": modulus / 2.6,
                "Gyz": modulus / 2.6,
                "nuxy": 0.3,
                "nuxz": 0.3,
                "nuyz": 0.3,
            },
        ),
        geometry={"rho_infused": 120.0, "resin_vf": 0.02},
        meta_cache_hit=False,
        provenance=None,
        diagnostics={"checks": [], "ties": []},
        case_hash="abc",
    )


def test_design_ccd_is_deterministic_and_inside_the_cube():
    params = default_bounds(uniaxial())
    params.params[0].fixed = False
    first = design(params, method="ccd", seed=1)
    second = design(params, method="ccd", seed=1)
    assert first.shape[1] == 1
    assert first.shape[0] >= 3
    assert abs(first - second).max() == 0
    assert abs(first).max() <= 1.0


def test_run_fit_with_nothing_free_is_a_prediction():
    case = uniaxial()
    spec = FitSpec(
        case=case.model_dump(mode="json"),
        params=default_bounds(case),
        targets=TargetSet(targets=[Target(quantity="Ex", value=1.0e6, rel_sd=0.05)]),
        backend="numpy",
        cache=None,
        stages={
            "halo_from_mass": False,
            "design": False,
            "rsm": False,
            "refine": {"max_solves": 4},
            "identify": False,
        },
    )
    result = run_fit(spec, solve=_record)
    assert result.status == "no_free_params"
    assert result.calibrated_case["core"]["source"] == "calibrated"
    assert "fitspec_hash" in result.provenance


def test_rsm_only_uses_the_stand_in_solves():
    case = uniaxial()
    params = default_bounds(case)
    params.params[0].fixed = False
    spec = FitSpec(
        case=case.model_dump(mode="json"),
        params=params,
        targets=TargetSet(
            targets=[Target(quantity="Ex", value=float(case.input.core.E), rel_sd=0.05)]
        ),
        backend="numpy",
        cache=None,
        stages={
            "halo_from_mass": False,
            "design": {"method": "ccd", "n": 4, "seed": 0},
            "rsm": {"log": False, "ridge": 1e-4},
            "refine": False,
            "identify": False,
        },
    )
    result = run_fit(spec, solve=_record)
    assert result.status == "rsm_only"
    assert result.rsm is not None
    assert "Ex" in result.rsm["coefs"]


def test_quadratic_recovers_a_flat_property():
    case = uniaxial()
    params = default_bounds(case)
    params.params[0].fixed = False
    points = design(params, method="ccd")
    table = evaluate(case, params, points, backend="numpy", solve=_record, workers=1)
    targets = TargetSet(
        targets=[Target(quantity="Ex", value=float(case.input.core.E), rel_sd=0.01)]
    )
    surface = fit_quadratic(table, targets, log=False, ridge=1e-8)
    start = solve_rsm(surface, targets, params)
    assert start.shape == (1,)
    predicted = surface.predict([0.0])["Ex"]
    assert abs(predicted - float(case.input.core.E)) / case.input.core.E < 0.05


def test_figures_write_pngs(tmp_path):
    from b3_core.viz.figs import write_figures

    rows = []
    for kx in (-0.001, 0.0, 0.001):
        for ky in (-0.001, 0.0, 0.001):
            rows.append(
                {
                    "kx": kx,
                    "ky": ky,
                    "ok": True,
                    "Ex": 1e6 * (1 + 10 * kx),
                    "Ey": 1e6,
                    "Ez": 2e6 * (1 - 5 * ky),
                    "Gxy": 4e5,
                    "Gxz": 4e5 * (1 + 3 * kx),
                    "Gyz": 4e5,
                    "rho_infused": 100 + 1000 * kx,
                    "resin_vf": 0.02 + kx,
                }
            )
    written = write_figures({"rows": rows}, tmp_path)
    assert (tmp_path / "props_curvature.png").is_file()
    assert (tmp_path / "sweep_heatmap.png").is_file()
    assert (tmp_path / "rho_curvature.png").is_file()
    assert "kerf_ranges_wedge" not in written


def test_kerf_map_solves_only_the_corners():
    from b3_core.sweep.kerf_map import kerf_map

    calls = []

    def solve(case, *, backend, cache):
        calls.append((case.curvature.kx, case.curvature.ky))
        return _record(case, backend=backend, cache=cache)

    report = kerf_map(uniaxial(), 0.004, solve=solve, backend="numpy")
    assert len(report["grid"]) == 9
    assert len(calls) == 4
    assert report["convention"]
    assert report["convention"][0]["opens_for"] in {"k>0", "k<0"}
    solved = [cell for cell in report["grid"] if cell["solved"]]
    assert len(solved) == 4
    assert all("Ex" in cell for cell in solved)


def test_bisect_reports_a_bounded_morph():
    from b3_core.checks.bisect import bisect

    def solve(mesh, field, case):
        return {
            "Ex": float(mesh.n_points),
            "Ey": 1.0,
            "Ez": 1.0,
            "Gxy": 1.0,
            "Gxz": 1.0,
            "Gyz": 1.0,
        }

    report = bisect(uniaxial(), kx=0.008, solve=solve)
    assert report["max_node_displacement_mm"] > 0.0
    assert report["max_node_displacement_mm"] < 5.0
    assert set(report["a_flat_mesh_flat_field"]) >= {"Ex"}
    assert "c2_curved_zmesh_morph_off" in report["delta_pct"]


def test_reproduce_distinguishes_drift_from_a_match():
    from b3_core.checks.reproduce import check_reproduce

    case = uniaxial().model_dump(mode="json")
    stamp = stamp_for("numpy")

    def solve(point, *, backend, cache):
        return _record(point, backend=backend, cache=cache, ex=1.0e6)

    values = {
        "points": [
            {
                "case": case,
                "kx": 0.0,
                "ky": 0.0,
                "backend": "numpy",
                "solver_stamp": stamp,
                "properties": {
                    "Ex": 1.0e6,
                    "Ey": 1.0e6,
                    "Ez": 1.0e6,
                    "Gxy": 1e6 / 2.6,
                    "Gxz": 1e6 / 2.6,
                    "Gyz": 1e6 / 2.6,
                },
            }
        ]
    }
    assert check_reproduce(values, solve=solve)["ok"] is True
    values["points"][0]["solver_stamp"] = "numpy/not-this-stamp"
    drifted = check_reproduce(values, solve=solve)
    assert drifted["ok"] is False
    assert drifted["rows"][0]["drift"] is True


def test_convergence_sees_two_madd_levels():
    from b3_core.checks.convergence import check_convergence

    case = uniaxial()

    def solve(point, *, backend, cache):
        return _record(point, backend=backend, cache=cache, ex=1e6 * len(point.madd))

    report = check_convergence(case, solve=solve, rtol_pct=0.0)
    assert report["rows"][0]["n_madd"] < report["rows"][1]["n_madd"]
    assert report["rel_pct"] > 0


def test_accept_fails_closed_and_passes_a_complete_project(tmp_path):
    from b3_core.checks.accept import check_accept

    empty = check_accept({})
    assert empty["ok"] is False
    cache = tmp_path / "cache"
    cache.mkdir()
    project = {
        "cache": str(cache),
        "grid": {
            "rows": [
                {
                    "ok": True,
                    "kx": 0.0,
                    "ky": 0.0,
                    "diagnostics": {
                        "checks": [{"id": "pd", "status": "ok"}],
                        "ties": [{"face": "z", "ok": True}],
                    },
                }
            ],
            "manifest": {"kx_inert": False, "ky_inert": False},
        },
        "provenance": {"commit": "abc", "dirty": False},
        "backends_report": {
            "rows": [
                {
                    "kx": 0.0,
                    "ky": 0.0,
                    "flags": [],
                    "pairs": {"a_vs_b": {"max_rel_pct": 0.001}},
                },
                {
                    "kx": 0.001,
                    "ky": 0.0,
                    "flags": ["curved_growth"],
                    "pairs": {"a_vs_b": {"max_rel_pct": 0.05}},
                },
            ]
        },
        "reproduce": {"ok": True, "rows": []},
        "fit_result": {"status": "no_free_params", "residuals": {"max_abs_z": 0.0}},
        "convention": [{"axis": "x", "mouth": "bottom", "opens_for": "k<0"}],
    }
    assert check_accept(project)["ok"] is True


def test_cache_export_round_trip(tmp_path):
    from b3_core.cache import cache_export, cache_import

    root = tmp_path / "cache" / "ab"
    root.mkdir(parents=True)
    payload = {
        "result": {"backend": "numpy"},
        "solver_stamp": stamp_for("numpy"),
        "input": {"curvature": {"kx": 0.0, "ky": 0.0}},
        "case_hash": "abc",
    }
    (root / "abcd.json").write_text(json.dumps(payload), encoding="utf-8")
    bundle = tmp_path / "cache.zip"
    exported = cache_export(tmp_path / "cache", bundle)
    assert exported["n"] == 1
    copied = cache_import(bundle, tmp_path / "other")
    assert copied["copied"]
    again = cache_import(bundle, tmp_path / "other")
    assert again["skipped"]


def test_prose_rejects_a_long_summary():
    with pytest.raises(ValueError):
        Prose(
            title="t",
            summary="x" * 401,
            intended_use="use",
            limitations="limit",
        )


def test_sheet_records_the_cache_directory():
    from b3_core.cli_ext import _sheet_provenance

    recorded = _sheet_provenance({"commit": "abc"}, "/tmp/cache")
    assert recorded["cache"] == "/tmp/cache"
    assert recorded["commit"] == "abc"
    assert "cache" not in _sheet_provenance(None, "")


def test_full_sheet_refuses_without_draft_and_compiles_with_it(tmp_path):
    prose = Prose(
        title="Draft core",
        summary="Predicted, not calibrated.",
        intended_use="Internal review of the flat card.",
        limitations="Curved values are FEniCSx until accept passes.",
    )
    data = assemble_data(
        prose=prose,
        properties={"Ex": 1.0e6, "rho_infused": 120.0},
        convention=[{"axis": "x", "mouth": "bottom", "opens_for": "k<0"}],
        kx=0.001,
        ky=0.0,
        draft=True,
    )
    assert "FEniCSx" in data["footnote"]
    with pytest.raises(DatasheetRejected):
        generate_full(tmp_path / "refused", data, draft=False, compile=False)
    pdf = generate_full(tmp_path / "draft", data, draft=True, compile=True)
    assert pdf.is_file()
    assert pdf.stat().st_size > 500
    assert (tmp_path / "draft" / "data.json").is_file()


def test_engineering_constants_use_the_fea_names():
    import numpy as np

    C = np.diag([2e9, 1e9, 3e9, 4e8, 5e8, 6e8])
    names = tensor.engineering_constants(C)
    assert names["Ex"] == names["E_x"]
    assert names["Gxy"] == names["G_xy"]
    assert names["Ex"] > 0


def _worker_solve(case, *, backend, cache):
    return _record(case, backend=backend, cache=cache)


def test_sweep_workers_pool_returns_every_row():
    from b3_core.sweep.curvature import sweep_curvature

    result = sweep_curvature(
        uniaxial(),
        "0:0.001:2",
        "0",
        backend="numpy",
        workers=2,
        solve=_worker_solve,
    )
    assert result["manifest"]["n_failed"] == 0
    assert len(result["rows"]) == 2
