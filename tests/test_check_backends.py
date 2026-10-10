"""check backends: parsing, skip reasons, and the flat-anchor gate."""

import json
from types import SimpleNamespace

import pytest

from b3_core.checks.backends import (
    check_backends,
    check_cross,
    parse_axis,
    parse_points,
    refine_madd,
)
from b3_core.core import run as run_mod
from b3_core.solvers.elasticity import isotropic_C


def _record(scale: float):
    stiffness = isotropic_C(4e9, 0.3) * scale
    props = {
        "Ex": 4e9 * scale,
        "Ey": 4e9 * scale,
        "Ez": 4e9 * scale,
        "Gxy": 1e9 * scale,
        "Gxz": 1e9 * scale,
        "Gyz": 1e9 * scale,
        "nuxy": 0.3,
        "nuxz": 0.3,
        "nuyz": 0.3,
    }
    return SimpleNamespace(
        result=SimpleNamespace(stiffness=stiffness, properties=props),
        solver_stamp="test/stamp",
        diagnostics={
            "checks": [{"id": "positive_definite", "status": "ok", "value": 1.0}],
            "ties": [],
        },
        provenance=None,
    )


def _solve_factory(scales):
    def solve(case, backend, kx, ky, pair, cache):
        del case, kx, ky, pair, cache
        return _record(scales[backend])

    return solve


CASE = {
    "dx": 20.0,
    "dy": 20.0,
    "thickness": 10.0,
    "xgr": [[0.0, 20.0, 6.0, 1.0]],
    "ygr": [[0.0, 20.0, 6.0, 1.0]],
    "core": {"E": 80e6, "nu": 0.3, "rho": 100.0},
    "resin": {"E": 3e9, "nu": 0.3, "rho": 1200.0},
    "madd": [-0.15, 0.0, 0.15],
}


def test_parse_points_and_axis_ranges():
    assert parse_points("0,0 5e-5,0 1e-3,1e-3") == [
        (0.0, 0.0),
        (5e-5, 0.0),
        (1e-3, 1e-3),
    ]
    assert parse_axis("1e-3") == [1e-3]
    assert parse_axis("0:2:3") == [0.0, 1.0, 2.0]
    with pytest.raises(ValueError, match="kx,ky"):
        parse_points("1e-3")


def test_refine_madd_inserts_midpoints():
    assert refine_madd([-0.15, 0.0, 0.15], 0) == [-0.15, 0.0, 0.15]
    assert refine_madd([-0.15, 0.0, 0.15], 1) == [
        -0.15,
        -0.075,
        0.0,
        0.075,
        0.15,
    ]
    assert refine_madd([0.0], 2) == [0.0]


def test_flat_anchor_fails_above_a_hundredth_of_a_percent():
    report = check_backends(
        CASE,
        backends=["numpy", "mfem"],
        points=[(0.0, 0.0)],
        solve=_solve_factory({"numpy": 1.0, "mfem": 1.0 + 2e-4}),
        check_installed=False,
    )
    assert report["ok"] is False
    assert report["rows"][0]["flags"] == ["flat_anchor"]
    # max(|Ca|, |Cb|) in the denominator, so 2e-4 / 1.0002 is 0.019996 %.
    assert report["worst_rel_pct"] == pytest.approx(0.02, abs=1e-4)


def test_curved_growth_is_flagged_and_does_not_fail():
    report = check_backends(
        CASE,
        backends=["fenicsx", "numpy"],
        points=[(0.0, 0.0), (1e-3, 1e-3)],
        solve=_solve_factory({"fenicsx": 1.0, "numpy": 1.02}),
        check_installed=False,
    )
    assert report["ok"] is False  # the flat pair is also 2 %
    curved = report["rows"][1]
    assert curved["flags"] == ["curved_growth"]
    assert curved["backends"]["fenicsx"]["properties"]["Ex"] == pytest.approx(4e9)
    assert curved["backends"]["fenicsx"]["diagnostics"]["checks"][0]["id"] == (
        "positive_definite"
    )


def test_agreeing_flat_and_growing_curve_passes():
    def solve(case, backend, kx, ky, pair, cache):
        del case, cache
        assert pair is (backend == "mfem" and (kx != 0.0 or ky != 0.0))
        scale = 1.0 + (0.01 if kx or ky else 0.0)
        if backend == "numpy" and (kx or ky):
            scale += 0.01
        return _record(scale)

    report = check_backends(
        CASE,
        backends=["fenicsx", "mfem", "numpy", "ccx"],
        points=[(0.0, 0.0), (1e-3, 0.0)],
        solve=solve,
        check_installed=False,
    )
    assert report["ok"] is True
    assert report["rows"][0]["flags"] == []
    assert "curved_growth" in report["rows"][1]["flags"]
    assert report["rows"][1]["backends"]["ccx"]["skipped"]
    assert report["rows"][1]["backends"]["mfem"]["allow_pair_periodicity"] is True


def test_cross_forces_the_flat_point():
    seen = []

    def solve(case, backend, kx, ky, pair, cache):
        del case, pair, cache
        seen.append((backend, kx, ky))
        return _record(1.0)

    report = check_cross(
        CASE,
        backends=["numpy", "mfem"],
        points=[(1e-3, 1e-3)],
        solve=solve,
        check_installed=False,
    )
    assert report["ok"] is True
    assert seen == [("numpy", 0.0, 0.0), ("mfem", 0.0, 0.0)]


def test_refine_is_a_separate_row():
    report = check_backends(
        CASE,
        backends=["numpy", "mfem"],
        points=[(0.0, 0.0)],
        refine=1,
        solve=_solve_factory({"numpy": 1.0, "mfem": 1.0}),
        check_installed=False,
    )
    assert report["ok"] is True
    assert [row["madd"] for row in report["rows"]] == [
        [-0.15, 0.0, 0.15],
        [-0.15, -0.075, 0.0, 0.075, 0.15],
    ]


def test_single_backend_flat_row_is_not_an_anchor():
    report = check_backends(
        CASE,
        backends=["numpy"],
        points=[(0.0, 0.0)],
        solve=_solve_factory({"numpy": 1.0}),
        check_installed=False,
    )
    # One solved backend has no pair, so there is nothing to disagree about.
    assert report["ok"] is True
    assert report["rows"][0]["flags"] == []
    assert report["rows"][0]["pairs"] == {}


def test_cli_parses_points_and_exits_on_a_failed_anchor(tmp_path, capsys, monkeypatch):
    case = tmp_path / "case.json"
    case.write_text(json.dumps(CASE), encoding="utf-8")
    seen = {}

    def fake_check(path, **kwargs):
        seen["points"] = kwargs["points"]
        seen["path"] = path
        return {
            "ok": False,
            "rtol": kwargs["rtol"],
            "worst_rel_pct": 1.0,
            "wall_s": 0.0,
            "rows": [],
        }

    def fake_cross(path, **kwargs):
        assert kwargs["points"] == [(0.0, 0.0)]
        return {
            "ok": True,
            "rtol": 1e-4,
            "worst_rel_pct": 0.0,
            "wall_s": 0.0,
            "rows": [],
        }

    monkeypatch.setattr("b3_core.checks.backends.check_backends", fake_check)
    monkeypatch.setattr("b3_core.checks.backends.check_cross", fake_cross)
    with pytest.raises(SystemExit) as caught:
        run_mod.cmd_check_backends(str(case), points="0,0 5e-5,0", as_json=True)
    assert caught.value.code == 1
    assert seen["points"] == [(0.0, 0.0), (5e-5, 0.0)]
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    run_mod.cmd_check_cross(str(case), points="1,1", as_json=True)
    assert json.loads(capsys.readouterr().out)["ok"] is True
