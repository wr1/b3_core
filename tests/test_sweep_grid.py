"""Curvature grid counts, the inert-axis guard, and a failed point."""

import json
from types import SimpleNamespace

import pytest

from b3_core.cases import plain, uniaxial
from b3_core.core import run as run_mod
from b3_core.sweep.curvature import InertAxisError, sweep_curvature, write_grid


def _record(kx: float, *, fail: bool = False):
    if fail:
        raise RuntimeError(f"boom at {kx}")
    props = {
        "Ex": 1.0,
        "Ey": 2.0,
        "Ez": 3.0,
        "Gxy": 0.4,
        "Gxz": 0.5,
        "Gyz": 0.6,
        "nuxy": 0.3,
        "nuxz": 0.2,
        "nuyz": 0.1,
    }
    return SimpleNamespace(
        result=SimpleNamespace(properties=props, backend="fake"),
        case_hash="abc",
        provenance=None,
        diagnostics={"checks": [{"id": "ties", "status": "ok"}], "ties": []},
        geometry={"rho_infused": 120.0, "resin_vf": 0.1},
        meta_cache_hit=False,
    )


def test_dry_run_includes_the_flat_point_and_counts_the_product():
    result = sweep_curvature(
        uniaxial(),
        "2:4:2",
        "0",
        unit="1/m",
        dry_run=True,
    )
    manifest = result["manifest"]
    assert manifest["n_solves"] == 3  # 0 plus the two converted samples, ky = 0
    assert manifest["kx"][0] == 0.0
    assert manifest["kx"][1] == pytest.approx(0.002)
    assert result["rows"] == []


def test_inert_ky_is_an_error_unless_acknowledged():
    with pytest.raises(InertAxisError, match="ky"):
        sweep_curvature(uniaxial(), [0.0], [0.0, 0.001], dry_run=True)
    result = sweep_curvature(
        uniaxial(),
        [0.0],
        [0.0, 0.001],
        allow_inert_axis=True,
        dry_run=True,
    )
    assert result["manifest"]["ky_inert"] is True


def test_a_failed_point_is_a_row_and_the_rest_continue():
    def solve(case, *, backend, cache):
        return _record(case.curvature.kx, fail=case.curvature.kx > 0)

    result = sweep_curvature(
        uniaxial(),
        [0.0, 0.001],
        [0.0],
        backend="numpy",
        solve=solve,
    )
    by_kx = {row["kx"]: row for row in result["rows"]}
    assert by_kx[0.0]["ok"] is True
    assert by_kx[0.0]["case_hash"] == "abc"
    assert by_kx[0.0]["diagnostics"]["checks"][0]["id"] == "ties"
    assert by_kx[0.001]["ok"] is False
    assert "boom" in by_kx[0.001]["error"]
    assert result["manifest"]["n_failed"] == 1


def test_cli_dry_run_prints_the_count(tmp_path, capsys):
    case = tmp_path / "case.json"
    plain().to_json(case)
    run_mod.cmd_sweep_grid(
        str(case), kx="0", ky="0", dry_run=True, as_json=True, root="", cache=""
    )
    out = capsys.readouterr().out
    assert "n_solves=1" in out
    payload = json.loads(out.strip().splitlines()[-1])
    assert payload["dry_run"] is True
    assert payload["backend"] == "fenicsx"


def test_write_grid_csv_and_manifest(tmp_path):
    result = sweep_curvature(
        uniaxial(),
        [0.0],
        [0.0],
        solve=lambda case, **kwargs: _record(0.0),
    )
    write_grid(result, tmp_path)
    assert (tmp_path / "grid.csv").is_file()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["n_solves"] == 1
    assert "parquet" in manifest
