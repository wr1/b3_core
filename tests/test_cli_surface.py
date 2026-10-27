"""CLI surfaces that must not start a finite-element solve."""

import json
import sys
from types import SimpleNamespace

import pytest

from b3_core.cases import uniaxial
from b3_core.fit import default_bounds
from b3_core.fit.spec import DesignTable, FitSpec, Target, TargetSet


def _case(tmp_path):
    path = tmp_path / "case.json"
    uniaxial().to_json(path)
    return path


def _params(tmp_path):
    params = default_bounds(uniaxial())
    params.params[0].fixed = False
    path = tmp_path / "params.json"
    path.write_text(params.model_dump_json(), encoding="utf-8")
    return path


def _targets(tmp_path):
    path = tmp_path / "targets.yaml"
    path.write_text(
        "targets:\n  - quantity: Ex\n    value: 1.0e6\n    rel_sd: 0.05\n",
        encoding="utf-8",
    )
    return path


def test_report_build_copies_a_template(tmp_path):
    from b3_core.cli_ext import cmd_report_build
    from b3_core.report.tables import markdown_table, typst_table

    project = tmp_path / "project.json"
    project.write_text(
        json.dumps(
            {
                "title": "Core @ report",
                "grid": {
                    "rows": [
                        {
                            "ok": True,
                            "kx": 5e-5,
                            "ky": 0.0,
                            "Ex": 1e6,
                            "Ez": 2e6,
                            "Gxz": 4e5,
                            "rho_infused": 120.0,
                        },
                        {"ok": False, "kx": 0.001, "ky": 0.0, "Ex": 1.0},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    template = tmp_path / "note.typ"
    template.write_text("= note\n", encoding="utf-8")
    cmd_report_build(str(project), out=str(tmp_path / "out"), template=str(template))
    assert (tmp_path / "out" / "report.md").is_file()
    assert (tmp_path / "out" / "note.typ").read_text(encoding="utf-8") == "= note\n"
    assert "\\@" in markdown_table(["a"], [["x@y"]])
    assert "\\@" in typst_table(["a"], [["x@y"]])


def test_cache_and_accept_commands(tmp_path):
    from b3_core.cli_ext import cmd_cache_export, cmd_cache_import, cmd_check_accept
    from b3_core.solvers.stamps import stamp_for

    root = tmp_path / "cache" / "ab"
    root.mkdir(parents=True)
    (root / "abcd.json").write_text(
        json.dumps(
            {
                "result": {"backend": "numpy"},
                "solver_stamp": stamp_for("numpy"),
                "input": {"curvature": {"kx": 0.0, "ky": 0.0}},
                "case_hash": "abc",
            }
        ),
        encoding="utf-8",
    )
    dest = tmp_path / "bundle.zip"
    cmd_cache_export(str(tmp_path / "cache"), str(dest), as_json=True)
    cmd_cache_import(str(dest), str(tmp_path / "other"), as_json=True)
    with pytest.raises(SystemExit):
        cmd_cache_export(str(tmp_path / "cache"), "")
    project = tmp_path / "empty.json"
    project.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        cmd_check_accept(str(project), as_json=True)
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        "checks:\n  - id: ties\n    note: slaves\n  - id: flat_card\n    quantities: [Ex]\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        cmd_check_accept(str(project), profile=str(profile))


def test_fit_commands_that_do_not_solve(tmp_path, monkeypatch):
    from b3_core.fit.commands import (
        cmd_fit_bounds,
        cmd_fit_design,
        cmd_fit_estimate_halo,
        cmd_fit_refine,
        cmd_fit_residuals,
        cmd_fit_rsm,
        cmd_fit_run,
        cmd_fit_sensitivity,
    )

    case = _case(tmp_path)
    cmd_fit_bounds(str(case), as_json=True)
    with pytest.raises(SystemExit):
        cmd_fit_estimate_halo(str(case), as_json=True)
    cmd_fit_estimate_halo(str(case), rho_infused=200.0, as_json=True)
    spec = FitSpec(
        case=uniaxial().model_dump(mode="json"),
        params=default_bounds(uniaxial()),
        targets=TargetSet(targets=[]),
        cache=None,
        stages={
            "halo_from_mass": False,
            "design": False,
            "rsm": False,
            "refine": {"max_solves": 1},
            "identify": False,
        },
    )
    spec_path = tmp_path / "fit.json"
    spec_path.write_text(spec.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(
        "b3_core.api.run_case",
        lambda *_a, **_k: SimpleNamespace(
            result=SimpleNamespace(properties={"Ex": 1.0e6, "Ey": 1.0e6, "Ez": 2.0e6}),
            geometry={"rho_infused": 100.0},
        ),
    )
    cmd_fit_run(str(spec_path), out=str(tmp_path / "calib"), as_json=True)
    assert (tmp_path / "calib" / "calibrated_case.json").is_file()

    params = _params(tmp_path)
    targets = _targets(tmp_path)

    def fake_evaluate(*_args, **_kwargs):
        return DesignTable(
            columns=["u0"],
            rows=[
                {
                    "ok": True,
                    "x_unit": [value],
                    "x": [1.0 + value],
                    "properties": {"Ex": 1.0e6 * (1.0 + 0.1 * value)},
                }
                for value in (-1.0, 0.0, 1.0)
            ],
        )

    monkeypatch.setattr(sys.modules["b3_core.fit.design"], "evaluate", fake_evaluate)
    cmd_fit_design(
        str(case), str(params), out=str(tmp_path / "design.json"), as_json=True
    )
    cmd_fit_rsm(
        str(tmp_path / "design.json"),
        str(targets),
        out=str(tmp_path / "rsm.json"),
        as_json=True,
    )
    assert (tmp_path / "rsm.json").is_file()

    cmd_fit_residuals(str(case), str(targets), as_json=True)

    class _Result:
        status = "converged"

        def model_dump(self, mode="json"):
            return {"status": self.status}

    monkeypatch.setattr(
        sys.modules["b3_core.fit.refine"],
        "refine",
        lambda *_a, **_k: _Result(),
    )
    cmd_fit_refine(str(case), str(params), str(targets), as_json=True)
    monkeypatch.setattr(
        "b3_core.fit.identify.sensitivity",
        lambda *_a, **_k: {"identifiable": True, "weak_params": []},
    )
    cmd_fit_sensitivity(str(case), str(params), str(targets), as_json=True)
    with pytest.raises(SystemExit):
        cmd_fit_residuals(str(case), str(tmp_path / "missing.yaml"))


def test_check_and_viz_commands_use_stand_ins(tmp_path, monkeypatch):
    from b3_core.cli_ext import (
        cmd_check_convergence,
        cmd_check_reproduce,
        cmd_diagnose_bisect,
        cmd_kerfs_map,
        cmd_viz_datasheet_sweep,
        cmd_viz_figs,
    )
    from b3_core.sweep.parallel import _limit_threads, map_workers

    _limit_threads()
    assert map_workers(lambda item: item + 1, [1], 1) == [2]
    with pytest.raises(ValueError):
        map_workers(lambda item: item, [1], 0)
    with pytest.raises(TypeError):
        map_workers(lambda item: item, [1, 2], 2)

    values = tmp_path / "values.json"
    values.write_text('{"points": []}', encoding="utf-8")
    cmd_check_reproduce(str(values), as_json=True)
    monkeypatch.setattr(
        "b3_core.checks.convergence.check_convergence",
        lambda *_a, **_k: {"ok": False, "rel_pct": 2.0},
    )
    with pytest.raises(SystemExit):
        cmd_check_convergence(str(_case(tmp_path)), as_json=True)
    monkeypatch.setattr(
        "b3_core.checks.bisect.bisect",
        lambda *_a, **_k: {"max_node_displacement_mm": 0.1, "backend": "numpy"},
    )
    cmd_diagnose_bisect(
        str(_case(tmp_path)), out=str(tmp_path / "bisect.json"), as_json=True
    )
    assert (tmp_path / "bisect.json").is_file()

    rows = []
    for kx in (-0.001, 0.0, 0.001):
        rows.append(
            {
                "kx": kx,
                "ky": 0.0,
                "ok": True,
                "Ex": 1e6,
                "Ey": 1e6,
                "Ez": 2e6,
                "Gxy": 4e5,
                "Gxz": 4e5,
                "Gyz": 4e5,
                "rho_infused": 100.0,
                "resin_vf": 0.02,
            }
        )
    grid = tmp_path / "grid.json"
    grid.write_text(
        json.dumps(
            {
                "rows": rows,
                "curves": [
                    {
                        "label": "x bottom",
                        "k": [-0.001, 0.0, 0.001],
                        "state": ["open", "flat", "closed"],
                        "opens_for": "k<0",
                    }
                ],
                "convention": [{"axis": "x", "mouth": "bottom", "opens_for": "k<0"}],
                "acceptance": {"ok": True, "checks": []},
            }
        ),
        encoding="utf-8",
    )
    cmd_viz_figs(str(grid), out=str(tmp_path / "figs"), as_json=True)
    assert (tmp_path / "figs" / "kerf_ranges_wedge.png").is_file()
    prose = tmp_path / "prose.json"
    prose.write_text(
        json.dumps(
            {
                "title": "Draft core",
                "summary": "Predicted.",
                "intended_use": "Review.",
                "limitations": "Draft only.",
            }
        ),
        encoding="utf-8",
    )
    cmd_viz_datasheet_sweep(
        str(grid),
        output=str(tmp_path / "sheet"),
        prose=str(prose),
        draft=True,
        cache=str(tmp_path / "cache"),
    )
    data = json.loads((tmp_path / "sheet" / "data.json").read_text(encoding="utf-8"))
    assert data["provenance"]["cache"] == str(tmp_path / "cache")
    assert (tmp_path / "sheet" / "datasheet.pdf").stat().st_size > 500

    monkeypatch.setattr(
        "b3_core.sweep.kerf_map.kerf_map",
        lambda *_a, **_k: {
            "grid": [],
            "convention": [{"opens_for": "k<0"}],
            "unit": "1/mm",
        },
    )
    cmd_kerfs_map(str(_case(tmp_path)), magnitude=0.0, out=str(tmp_path / "map.json"))
    assert (tmp_path / "map.json").is_file()


def test_axis_map_sorts_a_shear_name():
    from b3_core.fit.residuals import model_quantity

    mapped = model_quantity(
        Target(quantity="Gxz", value=1.0, rel_sd=0.1, axis_map={"xz": "zx"})
    )
    assert mapped == "Gxz"
