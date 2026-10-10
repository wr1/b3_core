"""Branches the publish path must hit without starting a finite-element solve."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from b3_core.cases import uniaxial
from b3_core.fit import default_bounds
from b3_core.fit.spec import (
    DesignTable,
    FitSpec,
    ParamSet,
    ParamSpec,
    Prior,
    Target,
    TargetSet,
)
from b3_core.models import Curvature


def _stand_in(point, *, backend, cache):
    del backend, cache
    young = float(point.core.E)
    return SimpleNamespace(
        result=SimpleNamespace(
            properties={
                "Ex": young,
                "Ey": young,
                "Ez": young * 1.5,
                "Gxy": young / 3.0,
                "Gxz": young / 3.0,
                "Gyz": young / 3.0,
                "nuxy": 0.3,
                "nuxz": 0.3,
                "nuyz": 0.3,
            },
            backend="numpy",
            stiffness=np.eye(6),
        ),
        geometry={"rho_infused": 100.0, "resin_vf": 0.02},
        provenance=None,
        diagnostics={"checks": [], "ties": []},
        case_hash="abc",
        solver_stamp="numpy/test",
        meta_cache_hit=False,
    )


def _free_scale():
    params = default_bounds(uniaxial())
    params.params = [
        param.model_copy(
            update={
                "fixed": param.name != "foam_E_scale",
                "prior": Prior(kind="normal", mean=1.0, sd=0.2),
            }
        )
        for param in params.params
    ]
    return params


def test_run_fit_chains_stages_on_a_stand_in(monkeypatch):
    from b3_core.fit.run import run_fit
    from b3_core.fit.volume import estimate_halo

    def rho(_case, cell):
        return 100.0 + 20.0 * float(cell)

    monkeypatch.setattr("b3_core.fit.volume._rho", rho)
    params = _free_scale()
    targets = TargetSet(
        targets=[
            Target(quantity="Ex", value=float(uniaxial().input.core.E), rel_sd=0.05),
            Target(quantity="rho_infused", value=110.0),
        ]
    )
    spec = FitSpec(
        case=uniaxial().model_dump(mode="json"),
        params=params,
        targets=targets,
        backend="numpy",
        cache=None,
        workers=1,
        stages={
            "halo_from_mass": True,
            "design": {"method": "lhs", "n": 4, "seed": 1},
            "rsm": {"log": False, "ridge": 1e-3},
            "refine": {"max_solves": 6, "tol_sigma": 5.0},
            "identify": {"step": 0.05},
        },
    )
    result = run_fit(spec, solve=_stand_in)
    assert result.provenance["fitspec_hash"]
    assert result.calibrated_case["core"]["source"] == "calibrated"
    halo = next(param for param in spec.params.params if param.name == "halo_cell_size")
    assert halo.value == pytest.approx(0.5, abs=0.05)

    surface_only = spec.model_copy(
        update={
            "stages": {
                "design": {"method": "ccd", "n": 3},
                "rsm": True,
                "refine": False,
            }
        }
    )
    rsm = run_fit(surface_only, solve=_stand_in)
    assert rsm.status == "rsm_only"
    with pytest.raises(ValueError, match="refine is off"):
        run_fit(
            spec.model_copy(update={"stages": {"design": False, "refine": False}}),
            solve=_stand_in,
        )
    estimate_halo(uniaxial(), rho_infused=100.0, bounds=(0.0, 2.0))
    estimate_halo(uniaxial(), rho_infused=140.0, bounds=(0.0, 2.0))
    estimate_halo(uniaxial(), uptake_kg_m2=3.3, bounds=(0.0, 2.0))
    with pytest.raises(ValueError):
        estimate_halo(uniaxial(), rho_infused=1.0, param="other")


def test_design_methods_and_a_failed_row(tmp_path):
    from b3_core.fit.design import design, evaluate

    params = _free_scale()
    assert design(ParamSet(params=[]), method="ccd").size == 0
    assert design(params, method="sobol", n=3, seed=0).shape[1] == 1
    assert design(params, method="LHS", n=3, seed=0).shape == (3, 1)
    padded = design(params, method="ccd", n=6, seed=2)
    assert len(padded) == 6
    with pytest.raises(ValueError, match="unknown design"):
        design(params, method="grid")

    def boom(*_a, **_k):
        raise RuntimeError("stand-in failed")

    empty = evaluate(uniaxial(), params, np.zeros((0, 1)), solve=_stand_in)
    assert empty.rows == []
    failed = evaluate(uniaxial(), params, np.array([[0.0]]), solve=boom)
    assert failed.rows[0]["ok"] is False
    cache = SimpleNamespace(root=str(tmp_path / "cache"))
    held = evaluate(
        uniaxial(),
        params,
        np.array([[0.0], [0.5]]),
        cache=cache,
        solve=_stand_in,
    )
    assert held.rows[0]["ok"] is True


def test_rsm_gradient_prior_and_empty_free_set():
    from b3_core.fit.rsm import ResponseSurface, fit_quadratic, solve_rsm

    surface = ResponseSurface(
        names=["Ex"],
        terms=["1", "u0", "u0^2"],
        coefs={"Ex": [1.0, 0.1, 0.0]},
        log=False,
    )
    assert surface.predict([0.0])["Ex"] == pytest.approx(1.0)
    assert len(surface.grad([0.0])["Ex"]) == 1
    with pytest.raises(ValueError, match="unit vector"):
        surface.predict([0.0, 1.0])
    params = _free_scale()
    params.params[0].prior = Prior(kind="lognormal", mean=1.0, sd=0.2)
    table = DesignTable(
        columns=["foam_E_scale"],
        rows=[
            {"ok": True, "x_unit": [value], "properties": {"Ex": 1.0 + value}}
            for value in (-0.5, 0.0, 0.5)
        ],
    )
    targets = TargetSet(
        targets=[Target(quantity="Ex", value=1.0, sd=0.1, basis="neat")]
    )
    fitted = fit_quadratic(table, targets, log=False)
    start = solve_rsm(fitted, targets, params)
    assert start.shape == (1,)
    fixed = ParamSet(params=[params.params[0].model_copy(update={"fixed": True})])
    assert solve_rsm(fitted, TargetSet(targets=[]), fixed).size == 0
    plain = _free_scale()
    plain.params[0].prior = Prior(kind="uniform")
    assert solve_rsm(fitted, TargetSet(targets=[]), plain).shape == (1,)


def test_refine_rejects_an_analytic_jacobian_and_a_log_prior():
    from b3_core.fit.refine import refine

    params = _free_scale()
    params.params[0].prior = Prior(kind="lognormal", mean=1.0, sd=0.25)
    targets = TargetSet(targets=[Target(quantity="Ex", value=1.0e8, rel_sd=0.2)])
    with pytest.raises(ValueError, match="jac"):
        refine(uniaxial(), params, targets, jac="3-point", solve=_stand_in)
    result = refine(
        uniaxial(),
        params,
        targets,
        backend="numpy",
        solve=_stand_in,
        max_solves=4,
        tol_sigma=1.0e9,
    )
    assert result.status == "converged"


def test_sensitivity_edges():
    from b3_core.fit.identify import sensitivity

    params = default_bounds(uniaxial())
    targets = TargetSet(targets=[Target(quantity="Ex", value=1.0, rel_sd=0.05)])
    idle = sensitivity(uniaxial(), params, targets, solve=_stand_in)
    assert idle.identifiable is True
    free = _free_scale()
    zero = sensitivity(uniaxial(), free, targets, solve=_stand_in, step=0.0)
    assert zero.weak_params


def test_materials_and_parameter_paths():
    from b3_core.fit.materials import (
        estimate_foam,
        resin_typical,
        scaling_family,
        split_basis,
    )
    from b3_core.fit.params import apply

    with pytest.raises(ValueError, match="unknown foam"):
        scaling_family("nope")
    with pytest.raises(ValueError, match="unknown resin"):
        resin_typical("vinyl")
    foam = estimate_foam(80.0)
    assert foam.source == "estimated"
    ortho = {
        "Ex": 1e6,
        "Ey": 1e6,
        "Ez": 2e6,
        "Gxy": 4e5,
        "Gxz": 4e5,
        "Gyz": 4e5,
        "nuxy": 0.3,
        "nuxz": 0.3,
        "nuyz": 0.3,
        "rho": 90.0,
    }
    neat = [
        {"quantity": name, "value": value, "basis": "neat"}
        for name, value in ortho.items()
        if name != "rho"
    ]
    neat.append({"quantity": "rho_infused", "value": ortho["rho"], "basis": "neat"})
    neat.append({"quantity": "Ex", "value": None, "basis": "neat"})
    neat.append(
        {
            "quantity": "Ex",
            "value": 3.0e9,
            "basis": "neat",
            "source": "resin",
        }
    )
    neat.append(
        {
            "quantity": "rho_infused",
            "value": 1100.0,
            "basis": "neat",
            "source": "resin",
        }
    )
    neat.append({"quantity": "Ex", "value": 1.0e6, "basis": "infused"})
    materials, infused = split_basis(neat)
    assert "core" in materials and "resin" in materials
    assert infused.targets[0].basis == "infused"
    with pytest.raises(ValueError, match="needs rho"):
        split_basis([{"quantity": "Ex", "value": 1.0, "basis": "neat"}])
    with pytest.raises(ValueError, match="needs E"):
        split_basis(
            [
                {"quantity": "rho_infused", "value": 90.0, "basis": "neat"},
                {"quantity": "nuxy", "value": 0.3, "basis": "neat"},
            ]
        )
    case = uniaxial().input
    logged = ParamSet(
        params=[
            ParamSpec(
                name="resin_E",
                path="resin.E",
                transform="log",
                value=1.0,
                fixed=True,
            )
        ]
    )
    updated = apply(case, logged)
    assert updated.resin.E == pytest.approx(np.exp(1.0))
    with pytest.raises(ValueError, match="cannot apply"):
        apply(
            case,
            ParamSet(
                params=[
                    ParamSpec(name="bad", path="not a path", transform="abs", value=1.0)
                ]
            ),
        )
    with pytest.raises(ValueError, match="x has length"):
        apply(case, _free_scale(), [1.0, 2.0])


def test_sweep_curvature_records_failures_and_writes(tmp_path, monkeypatch):
    from b3_core.sweep.curvature import InertAxisError, sweep_curvature, write_grid

    case = uniaxial()
    dry = sweep_curvature(case, "0:0.001:2", "0", unit="1/m", dry_run=True)
    assert dry["manifest"]["n_solves"] >= 2
    with pytest.raises(InertAxisError):
        sweep_curvature(case, [0.0], [0.0, 0.001], solve=_stand_in)
    allowed = sweep_curvature(
        case,
        [0.0, 0.001],
        [0.0],
        cell_sizes=[0.0, 0.4],
        allow_inert_axis=False,
        solve=_stand_in,
        fit_result=str(_fit_file(tmp_path)),
    )
    assert allowed["manifest"]["fit"]["status"] == "converged"
    assert any(row["ok"] for row in allowed["rows"])

    def boom(*_a, **_k):
        raise RuntimeError("mesh")

    failed = sweep_curvature(case, [0.0], [0.0], solve=boom, workers=1)
    assert failed["rows"][0]["ok"] is False
    with pytest.raises(ValueError, match="workers"):
        sweep_curvature(case, [0.0], [0.0], solve=_stand_in, workers=0)
    monkeypatch.setattr(
        pd.DataFrame,
        "to_parquet",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("pq")),
    )
    dest = write_grid(failed, tmp_path / "grid")
    manifest = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["parquet"] is False


def _fit_file(tmp_path: Path) -> Path:
    path = tmp_path / "fit.json"
    path.write_text(
        json.dumps(
            {
                "status": "converged",
                "residuals": {"chi2": 0.1, "max_abs_z": 0.2, "n_used": 1},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_figures_accept_a_kerf_grid_and_a_bare_list(tmp_path):
    from b3_core.viz.figs.curvature import _curves, _flat, _pct, _rows, write_figures

    rows = [
        {
            "ok": True,
            "kx": 0.0,
            "ky": 0.0,
            "Ex": 1.0,
            "Ey": 1.0,
            "Ez": 2.0,
            "Gxy": 1.0,
            "Gxz": 1.0,
            "Gyz": 1.0,
            "rho_infused": 10.0,
            "resin_vf": 0.01,
        },
        {
            "ok": True,
            "kx": 0.001,
            "ky": 0.0,
            "Ex": 1.1,
            "Ey": 1.0,
            "Ez": None,
            "Gxy": 1.0,
            "Gxz": 1.0,
            "Gyz": 1.0,
            "rho_infused": 11.0,
            "resin_vf": 0.02,
        },
    ]
    assert _pct(rows[1], rows[0], "Ez") is None
    assert _rows(rows)[0]["kx"] == 0.0
    with pytest.raises(ValueError, match="no flat"):
        _flat([{"kx": 0.001, "ky": 0.0, "ok": True}])
    grid = {
        "grid": [
            {
                "kx": -0.001,
                "ky": 0.0,
                "kerfs": [
                    {
                        "axis": "x",
                        "mouth": "bottom",
                        "opens_for": "k<0",
                        "state": "open",
                    }
                ],
            }
        ]
    }
    assert _curves(grid)[0]["opens_for"] == "k<0"
    with pytest.raises(ValueError, match="wedge"):
        _curves({"rows": rows})
    written = write_figures({"rows": rows, "curves": _curves(grid)}, tmp_path / "figs")
    assert "kerf_ranges_wedge" in written


def test_publish_gate_reads_paths_and_flags(tmp_path):
    from b3_core.checks.accept import check_accept

    grid = {
        "rows": [
            {
                "ok": False,
                "kx": 0.0,
                "ky": 0.0,
                "diagnostics": json.dumps(
                    {
                        "checks": [{"id": "energy", "status": "fail"}],
                        "ties": [{"face": "z", "ok": False}],
                    }
                ),
            }
        ],
        "manifest": {"kx_inert": True, "allow_inert_axis": True},
    }
    grid_path = tmp_path / "grid.json"
    grid_path.write_text(json.dumps(grid), encoding="utf-8")
    fit_path = tmp_path / "fit.json"
    fit_path.write_text(
        json.dumps({"status": "no_free_params", "residuals": {}}), encoding="utf-8"
    )
    backends = tmp_path / "backends.json"
    backends.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "kx": 0.0,
                        "ky": 0.0,
                        "flags": ["flat_anchor"],
                        "pairs": {"fenicsx_vs_mfem": {"max_rel_pct": 0.2}},
                    },
                    {
                        "kx": 0.001,
                        "ky": 0.0,
                        "pairs": {"fenicsx_vs_numpy": {"max_rel_pct": 0.2}},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    reproduced = tmp_path / "repro.json"
    reproduced.write_text(json.dumps({"ok": True, "rows": []}), encoding="utf-8")
    provenance = tmp_path / "prov.json"
    provenance.write_text(
        json.dumps({"commit": "abc", "dirty": False}), encoding="utf-8"
    )
    convention = tmp_path / "conv.yaml"
    convention.write_text(
        "rows:\n  - axis: x\n    mouth: bottom\n    opens_for: k<0\n",
        encoding="utf-8",
    )
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        "checks:\n  - id: card\n    quantities: [Ex]\n    rtol_pct: 5\n  - id: ties\n    note: slaves\n",
        encoding="utf-8",
    )
    project = tmp_path / "project.yaml"
    project.write_text(
        "\n".join(
            [
                f"grid: {grid_path}",
                f"fit_result: {fit_path}",
                f"backends_report: {backends}",
                f"reproduce: {reproduced}",
                f"provenance: {provenance}",
                f"convention: {convention}",
                "cache: ''",
                "reference:",
                "  Ex: 1",
            ]
        ),
        encoding="utf-8",
    )
    report = check_accept(project, profile=profile)
    ids = {item["id"]: item["ok"] for item in report["checks"]}
    assert ids["fit"] is True
    assert ids["profile:card"] is True
    assert ids["profile:ties"] is True
    assert report["ok"] is False


def test_backend_comparison_uses_an_injected_solver():
    from b3_core.checks.backends import (
        check_backends,
        check_cross,
        format_table,
        parse_axis,
        parse_points,
        refine_madd,
    )
    from b3_core.pipeline import BackendCapabilityError, _curved_choice
    from b3_core.solvers import UnknownBackendError, register
    from b3_core.solvers.protocol import Capabilities

    with pytest.raises(ValueError):
        parse_points("0")
    with pytest.raises(ValueError):
        parse_points("   ")
    with pytest.raises(ValueError):
        parse_axis("0:1:0")
    assert refine_madd([0.0], 3) == [0.0]
    case = uniaxial()
    curved = case.input.model_copy(
        update={"curvature": Curvature(kx=1e-3, ky=0.0), "allow_pair_periodicity": True}
    )
    ortho = case.input.model_copy(deep=True)
    ortho.core.Ex = 1.0e8
    ortho.core.Ey = 1.0e8
    ortho.core.Ez = 1.0e8
    ortho.core.Gxy = 4.0e7
    ortho.core.Gxz = 4.0e7
    ortho.core.Gyz = 4.0e7
    ortho.core.nuxy = 0.3
    ortho.core.nuxz = 0.3
    ortho.core.nuyz = 0.3
    ortho.core.cell_size = 0.5

    def solver(prepared, name, kx, ky, pair, cache):
        del prepared, kx, ky, pair, cache
        if name == "numpy":
            raise RuntimeError("skip")
        record = _stand_in(ortho, backend="numpy", cache=None)
        record.result.backend = name
        return record

    report = check_backends(
        ortho,
        backends=["numpy", "ccx", "mfem"],
        points=[(0.0, 0.0), (1.0e-3, 0.0)],
        refine=1,
        solve=solver,
        check_installed=False,
    )
    assert format_table(report).endswith("\n")
    flat = check_cross(
        case, backends=["numpy"], solve=_solver_ok, check_installed=False
    )
    assert flat["rows"][0]["kx"] == 0.0
    with pytest.raises(UnknownBackendError):
        _curved_choice(curved, "not-a-backend")

    class _NoInterp:
        name = "nointerp_test"
        capabilities = Capabilities(
            orthotropic=True,
            halo=True,
            face_layer=True,
            displacements=False,
            interpolated_periodicity=False,
        )

    class _YesInterp(_NoInterp):
        name = "yesinterp_test"
        capabilities = Capabilities(
            orthotropic=True,
            halo=True,
            face_layer=True,
            displacements=False,
            interpolated_periodicity=True,
        )

    register(_NoInterp)
    register(_YesInterp)
    try:
        with pytest.raises(BackendCapabilityError):
            _curved_choice(curved, "nointerp_test")
        assert _curved_choice(curved, "yesinterp_test") == "yesinterp_test"
    finally:
        from b3_core import solvers as solvers_mod

        solvers_mod._REGISTRY.pop("nointerp_test", None)
        solvers_mod._REGISTRY.pop("yesinterp_test", None)


def _solver_ok(prepared, name, kx, ky, pair, cache):
    del prepared, name, kx, ky, pair, cache
    return _stand_in(uniaxial().input, backend="numpy", cache=None)


def test_cli_callbacks_do_not_solve(tmp_path, monkeypatch, capsys):
    from b3_core.core import run as run_mod

    case = tmp_path / "case.json"
    uniaxial().to_json(case)
    with pytest.raises(SystemExit):
        run_mod.cmd_doctor(as_json=True)
    with pytest.raises(SystemExit):
        run_mod.cmd_doctor(as_json=False)
    run_mod.cmd_skill(stdout=False)
    text = capsys.readouterr().out
    assert "SKILL.md" in text or text
    run_mod.cmd_skill(stdout=True)
    assert "b3_core" in capsys.readouterr().out

    assert run_mod.resolve_sweep_root(str(tmp_path)) == tmp_path
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        run_mod.resolve_sweep_root("")
    (tmp_path / "bases").mkdir()
    assert run_mod.resolve_sweep_root("") == tmp_path

    run_mod.cmd_sweep_grid(
        str(case),
        kx="-1:1:3",
        ky="0",
        unit="R-m",
        dry_run=True,
        as_json=True,
        cache=str(tmp_path / "cache"),
        allow_inert_axis=True,
    )
    assert "n_solves" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        run_mod.cmd_sweep_grid(str(case), ky="0:0.001:2", as_json=True)

    monkeypatch.setattr("b3_core.sweep.homogenize.run_thickness", lambda *_a, **_k: 0)
    monkeypatch.setattr("b3_core.sweep.homogenize.run_curvature", lambda *_a, **_k: 3)
    monkeypatch.setattr("b3_core.sweep.homogenize.run_patterns", lambda *_a, **_k: 0)
    with pytest.raises(SystemExit):
        run_mod.cmd_sweep_thickness(str(tmp_path), "")
    with pytest.raises(SystemExit) as curved:
        run_mod.cmd_sweep_curvature(str(tmp_path), str(tmp_path / "cache"))
    assert curved.value.code == 3
    with pytest.raises(SystemExit):
        run_mod.cmd_sweep_patterns(str(tmp_path), "")

    run_mod.cmd_cache_stats(directory=str(tmp_path / "empty-cache"))
    run_mod.cmd_cache_inspect(directory=str(tmp_path / "empty-cache"), backend="numpy")
    with pytest.raises(SystemExit):
        run_mod.cmd_cache_purge(directory=str(tmp_path / "empty-cache"))
    run_mod.cmd_cache_purge(
        directory=str(tmp_path / "empty-cache"), stale=True, as_json=True
    )

    points = run_mod._select_points(
        cross=False, points="1,0 0,1", kx="", ky="", grid=False, unit="1/m"
    )
    assert points[0][0] == pytest.approx(0.001)
    grid = run_mod._select_points(
        cross=False, points="", kx="0:1:2", ky="0", grid=True, unit="R-mm"
    )
    assert len(grid) == 2
    assert run_mod._select_points(cross=True, points="", kx="1", ky="1", grid=True) == [
        (0.0, 0.0)
    ]

    def fake_check(*_a, **_k):
        return {
            "ok": False,
            "rows": [
                {
                    "kx": 0.0,
                    "ky": 0.0,
                    "refine": 0,
                    "pairs": {},
                    "flags": ["no_backends"],
                    "backends": {"fenicsx": {"skipped": "missing"}},
                }
            ],
            "rtol": 1e-4,
            "worst_rel_pct": 0.0,
            "wall_s": 0.0,
        }

    monkeypatch.setattr("b3_core.checks.backends.check_backends", fake_check)
    monkeypatch.setattr("b3_core.checks.backends.check_cross", fake_check)
    monkeypatch.setattr(
        "b3_core.checks.backends.format_table", lambda report: "table\n"
    )
    with pytest.raises(SystemExit):
        run_mod._dispatch_check(
            False, str(case), "numpy", "", "", "", False, 0, 1e-4, "", "", False
        )
    out = tmp_path / "check" / "out.json"
    with pytest.raises(SystemExit):
        run_mod._dispatch_check(
            True,
            str(case),
            "numpy",
            "",
            "0",
            "0",
            False,
            0,
            1e-4,
            str(tmp_path / "cache"),
            str(out),
            True,
            "1/mm",
        )
    assert out.is_file()

    def explode(*_a, **_k):
        raise RuntimeError("no solve")

    monkeypatch.setattr("b3_core.checks.backends.check_backends", explode)
    with pytest.raises(SystemExit):
        run_mod.cmd_check_backends(str(case), as_json=True)
    with pytest.raises(RuntimeError):
        run_mod.cmd_check_backends(str(case), as_json=False)

    record = SimpleNamespace(
        diagnostics={"checks": [{"id": "energy", "status": "fail"}], "ties": []},
        to_agent_json=lambda **_k: {"ok": True},
    )
    monkeypatch.setattr("b3_core.api.run_case", lambda *_a, **_k: record)
    with pytest.raises(SystemExit):
        run_mod.cmd_run(str(case), as_json=False, no_write=True, strict=True)
    with pytest.raises(RuntimeError):
        monkeypatch.setattr(
            "b3_core.api.run_case",
            lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("x")),
        )
        run_mod.cmd_run(str(case), as_json=False, no_write=True)

    run_mod.cmd_viz_sign(output=str(tmp_path / "sign.png"), kx=4.0, unit="1/m")
    assert (tmp_path / "sign.png").is_file()

    class _View:
        @classmethod
        def from_json(cls, _path):
            return cls()

        def serve(self, path):
            Path(path).write_text("served", encoding="utf-8")

        def geometry_png(self, path, cutaway=False):
            del cutaway
            Path(path).write_text("g", encoding="utf-8")

        def slices_png(self, path):
            Path(path).write_text("s", encoding="utf-8")

        def deformation_png(self, path, warp=0.0):
            del warp
            Path(path).write_text("d", encoding="utf-8")

        def modulus_surface_png(self, path):
            Path(path).write_text("m", encoding="utf-8")

        def modulus_polar_png(self, path):
            Path(path).write_text("p", encoding="utf-8")

        def stiffness_heatmap_png(self, path):
            Path(path).write_text("h", encoding="utf-8")

        def gallery(self, path):
            Path(path).write_text("gallery", encoding="utf-8")

    monkeypatch.setattr("b3_core.viz.GroovedCoreView", _View)
    run_mod.cmd_viz_view(str(case), "gallery", "", str(tmp_path / "view.html"), 0.2)
    run_mod.cmd_viz_view(str(case), "all", str(tmp_path / "all"), "", 0.2)
    run_mod.cmd_viz_view(str(case), "slices", "", "", 0.2)
    assert (tmp_path / "view.html").is_file()

    monkeypatch.setattr(
        "b3_core.viz.halo.render_halo_figures",
        lambda *_a, **_k: [tmp_path / "halo.png"],
    )
    monkeypatch.setattr(
        "b3_core.viz.halo.render_halo_curvature_figures",
        lambda *_a, **_k: [tmp_path / "curve.png"],
    )
    monkeypatch.setattr(
        "b3_core.deformed.render_deformed_modes",
        lambda *_a, **_k: None,
    )
    sibling = tmp_path / "case_halo.json"
    sibling.write_text(case.read_text(encoding="utf-8"), encoding="utf-8")
    sharp = tmp_path / "case.json"
    assert run_mod._sharp_sibling(str(sibling)) == sharp
    run_mod.cmd_viz_halo(str(sibling), str(tmp_path / "img"), "")
    run_mod.cmd_viz_halo_curvature(
        str(tmp_path / "img"), str(case), 1.0, -1.0, unit="1/m"
    )
    run_mod.cmd_viz_deformed(str(case), str(tmp_path / "def.png"), 0.2)

    monkeypatch.setattr(
        "b3_core.datasheet.generate", lambda *_a, **_k: SimpleNamespace()
    )
    run_mod.cmd_viz_datasheet(
        str(case), output=str(tmp_path / "one.pdf"), png="fig.png"
    )

    class _Surrogate:
        targets = ["Ex"]

        def to_json(self, path):
            Path(path).write_text("{}", encoding="utf-8")

        @classmethod
        def from_json(cls, _path):
            return cls()

        def lookup(self, kx, cell_size=0.0):
            del cell_size
            return pd.DataFrame({"kx": kx, "Ex": np.ones(len(kx))})

    monkeypatch.setattr(
        "b3_core.physics_surrogate.fit_from_homogenization", lambda **_k: _Surrogate()
    )
    monkeypatch.setattr("b3_core.physics_surrogate.CorePhysicsSurrogate", _Surrogate)
    with pytest.raises(SystemExit):
        run_mod.cmd_surrogate_fit("", "")
    run_mod.cmd_surrogate_fit(str(tmp_path / "surr.json"), str(tmp_path / "cache"))
    run_mod.cmd_surrogate_lookup(
        str(tmp_path / "surr.json"), "0,1", 0.2, str(tmp_path / "look.csv"), "1/m"
    )
    assert (tmp_path / "look.csv").is_file()
    run_mod.cmd_surrogate_lookup(str(tmp_path / "surr.json"), "", 0.0, "", "1/mm")

    argv = run_mod._rewrite_payload_json(["--json"])
    assert argv == ["--json"]
    assert run_mod._rewrite_payload_json(["run", "--json"]) == ["run", "--agent-json"]
    assert run_mod._rewrite_payload_json(["run"]) == ["run"]


def test_provenance_fallbacks(monkeypatch):
    import b3_core.provenance as provenance

    def timeout(*_a, **_k):
        raise subprocess.TimeoutExpired(cmd="git", timeout=1)

    monkeypatch.setattr(provenance.subprocess, "run", timeout)
    assert provenance._git_identity() == (None, None)

    def failed(*_a, **_k):
        return SimpleNamespace(returncode=1, stdout="", stderr="")

    monkeypatch.setattr(provenance.subprocess, "run", failed)
    assert provenance._git_identity() == (None, None)

    calls = {"n": 0}

    def split(*_a, **_k):
        calls["n"] += 1
        code = 0 if calls["n"] == 1 else 1
        return SimpleNamespace(returncode=code, stdout="abc\n", stderr="")

    monkeypatch.setattr(provenance.subprocess, "run", split)
    assert provenance._git_identity()[0] == "abc"

    monkeypatch.setattr(provenance, "_git_root", lambda: None)

    class _Dist:
        def read_text(self, _name):
            return json.dumps({"vcs_info": {"commit_id": "1234567890abcdef"}})

    monkeypatch.setattr("importlib.metadata.distribution", lambda _name: _Dist())
    assert provenance._commit_from_metadata() == "1234567890ab"
    monkeypatch.setattr(
        "importlib.metadata.distribution",
        lambda _name: SimpleNamespace(read_text=lambda _n: "not-json"),
    )
    assert provenance._commit_from_metadata() is None
    monkeypatch.setattr(
        "importlib.metadata.distribution",
        lambda _name: (_ for _ in ()).throw(RuntimeError()),
    )
    assert provenance.code_revision() == (None, None)
    assert provenance._module_version("json")
    monkeypatch.setattr(provenance.shutil, "which", lambda _name: None)
    assert provenance._ccx_version() is None
    monkeypatch.setattr(provenance.shutil, "which", lambda _name: "/usr/bin/ccx")
    monkeypatch.setattr(
        provenance.subprocess,
        "run",
        lambda *_a, **_k: (_ for _ in ()).throw(subprocess.TimeoutExpired("ccx", 1)),
    )
    assert provenance._ccx_version() == "present"
    monkeypatch.setattr(
        provenance.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(returncode=0, stdout="no tag\n", stderr=""),
    )
    assert provenance._ccx_version() == "present"
    monkeypatch.setattr(
        provenance.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(
            returncode=0, stdout="CalculiX Version 2.22\n", stderr=""
        ),
    )
    assert "Version" in provenance._ccx_version()


def test_sheet_compile_and_tensor_directions(tmp_path, monkeypatch):
    from b3_core.report.sheet import DatasheetRejected, _cell, _curved, generate_full
    from b3_core.solvers.elasticity import (
        constituent_dict,
        face_material_dict,
        isotropic_C,
        scoring_payload,
    )
    from b3_core.viz.tensor import (
        linear_compressibility,
        modulus_surface,
        poisson_ratio,
        shear_modulus,
    )

    assert _curved({"kx": "bad", "grid_rows": [{"kx": 0.001, "ky": 0.0}]}) is True
    assert _cell({"a": True}) == '{"a": true}'
    data = {
        "title": "Draft",
        "acceptance": {"ok": False},
        "prose": {},
        "properties": {},
        "convention": [],
        "figures": {},
        "kx": 0.0,
        "ky": 0.0,
        "grid_rows": [],
    }
    with pytest.raises(DatasheetRejected):
        generate_full(tmp_path / "refused", data, draft=False, compile=False)

    import types

    typst_mod = types.ModuleType("typst")

    def compile_typst(_src, output):
        Path(output).write_bytes(b"%PDF-1.4\n")

    typst_mod.compile = compile_typst
    monkeypatch.setitem(sys.modules, "typst", typst_mod)
    pdf = generate_full(tmp_path / "py", data, draft=True, compile=True)
    assert pdf.is_file()

    monkeypatch.delitem(sys.modules, "typst", raising=False)
    monkeypatch.setattr("b3_core.report.sheet.shutil.which", lambda _name: None)
    with pytest.raises(RuntimeError, match="not installed"):
        generate_full(tmp_path / "missing", data, draft=True, compile=True)
    monkeypatch.setattr("b3_core.report.sheet.shutil.which", lambda _name: "typst")
    monkeypatch.setattr(
        "b3_core.report.sheet.subprocess.run",
        lambda *_a, **_k: SimpleNamespace(returncode=2, stderr="bad"),
    )
    with pytest.raises(RuntimeError, match="typst compile failed"):
        generate_full(tmp_path / "bad", data, draft=True, compile=True)

    class _Pairs:
        def keys(self):
            return ["E_x", "nu"]

        def __getitem__(self, key):
            return {"E_x": 1.0e9, "nu": 0.3}[key]

    assert constituent_dict(_Pairs())["nu"] == 0.3
    assert constituent_dict({"E1": 1.0e9, "nu": 0.3})["Ex"] == 1.0e9
    assert face_material_dict(None) is None
    assert face_material_dict({}) is None
    face = face_material_dict({"thickness": 1.0})
    assert face["E"] == 12_000_000_000.0
    assert scoring_payload(None) is None
    assert scoring_payload({}) is None
    stiffness = isotropic_C(1.0e9, 0.3)
    direction = np.array([1.0, 0.0, 0.0])
    lateral = np.array([0.0, 1.0, 0.0])
    assert linear_compressibility(stiffness, direction) > 0
    assert shear_modulus(stiffness, direction, lateral) > 0
    assert poisson_ratio(stiffness, direction, lateral) == pytest.approx(0.3, abs=1e-6)
    surface = modulus_surface(stiffness, kind="beta", resolution=6)
    assert surface.n_points > 0
    with pytest.raises(ValueError):
        modulus_surface(stiffness, kind="nope", resolution=4)


def test_training_frame_and_checks_without_a_solver(tmp_path, monkeypatch):
    from b3_core.checks.convergence import check_convergence
    from b3_core.checks.reproduce import check_reproduce
    from b3_core.physics_surrogate import (
        CorePhysicsSurrogate,
        GeometrySpec,
        build_training_frame,
        fit_physics_surrogate,
    )
    from b3_core.report.build import build_report
    from b3_core.sweep.kerf_map import kerf_map
    from b3_core.sweep.parallel import map_workers

    def fake_grid(**_kwargs):
        return [
            {
                "kx": 0.0,
                "ky": 0.0,
                "cell_size": 0.0,
                "resin_vf": 0.02,
                "Ex": 1.0e6,
                "Ey": 1.0e6,
                "Ez": 2.0e6,
            }
        ]

    monkeypatch.setattr(
        "b3_core.sweep.curvature_grid.sweep_halo_curvature_grid", fake_grid
    )
    frame = build_training_frame(kx_values=[0.0], ky_values=[0.0], cell_sizes=[0.0])
    assert "rho_infused" in frame.columns
    fitted = fit_physics_surrogate(frame, geometry={"dx": 30.0})
    with pytest.raises(ValueError, match="ky"):
        fitted.lookup([0.0, 0.001], ky=[0.0], cell_size=0.0)
    loaded = CorePhysicsSurrogate.from_json(fitted.to_json())
    assert "Ex" in loaded.targets
    GeometrySpec.from_case(
        {
            "xgr": [{"offset": 1, "pitch": 10, "depth": 4, "width": 2, "mouth": "top"}],
            "core": {"E": 1.0e8, "nu": 0.3, "rho": 80},
        }
    )

    grid = tmp_path / "grid.yaml"
    grid.write_text(
        "rows:\n  - ok: true\n    kx: 0\n    ky: 0\n    Ex: 1.0e6\n    Ez: 2.0e6\n    Gxz: 4.0e5\n    rho_infused: 100\n",
        encoding="utf-8",
    )
    project = tmp_path / "project.yaml"
    project.write_text(f"title: A @ b\ngrid: {grid}\n", encoding="utf-8")
    written = build_report(project, tmp_path / "report")
    assert Path(written["report_md"]).is_file()

    case = uniaxial()
    mapped = kerf_map(case, 1.0e-4, solve=_stand_in, backend="numpy")
    assert any(cell["solved"] for cell in mapped["grid"])
    idle = check_convergence(case, solve=_stand_in, levels=(0, 0))
    assert idle["ok"] is False
    moved = check_convergence(
        case.input.model_copy(update={"madd": [-0.3, 0.3]}),
        solve=_stand_in,
        levels=(0, 1),
        rtol_pct=0.0,
    )
    assert "rel_pct" in moved
    values = {
        "points": [
            {
                "case": case.input.model_dump(mode="json"),
                "kx": 0.0,
                "ky": 0.0,
                "backend": "numpy",
                "solver_stamp": "other",
                "properties": {"Ex": 1.0},
            }
        ]
    }
    report = check_reproduce(values, solve=_stand_in)
    assert report["ok"] is False
    assert map_workers(_echo, [1, 2], 1) == [1, 2]


def _echo(value):
    return value


def test_sweep_stage_names_and_doctor_version(tmp_path, monkeypatch):
    from b3_core.doctor import _version
    from b3_core.pipeline import _first_capable, _needs
    from b3_core.sweep import run as sweep_run

    assert _version(object(), ("not-a-distribution",)) == "unknown"
    monkeypatch.setattr("b3_core.sweep.plots.run", lambda _ctx: 0)
    monkeypatch.setattr("b3_core.sweep.render_viz.run", lambda _ctx: 0)
    assert sweep_run("viz", tmp_path) == 0
    assert sweep_run("no-such-stage", tmp_path) == 1
    monkeypatch.setattr("b3_core.pipeline._PREFERENCE", ("missing-backend",))
    assert _first_capable(_needs(uniaxial().input), require_available=True) == "numpy"
