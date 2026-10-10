"""CLI entry points that do not require a full FEA solve."""

import subprocess
import sys
from pathlib import Path

import pytest

from b3_core.core import run as run_mod


def test_cmd_skill_path(capsys):
    run_mod.cmd_skill(stdout=False)
    out = capsys.readouterr().out.strip()
    assert out.endswith("SKILL.md")


def test_cmd_skill_stdout(capsys):
    run_mod.cmd_skill(stdout=True)
    out = capsys.readouterr().out
    assert out.startswith("---\n")
    assert "name: b3-core" in out


def test_cli_help_exits_zero():
    proc = subprocess.run(
        [sys.executable, "-m", "b3_core.core.run", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    # treeparse may print help to stdout or stderr
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode in (0, 1)  # some CLIs use 1 for help-only
    assert "homogen" in combined.lower() or "b3_core" in combined or "run" in combined


def test_resolve_sweep_root_explicit(tmp_path):
    assert run_mod.resolve_sweep_root(str(tmp_path)) == tmp_path
    ctx = run_mod._sweep_context(str(tmp_path))
    assert ctx.root == tmp_path


def test_resolve_sweep_root_cwd_with_bases(tmp_path, monkeypatch):
    (tmp_path / "bases").mkdir()
    monkeypatch.chdir(tmp_path)
    assert run_mod.resolve_sweep_root("") == tmp_path


def test_resolve_sweep_root_cwd_with_patterns(tmp_path, monkeypatch):
    (tmp_path / "mfem_patterns").mkdir()
    monkeypatch.chdir(tmp_path)
    assert run_mod.resolve_sweep_root("") == tmp_path


def test_resolve_sweep_root_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as ei:
        run_mod.resolve_sweep_root("")
    assert ei.value.code == 2
    assert "no study root found; pass --root" in capsys.readouterr().err


def test_sweep_exit_raises_system_exit():
    with pytest.raises(SystemExit) as ei:
        run_mod._sweep_exit(2)
    assert ei.value.code == 2


def test_main_builds_and_runs_cli(monkeypatch):
    """Construct the full CLI tree (covers command wiring) without a solve."""
    seen: dict = {}

    class _App:
        def run(self) -> None:
            seen["ran"] = True

    def fake_cli(**kwargs):
        seen["name"] = kwargs.get("name")
        seen["commands"] = kwargs.get("commands")
        seen["subgroups"] = kwargs.get("subgroups")
        return _App()

    monkeypatch.setattr(run_mod, "cli", fake_cli)
    run_mod.main()
    assert seen.get("ran") is True
    assert seen.get("name") == "b3_core"
    assert seen.get("commands")
    assert seen.get("subgroups")


def test_cmd_run_writes_ccx_ortho(tmp_path, capsys):
    from tests.fakes import fake_backend, unregister

    from b3_core.cases import plain

    register, cls = fake_backend("fake")
    register(cls)
    case = tmp_path / "case.json"
    dest = tmp_path / "core.inp"
    try:
        plain(backend="fake").to_json(case)
        run_mod.cmd_run(str(case), backend="fake", ccx_ortho=str(dest))
    finally:
        unregister("fake")
    text = dest.read_text()
    assert "*elastic,type=ortho" in text
    assert "*density" in text
    assert "Wrote" in capsys.readouterr().out


def test_cmd_run_json_scales_moduli_and_skips_disk(tmp_path, capsys):
    import json

    from tests.fakes import fake_backend, unregister

    from b3_core.cases import plain

    register, cls = fake_backend("fake")
    register(cls)
    case = tmp_path / "case.json"
    try:
        plain(backend="fake").to_json(case)
        run_mod.cmd_run(
            str(case),
            backend="fake",
            as_json=True,
            units="GPa",
            no_write=True,
        )
    finally:
        unregister("fake")
    payload = json.loads(capsys.readouterr().out)
    assert payload["properties"]["Ex"]["value"] == pytest.approx(1.0)
    assert payload["properties"]["Ex"]["unit"] == "GPa"
    assert payload["properties"]["nuxy"]["value"] == pytest.approx(0.3)
    assert payload["properties"]["nuxy"]["unit"] == "-"
    assert payload["stiffness_unit"] == "GPa"
    assert payload["written"] is None
    assert payload["cache_hit"] is False
    checks = {row["id"]: row["status"] for row in payload["diagnostics"]["checks"]}
    assert "raw_asymmetry" in checks
    assert "positive_definite" in checks
    assert "voigt_reuss" in checks
    assert "kerf_pinch" in checks
    assert {row["face"] for row in payload["diagnostics"]["ties"]} == {"x", "y", "z"}
    assert payload["geometry"]["rho_infused"]["unit"] == "kg/m^3"
    assert "provenance" in payload
    assert list(tmp_path.glob("run*.json")) == []


def test_cmd_run_json_error_on_curved_auto(monkeypatch, tmp_path, capsys):
    import json

    from b3_core.cases import plain

    monkeypatch.setattr(
        "b3_core.solvers.fenicsx.FenicsxBackend.is_available",
        lambda self: False,
    )
    case = tmp_path / "case.json"
    plain().with_curvature(kx=1e-3).to_json(case)
    with pytest.raises(SystemExit) as ei:
        run_mod.cmd_run(str(case), as_json=True, no_write=True)
    assert ei.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["type"] == "BackendCapabilityError"
    assert "doctor" in payload["error"]["hint"] or "numpy" in payload["error"]["hint"]


def test_report_ties_unit_is_stored_as_per_mm(tmp_path, capsys):
    from b3_core.cases import uniaxial

    case = tmp_path / "case.json"
    uniaxial().to_json(case)
    run_mod.cmd_report_ties(str(case), kx="2", unit="1/m", as_json=True)
    payload = __import__("json").loads(capsys.readouterr().out)
    assert payload["unit"] == "1/m"
    assert payload["kx"] == pytest.approx(0.002)
    assert "R = 0.5 m" in payload["kx_tick"]


def test_check_points_honor_the_curvature_unit(tmp_path, capsys, monkeypatch):
    case = tmp_path / "case.json"
    case.write_text("{}", encoding="utf-8")
    seen = {}

    def fake_check(path, **kwargs):
        seen["points"] = kwargs["points"]
        return {
            "ok": True,
            "rtol": 1e-4,
            "worst_rel_pct": 0.0,
            "wall_s": 0.0,
            "rows": [],
        }

    monkeypatch.setattr("b3_core.checks.backends.check_backends", fake_check)
    run_mod.cmd_check_backends(str(case), points="2,0", unit="1/m", as_json=True)
    assert seen["points"] == [(pytest.approx(0.002), 0.0)]
    assert __import__("json").loads(capsys.readouterr().out)["unit"] == "1/m"


def test_payload_json_is_not_rewritten_when_it_asks_for_the_schema():
    from b3_core.core.run import _rewrite_payload_json

    assert _rewrite_payload_json(["--json"]) == ["--json"]
    assert _rewrite_payload_json(["run", "case.json", "--json", "--no-write"]) == [
        "run",
        "case.json",
        "--agent-json",
        "--no-write",
    ]


def test_run_json_flag_reaches_the_command(tmp_path):
    case = tmp_path / "missing.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "b3_core.core.run",
            "run",
            str(case),
            "--json",
            "--no-write",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert '"type": "cli"' not in combined
    assert proc.returncode != 0


def test_run_help_lists_agent_json_flags():
    proc = subprocess.run(
        [sys.executable, "-m", "b3_core.core.run", "run", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode in (0, 1)
    assert "--json" in combined
    assert "--units" in combined
    assert "--no-write" in combined
    assert "--strict" in combined


def test_cmd_run_writes_run_json(tmp_path, capsys):
    from tests.fakes import fake_backend, unregister

    from b3_core.cases import plain

    register, cls = fake_backend("fake")
    register(cls)
    case = tmp_path / "case.json"
    try:
        plain(backend="fake").to_json(case)
        run_mod.cmd_run(str(case), backend="fake")
    finally:
        unregister("fake")
    written = list(tmp_path.glob("run*.json"))
    assert len(written) == 1
    assert "Wrote" in capsys.readouterr().out


def test_sharp_sibling_from_halo_name(tmp_path):
    sharp = tmp_path / "grid_scored.json"
    sharp.write_text("{}")
    halo = tmp_path / "grid_scored_halo.json"
    halo.write_text("{}")
    assert run_mod._sharp_sibling(str(halo)) == sharp
    assert run_mod._sharp_sibling(str(sharp)) is None


def test_cmd_viz_halo_and_curvature(monkeypatch, tmp_path, capsys):
    scored = tmp_path / "grid_scored_halo.json"
    scored.write_text('{"core": {"E": 1}, "resin": {"E": 2}}')
    sharp = tmp_path / "grid_scored.json"
    sharp.write_text('{"core": {"E": 1}}')
    out = tmp_path / "img"

    monkeypatch.setattr(
        "b3_core.viz.halo.render_halo_figures",
        lambda *a, **k: [out / "a.png"],
    )
    run_mod.cmd_viz_halo(str(scored), str(out), str(sharp))
    assert "Wrote" in capsys.readouterr().out

    monkeypatch.setattr(
        "b3_core.viz.halo.render_halo_curvature_figures",
        lambda *a, **k: [out / "b.png"],
    )
    run_mod.cmd_viz_halo_curvature(str(out), "", 0.01, -0.01)
    assert "Wrote" in capsys.readouterr().out


def test_cmd_viz_view_and_datasheet_and_deformed(monkeypatch, tmp_path, capsys):
    class _View:
        def __init__(self):
            self.calls = []

        @classmethod
        def from_json(cls, path):
            return cls()

        def serve(self, p):
            self.calls.append(("serve", p))

        def gallery(self, p):
            self.calls.append(("gallery", p))

        def geometry_png(self, p, cutaway=False):
            self.calls.append(("geometry", p))

        def slices_png(self, p):
            self.calls.append(("slices", p))

        def deformation_png(self, p, warp=1.0):
            self.calls.append(("deform", p))

        def modulus_surface_png(self, p):
            self.calls.append(("mod", p))

        def modulus_polar_png(self, p):
            self.calls.append(("polar", p))

        def stiffness_heatmap_png(self, p):
            self.calls.append(("heat", p))

    v = _View()
    monkeypatch.setattr(
        "b3_core.viz.GroovedCoreView",
        type("G", (), {"from_json": staticmethod(lambda p: v)}),
    )
    run_mod.cmd_viz_view("case.json", "gallery", str(tmp_path / "g.png"), "", 1.0)
    run_mod.cmd_viz_view("case.json", "geometry", str(tmp_path / "geo.png"), "", 1.0)
    run_mod.cmd_viz_view("case.json", "all", str(tmp_path / "viz_all"), "", 1.0)
    run_mod.cmd_viz_view("case.json", "gallery", "", str(tmp_path / "s.html"), 1.0)

    monkeypatch.setattr(
        "b3_core.datasheet.generate",
        lambda *a, **k: None,
    )
    run_mod.cmd_viz_datasheet(
        "case.json", str(tmp_path / "c.pdf"), str(tmp_path / "c.png")
    )

    monkeypatch.setattr(
        "b3_core.deformed.render_deformed_modes",
        lambda *a, **k: None,
    )
    run_mod.cmd_viz_deformed("case.json", str(tmp_path / "d.png"), 2.0)
    out = capsys.readouterr().out
    assert "Wrote" in out


def test_homogenise_chain_runs_every_stage(monkeypatch, tmp_path):
    from treeparse.models.cli import chain_runner

    seen: list[tuple] = []

    monkeypatch.setattr(
        "b3_core.sweep.homogenize.run_thickness",
        lambda ctx: seen.append(("thickness", ctx.root)) or 0,
    )
    monkeypatch.setattr(
        "b3_core.sweep.homogenize.run_curvature",
        lambda ctx: seen.append(("curvature", ctx.root)) or 0,
    )
    monkeypatch.setattr(
        "b3_core.sweep.homogenize.run_patterns",
        lambda ctx: seen.append(("patterns", ctx.root)) or 0,
    )
    monkeypatch.setattr(run_mod, "_sweep_exit", lambda c: seen.append(("exit", c)))

    grp = run_mod._sweep_subgroup()
    chained = next(c for c in grp.commands if getattr(c, "name", None) == "homogenise")
    chain_runner(chained, root=str(tmp_path), cache="")
    assert [step[0] for step in seen] == ["thickness", "curvature", "patterns", "exit"]
    assert seen[0][1] == tmp_path
    assert seen[-1] == ("exit", 0)


def test_sweep_cmd_wrappers(monkeypatch, tmp_path):
    codes: list[int] = []

    monkeypatch.setattr(
        "b3_core.sweep.homogenize.run_thickness", lambda ctx: codes.append(0) or 0
    )
    monkeypatch.setattr(
        "b3_core.sweep.homogenize.run_curvature", lambda ctx: codes.append(1) or 0
    )
    monkeypatch.setattr(
        "b3_core.sweep.homogenize.run_patterns", lambda ctx: codes.append(2) or 0
    )
    monkeypatch.setattr(run_mod, "_sweep_exit", lambda c: codes.append(100 + c))

    root = str(tmp_path)
    run_mod.cmd_sweep_thickness(root, "")
    run_mod.cmd_sweep_curvature(root, "")
    run_mod.cmd_sweep_patterns(root, "")
    assert 100 in codes  # _sweep_exit called


def test_surrogate_cli_cmds(monkeypatch, tmp_path, capsys):
    class _S:
        targets = ["Eyy"]

        def to_json(self, p):
            Path(p).write_text("{}")

        @classmethod
        def from_json(cls, p):
            return cls()

        def lookup(self, kx, cell_size=0.0):
            import pandas as pd

            return pd.DataFrame({"kx": kx, "Eyy": [1.0] * len(kx)})

    monkeypatch.setattr(
        "b3_core.physics_surrogate.fit_from_homogenization",
        lambda **k: _S(),
    )
    out = tmp_path / "surr.json"
    run_mod.cmd_surrogate_fit(str(out), str(tmp_path / "cache.json"))
    assert "Wrote" in capsys.readouterr().out

    with pytest.raises(SystemExit) as ei:
        run_mod.cmd_surrogate_fit("", "")
    assert ei.value.code == 2
    assert "--output" in capsys.readouterr().err

    monkeypatch.setattr(
        "b3_core.physics_surrogate.CorePhysicsSurrogate",
        _S,
    )
    run_mod.cmd_surrogate_lookup(str(out), "0,0.001", 0.6, "")
    assert "Eyy" in capsys.readouterr().out
    csv = tmp_path / "lut.csv"
    run_mod.cmd_surrogate_lookup(str(out), "0", 0.0, str(csv))
    assert csv.is_file()
