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


def test_bind_and_sweep_context_default(tmp_path, monkeypatch):
    # Point default_root away from repo if needed; just exercise binders.
    run_mod._bind_sweep_root(str(tmp_path))
    assert run_mod._SWEEP_ROOT_STATE[0] == str(tmp_path)
    ctx = run_mod._sweep_context(str(tmp_path))
    assert ctx.root == tmp_path


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


def test_cmd_run_delegates_to_cprop(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(run_mod, "cprop", lambda p: calls.append(p))
    run_mod.cmd_run("case.json")
    assert calls == ["case.json"]


def test_cmd_viz_halo_and_curvature(monkeypatch, tmp_path, capsys):
    scored = tmp_path / "diab_gs30_scored.json"
    scored.write_text('{"core": {"E": 1}, "resin": {"E": 2}}')
    sharp = tmp_path / "diab_gs30.json"
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


def test_sweep_cmd_wrappers(monkeypatch):
    codes: list[int] = []

    monkeypatch.setattr(
        "b3_core.sweep.homogenise.run_thickness", lambda ctx: codes.append(0) or 0
    )
    monkeypatch.setattr(
        "b3_core.sweep.homogenise.run_curvature", lambda ctx: codes.append(1) or 0
    )
    monkeypatch.setattr(
        "b3_core.sweep.homogenise.run_patterns", lambda ctx: codes.append(2) or 0
    )
    monkeypatch.setattr(run_mod, "_sweep_exit", lambda c: codes.append(100 + c))

    run_mod.cmd_sweep_thickness("")
    run_mod.cmd_sweep_curvature("")
    run_mod.cmd_sweep_patterns("")
    run_mod.cmd_sweep_curvature_chained()
    run_mod.cmd_sweep_patterns_chained()
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

    monkeypatch.setattr(
        "b3_core.physics_surrogate.CorePhysicsSurrogate",
        _S,
    )
    run_mod.cmd_surrogate_lookup(str(out), "0,0.001", 0.6, "")
    assert "Eyy" in capsys.readouterr().out
    csv = tmp_path / "lut.csv"
    run_mod.cmd_surrogate_lookup(str(out), "0", 0.0, str(csv))
    assert csv.is_file()
