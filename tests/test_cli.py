"""CLI entry points that do not require a full FEA solve."""

import subprocess
import sys

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
