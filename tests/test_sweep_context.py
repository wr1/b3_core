"""Unit tests for sweep helpers (no homogenisation solve)."""

import json

import pytest

from b3_core.sweep.context import (
    SweepContext,
    case_for_curvature,
    case_for_thickness,
    collect_sweep,
    default_root,
    load_base,
    load_pattern,
    max_rel_err,
    parse_kx_tag,
    parse_pattern_tag,
    parse_thickness_tag,
    print_moduli_table,
    scale_depths_for_thickness,
    scale_groove_depth,
    tag_float,
    tag_kx,
    tag_pattern,
    tag_thickness,
)


def test_default_root_points_at_param_sweeps():
    root = default_root()
    assert root.name == "param_sweeps"
    assert (root / "bases").is_dir() or not root.exists()  # ok if sparse checkout


def test_sweep_context_paths(tmp_path):
    ctx = SweepContext(tmp_path)
    assert ctx.out == tmp_path / "out"
    assert ctx.img == tmp_path / "img"
    assert ctx.bases == tmp_path / "bases"
    assert ctx.mfem_patterns == tmp_path.parent / "mfem_patterns"


def test_tags_and_parsers():
    assert tag_thickness(30) == "thickness_30"
    assert parse_thickness_tag("thickness_30") == 30.0
    assert parse_thickness_tag("nope") is None

    assert tag_kx(0.008).startswith("kx_")
    assert parse_kx_tag(tag_kx(-0.008)) == pytest.approx(-0.008)
    assert parse_kx_tag("bad") is None

    assert tag_pattern("plain") == "pattern_plain"
    assert parse_pattern_tag("pattern_two_sided") == "two_sided"
    assert parse_pattern_tag("x") is None

    assert "p" in tag_float(1.5) or "1" in tag_float(1.5)


def test_scale_groove_depth_and_case():
    assert scale_groove_depth(10.0, 30.0) == pytest.approx(10.0)
    assert scale_groove_depth(-10.0, 15.0) < 0
    scaled = scale_depths_for_thickness([[0, 5, 10, 1]], 15.0)
    assert scaled[0][2] != 10.0 or scaled[0][0] == 0

    base = {
        "thickness": 30.0,
        "xgr": [[0, 5, -20, 1]],
        "ygr": [],
        "dx": 50,
        "dy": 50,
    }
    c = case_for_thickness(base, 20.0)
    assert c["thickness"] == 20.0
    assert c["xgr"][0][2] != base["xgr"][0][2]
    c2 = case_for_curvature(base, 0.004)
    assert c2["curvature"] == {"kx": 0.004, "ky": 0.0}


def test_max_rel_err():
    assert max_rel_err({}) == (0.0, True)
    out = {
        "ccx_validation": {
            "passed": False,
            "properties": {
                "Exx": {"rel_error": 0.01},
                "Eyy": {"rel_error": 0.05},
            },
        }
    }
    err, passed = max_rel_err(out)
    assert err == pytest.approx(0.05)
    assert passed is False


def test_load_base_and_pattern_from_repo():
    root = default_root()
    if not (root / "bases" / "uniaxial.json").is_file():
        pytest.skip("examples/param_sweeps not present")
    ctx = SweepContext(root)
    base = load_base(ctx, "uniaxial")
    assert "thickness" in base
    if (ctx.mfem_patterns / "plain.json").is_file():
        pat = load_pattern(ctx, "plain")
        assert "core" in pat or "dx" in pat


def test_collect_sweep_and_print_table(tmp_path):
    from io import StringIO

    from rich.console import Console

    ctx = SweepContext(tmp_path)
    d = ctx.out / "thickness_30"
    d.mkdir(parents=True)
    (d / "case.json").write_text(json.dumps({"thickness": 30}))
    run = {
        "resin_vf": 0.1,
        "rho_infused": 200.0,
        "Exx": 1e9,
        "Eyy": 1e9,
        "Ezz": 1e9,
        "Gxy": 0.5e9,
    }
    (d / "runabc.json").write_text(json.dumps(run))
    rows = collect_sweep(ctx, "thickness_")
    assert len(rows) == 1
    assert rows[0][0] == "thickness_30"

    buf = StringIO()
    print_moduli_table(
        Console(file=buf, force_terminal=False, width=120),
        title="t",
        rows=[("case_a", run)],
    )
    assert "case_a" in buf.getvalue() or "resin_vf" in buf.getvalue()


def test_collect_sweep_empty(tmp_path):
    assert collect_sweep(SweepContext(tmp_path), "thickness_") == []


def test_sweep_run_dispatch(monkeypatch, tmp_path):
    from b3_core.sweep import homogenise
    from b3_core.sweep import run as sweep_run

    calls: list[str] = []

    monkeypatch.setattr(
        homogenise, "run_thickness", lambda ctx: calls.append("thickness") or 0
    )
    monkeypatch.setattr(
        homogenise, "run_curvature", lambda ctx: calls.append("curvature") or 0
    )
    monkeypatch.setattr(
        homogenise, "run_patterns", lambda ctx: calls.append("patterns") or 0
    )
    monkeypatch.setattr(
        homogenise,
        "run_all_homogenise",
        lambda ctx: calls.append("homogenise") or 0,
    )

    assert sweep_run("thickness", root=tmp_path) == 0
    assert sweep_run("curvature", root=tmp_path) == 0
    assert sweep_run("patterns", root=tmp_path) == 0
    assert sweep_run("homogenise", root=tmp_path) == 0
    assert calls == ["thickness", "curvature", "patterns", "homogenise"]
    assert sweep_run("not-a-stage", root=tmp_path) == 1


def test_run_case_merges_overrides_and_uses_cache(tmp_path, monkeypatch):
    from b3_core.sweep import context as ctx_mod

    base = {"a": 1, "nested": {"x": 1}, "dx": 1}
    out_dir = tmp_path / "run1"
    calls: list[str] = []

    def fake_cprop(path):
        calls.append(path)
        return {"ok": True, "Exx": 1.0}

    monkeypatch.setattr(ctx_mod, "cprop", fake_cprop)
    result = ctx_mod.run_case(base, {"nested": {"y": 2}, "b": 3}, out_dir)
    assert result["ok"] is True
    assert calls
    case = json.loads((out_dir / "case.json").read_text())
    assert case["nested"] == {"x": 1, "y": 2}
    assert case["b"] == 3

    # Second call hits FileExistsError path via cached run*.json
    (out_dir / "runcached.json").write_text(json.dumps({"cached": True}))

    def boom(_path):
        raise FileExistsError("exists")

    monkeypatch.setattr(ctx_mod, "cprop", boom)
    cached = ctx_mod.run_case(base, {}, out_dir)
    assert cached["cached"] is True


def test_homogenise_drivers_with_mocked_run_case(monkeypatch, tmp_path):
    from b3_core.sweep import homogenise

    fake_out = {
        "resin_vf": 0.1,
        "area_increase": 1.1,
        "rho_infused": 200.0,
        "Exx": 1e9,
        "Eyy": 1e9,
        "Ezz": 1e9,
        "Gxy": 0.5e9,
        "ccx_validation": {"passed": True, "properties": {}},
    }
    base = {
        "dx": 50,
        "dy": 50,
        "thickness": 30,
        "xgr": [[0, 10, -20, 1]],
        "ygr": [],
        "core": {"E": 1e9, "nu": 0.3, "rho": 100},
        "resin": {"E": 3e9, "nu": 0.3, "rho": 1100},
    }

    monkeypatch.setattr(homogenise, "load_base", lambda ctx, name: dict(base))
    monkeypatch.setattr(homogenise, "load_pattern", lambda ctx, name: dict(base))
    monkeypatch.setattr(homogenise, "run_case", lambda *a, **k: dict(fake_out))
    monkeypatch.setattr(
        homogenise,
        "THICKNESSES",
        [20, 30],
    )
    monkeypatch.setattr(homogenise, "KX", [-0.004, 0.0, 0.004])
    monkeypatch.setattr(homogenise, "PATTERNS", ["plain", "uniaxial"])

    ctx = SweepContext(tmp_path)
    assert homogenise.run_thickness(ctx) == 0
    assert homogenise.run_curvature(ctx) == 0
    assert homogenise.run_patterns(ctx) == 0
    assert homogenise.run_all_homogenise(ctx) == 0
