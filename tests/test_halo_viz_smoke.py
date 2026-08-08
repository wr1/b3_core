"""Smoke-exercise remaining halo figure entry points (coverage lift)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from b3_core.viz import halo

CASE = json.loads(
    Path(__file__)
    .resolve()
    .parents[1]
    .joinpath("examples/diab_gs30_scored.json")
    .read_text()
)
SHARP = {
    **CASE,
    "core": {k: v for k, v in CASE["core"].items() if k != "cell_size"},
}
# drop scoring so field inactive for sharp kerf-only visuals
SHARP = {k: v for k, v in SHARP.items() if k != "scoring"}
SHARP["core"] = {k: v for k, v in CASE["core"].items() if k != "cell_size"}


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def test_cross_section_strip():
    fig, ax = halo.plot_halo_cross_section_strip(CASE)
    assert ax is not None
    plt.close(fig)


def test_cross_section_strip_inactive_raises():
    with pytest.raises(ValueError, match="inactive"):
        halo.plot_halo_cross_section_strip(SHARP)


def test_sample_halo_plane():
    import numpy as np

    mesh, mat, field = halo._mesh_and_field(CASE)
    ux = np.linspace(float(mesh.bounds[0]) + 1e-3, float(mesh.bounds[1]) - 1e-3, 20)
    uz = np.linspace(float(mesh.bounds[4]) + 1e-3, float(mesh.bounds[5]) - 1e-3, 15)
    p, phase = halo.sample_halo_plane(
        mesh,
        mat,
        field,
        u_axis=0,
        v_axis=2,
        fixed_axis=1,
        coord=float(mesh.center[1]),
        u_vals=ux,
        v_vals=uz,
    )
    assert p.shape == (15, 20)
    assert phase.shape == p.shape


def test_sharp_vs_scored():
    fig, axes = halo.plot_halo_sharp_vs_scored(SHARP, CASE)
    assert axes is not None
    plt.close(fig)


def test_curvature_compose_and_wall_strip():
    fig = halo.plot_halo_curvature_compose(cell_size=0.8)
    assert fig is not None
    plt.close(fig)
    fig_s, _ = halo.plot_halo_curvature_wall_strip()
    plt.close(fig_s)


def test_follows_angled_walls():
    fig = halo.plot_halo_follows_angled_walls(kx=-0.008)
    assert fig is not None
    plt.close(fig)


def test_render_halo_3d_png(tmp_path):
    out = halo.render_halo_3d_png(CASE, tmp_path / "h3d.png")
    assert out.is_file() and out.stat().st_size > 0


def test_render_halo_figures_bundle(tmp_path):
    paths = halo.render_halo_figures(CASE, tmp_path, sharp_inp=SHARP, dpi=80)
    assert len(paths) >= 4
    assert all(p.is_file() for p in paths if p.suffix == ".png")


def test_render_halo_curvature_figures_no_param(tmp_path):
    paths = halo.render_halo_curvature_figures(tmp_path, run_parametric=False, dpi=80)
    assert len(paths) >= 2
    names = {p.name for p in paths}
    assert "halo_curvature_compose.png" in names
    assert "halo_curvature_wall_strip.png" in names


def test_stiffness_plots_from_tiny_grid():
    rows = halo.sweep_halo_curvature_grid(
        kx_values=[-0.004, 0.0, 0.004],
        cell_sizes=[0.0, 0.6],
    )
    fig = halo.plot_stiffness_moduli_vs_curvature(rows, cell_size=0.6)
    assert fig is not None
    plt.close(fig)
    fig2 = halo.plot_stiffness_vs_curvature_halo(rows)
    assert len(fig2.axes) >= 2
    plt.close(fig2)


def test_helpers_with_kx_and_default_case():
    base = halo._default_halo_curvature_case(cell_size=0.5)
    assert base["core"].get("cell_size") == 0.5 or "cell_size" in base["core"]
    opened = halo._with_kx(base, 0.01)
    assert opened["curvature"]["kx"] == 0.01
    cmap = halo._halo_band_cmap()
    assert cmap is not None
