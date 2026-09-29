"""Resin-halo figures. Homogenisation grids live in ``b3_core.sweep.curvature_grid``."""

from __future__ import annotations

from b3_core.viz.halo.bundle import render_halo_curvature_figures, render_halo_figures
from b3_core.viz.halo.curvature_figs import (
    plot_halo_curvature_compose,
    plot_halo_curvature_wall_strip,
    plot_halo_follows_angled_walls,
    plot_stiffness_moduli_vs_curvature,
    plot_stiffness_vs_curvature_halo,
)
from b3_core.viz.halo.probability import (
    effective_modulus_ratio,
    plot_halo_degradation,
    resin_probability_vs_distance,
)
from b3_core.viz.halo.render3d import render_halo_3d_png
from b3_core.viz.halo.sections import (
    mesh_and_field,
    plot_halo_cross_section_strip,
    plot_halo_intuitive_board,
    plot_halo_sharp_vs_scored,
    plot_halo_side_cut,
    sample_halo_plane,
)

__all__ = [
    "effective_modulus_ratio",
    "homogenize_halo_curvature",
    "mesh_and_field",
    "parametric_base_case",
    "plot_halo_cross_section_strip",
    "plot_halo_curvature_compose",
    "plot_halo_curvature_wall_strip",
    "plot_halo_degradation",
    "plot_halo_follows_angled_walls",
    "plot_halo_intuitive_board",
    "plot_halo_sharp_vs_scored",
    "plot_halo_side_cut",
    "render_halo_3d_png",
    "render_halo_curvature_figures",
    "render_halo_figures",
    "resin_probability_vs_distance",
    "sample_halo_plane",
    "sweep_halo_curvature_grid",
    "plot_stiffness_moduli_vs_curvature",
    "plot_stiffness_vs_curvature_halo",
]


def __getattr__(name: str):
    if name in {
        "homogenize_halo_curvature",
        "parametric_base_case",
        "sweep_halo_curvature_grid",
    }:
        from b3_core.sweep import curvature_grid

        return getattr(curvature_grid, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
