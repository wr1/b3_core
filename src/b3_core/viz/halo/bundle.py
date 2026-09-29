"""Write the halo figure bundles to a directory."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from b3_core.viz.halo.curvature_figs import (
    CURVED_PANEL_KX_CLOSED,
    CURVED_PANEL_KX_OPEN,
    plot_halo_curvature_compose,
    plot_halo_curvature_wall_strip,
    plot_stiffness_moduli_vs_curvature,
    plot_stiffness_vs_curvature_halo,
)
from b3_core.viz.halo.probability import plot_halo_degradation
from b3_core.viz.halo.render3d import render_halo_3d_png
from b3_core.viz.halo.sections import (
    plot_halo_cross_section_strip,
    plot_halo_intuitive_board,
    plot_halo_sharp_vs_scored,
    plot_halo_side_cut,
)


def _sweep_grid(*args, **kwargs):
    from b3_core.sweep.curvature_grid import sweep_halo_curvature_grid

    return sweep_halo_curvature_grid(*args, **kwargs)


def render_halo_curvature_figures(
    out_dir: str | Path,
    *,
    base_inp: dict | None = None,
    kx_open: float = CURVED_PANEL_KX_OPEN,
    kx_closed: float = CURVED_PANEL_KX_CLOSED,
    dpi: int = 200,
    run_parametric: bool = True,
) -> list[Path]:
    """Write halo on curved_panel open/closed configs to *out_dir*."""
    import matplotlib

    matplotlib.use("Agg")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    fig = plot_halo_curvature_compose(base_inp, kx_open=kx_open, kx_closed=kx_closed)
    p = out_dir / "halo_curvature_compose.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    fig_s, _ = plot_halo_curvature_wall_strip(
        base_inp, kx_open=kx_open, kx_closed=kx_closed
    )
    p_s = out_dir / "halo_curvature_wall_strip.png"
    fig_s.savefig(p_s, dpi=dpi, bbox_inches="tight")
    plt.close(fig_s)
    written.append(p_s)

    if run_parametric:
        cache = out_dir / "halo_curvature_param_grid.json"
        rows = _sweep_grid(cache_path=cache)
        fig_p = plot_stiffness_vs_curvature_halo(rows)
        p_p = out_dir / "halo_curvature_stiffness.png"
        fig_p.savefig(p_p, dpi=dpi, bbox_inches="tight")
        plt.close(fig_p)
        written.append(p_p)

        fig_m = plot_stiffness_moduli_vs_curvature(rows, cell_size=0.6)
        p_m = out_dir / "halo_curvature_moduli_vs_kx.png"
        fig_m.savefig(p_m, dpi=dpi, bbox_inches="tight")
        plt.close(fig_m)
        written.append(p_m)
        written.append(cache)

    return written


def render_halo_figures(
    scored_inp: dict,
    out_dir: str | Path,
    *,
    sharp_inp: dict | None = None,
    dpi: int = 200,
) -> list[Path]:
    """Write the full resin-halo figure bundle to *out_dir*."""
    import matplotlib

    matplotlib.use("Agg")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    core = scored_inp.get("core") or {}
    resin = scored_inp.get("resin") or {}
    e3_foam = float(core.get("E3") or core.get("E") or 70e6)
    e_resin = float(resin.get("E", 3e9))

    written: list[Path] = []

    fig, _ = plot_halo_degradation(
        [0.3, 0.6, {"mean": 0.25, "std": 0.08, "dist": "lognormal"}],
        labels=[
            "uniform cell_size = 0.3 mm",
            "uniform cell_size = 0.6 mm",
            "lognormal mean=0.25 mm, σ=0.08 mm",
        ],
        e_foam=e3_foam,
        e_resin=e_resin,
        highlight_index=1,
        modulus_label="E₃",
        title="Grid-scored foam: resin halo degradation from cut surface",
    )
    p = out_dir / "halo_degradation.png"
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    for name, plot_fn in (
        (
            "halo_strip_grid_scored.png",
            lambda: plot_halo_cross_section_strip(scored_inp),
        ),
        ("halo_side_cut.png", lambda: plot_halo_side_cut(scored_inp)),
    ):
        fig_i, _ = plot_fn()
        path = out_dir / name
        fig_i.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig_i)
        written.append(path)

    fig_board = plot_halo_intuitive_board(scored_inp)
    p_board = out_dir / "halo_intuitive_board.png"
    fig_board.savefig(p_board, dpi=dpi, bbox_inches="tight")
    plt.close(fig_board)
    written.append(p_board)

    written.append(render_halo_3d_png(scored_inp, out_dir / "halo_3d.png"))

    if sharp_inp is not None:
        fig_cmp, _ = plot_halo_sharp_vs_scored(sharp_inp, scored_inp)
        p_cmp = out_dir / "halo_sharp_vs_scored.png"
        fig_cmp.savefig(p_cmp, dpi=dpi, bbox_inches="tight")
        plt.close(fig_cmp)
        written.append(p_cmp)

    return written
