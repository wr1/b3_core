"""Halo cross-sections sampled on the grooved mesh."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from b3_core.viz.halo.probability import (
    effective_modulus_ratio,
    resin_probability_vs_distance,
)
from b3_core.viz.theme import DEFAULT_THEME, CoreTheme


def plot_halo_cross_section_strip(
    inp: dict,
    *,
    span_mm: float | None = None,
    n: int = 400,
    theme: CoreTheme = DEFAULT_THEME,
    figsize: tuple[float, float] = (9.0, 2.2),
) -> tuple[plt.Figure, plt.Axes]:
    """1D strip: ``P(resin)`` normal to a groove wall at mid groove depth."""
    from b3_core.core.scoring import ScoreField

    field = ScoreField(inp)
    if not field.active:
        msg = "ScoreField inactive — set core.cell_size and grooves in inp"
        raise ValueError(msg)

    groove = field.grooves[0]
    g_axis, c0, hw, _slope, depth = groove
    z = depth * 0.5 if depth > 0 else field.thickness + depth * 0.5
    if span_mm is None:
        span_mm = field.reach * 1.15

    # Sample outward from the +x wall of the first groove (in-plane normal).
    t = np.linspace(0.0, span_mm, n)
    pts = np.zeros((n, 3))
    pts[:, 2] = z
    if g_axis == 0:
        pts[:, 0] = c0 + hw + t
        pts[:, 1] = float(inp["dy"]) * 0.5
        xlabel = "Distance from groove wall [mm]"
    else:
        pts[:, 1] = c0 + hw + t
        pts[:, 0] = float(inp["dx"]) * 0.5
        xlabel = "Distance from groove wall [mm]"

    p = field.resin_probability(pts)

    with plt.rc_context(theme.publication_rcparams()):
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        ax.fill_between(t, 0, p, color=theme.resin_color, alpha=0.35)
        ax.plot(t, p, color=theme.resin_color, lw=2.0)
        ax.axvline(
            field.reach,
            color="#888888",
            ls=":",
            lw=1.0,
            label=f"cell_size reach ≈ {field.reach:.2g} mm",
        )
        ax.set_xlim(0.0, span_mm)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("P(resin)")
        ax.set_title("Halo strip — normal to groove wall (mid depth)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

    return fig, ax


def mesh_and_field(inp: dict):
    from b3_core.loaders import normalize_case
    from b3_core.pipeline import prepare
    from b3_core.viz import geometry

    model, _workdir = normalize_case(inp)
    prep = prepare(model)
    field = prep.score_field
    if field is None or not getattr(field, "active", False):
        msg = "ScoreField inactive — set core.cell_size and grooves in inp"
        raise ValueError(msg)
    mesh = prep.mesh
    mat = geometry.cell_material(mesh)
    return mesh, mat, field


def sample_halo_plane(
    mesh,
    mat: np.ndarray,
    field,
    u_axis: int,
    v_axis: int,
    fixed_axis: int,
    coord: float,
    u_vals: np.ndarray,
    v_vals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``P(resin)`` and phase (0 foam, 1 neat resin) on a plane grid."""
    from b3_core.viz.theme import RESIN

    uu, vv = np.meshgrid(u_vals, v_vals)
    pts = np.zeros((uu.size, 3))
    pts[:, u_axis] = uu.ravel()
    pts[:, v_axis] = vv.ravel()
    pts[:, fixed_axis] = coord
    cids = np.asarray(mesh.find_containing_cell(pts.astype(float)))

    p = np.full(uu.size, np.nan)
    phase = np.full(uu.size, np.nan)
    inside = cids >= 0
    if not inside.any():
        return p.reshape(uu.shape), phase.reshape(uu.shape)

    p_field = field.resin_probability(pts[inside])
    for j, flat in enumerate(np.flatnonzero(inside)):
        cid = int(cids[flat])
        if mat[cid] == RESIN:
            p[flat] = 1.0
            phase[flat] = 1.0
        else:
            p[flat] = p_field[j]
            phase[flat] = 0.0
    return p.reshape(uu.shape), phase.reshape(uu.shape)


def plot_halo_side_cut(
    inp: dict,
    *,
    theme: CoreTheme = DEFAULT_THEME,
    figsize: tuple[float, float] = (8.5, 4.8),
    px: int = 320,
) -> tuple[plt.Figure, plt.Axes]:
    """Side cut (x–z) coloured by halo probability with zone callouts.

    Neat machined kerf cells are drawn solid resin; adjacent foam shows the
    graded ``P(resin)`` field (opened cells along the saw cut).
    """
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch

    mesh, mat, field = mesh_and_field(inp)
    xv, zv = np.unique(mesh.x), np.unique(mesh.z)
    x0, x1, z0, z1 = xv[0], xv[-1], zv[0], zv[-1]
    yc = float(inp["dy"]) * 0.5

    Lx, Lz = x1 - x0, z1 - z0
    ref = max(Lx, Lz, float(inp["dx"]), float(inp["dy"]))
    nx = int(np.clip(px * Lx / ref, 80, px))
    nz = int(np.clip(px * Lz / ref, 80, px))
    ux = np.linspace(x0 + 1e-4, x1 - 1e-4, nx)
    uz = np.linspace(z0 + 1e-4, z1 - 1e-4, nz)

    p_grid, phase = sample_halo_plane(mesh, mat, field, 0, 2, 1, yc, ux, uz)
    resin_rgb = np.array(to_rgb(theme.halo_resin_color()))
    halo_cmap = theme.halo_cmap()
    rgba = np.zeros((*p_grid.shape, 4))
    valid = ~np.isnan(p_grid)
    foam = valid & (phase < 0.5)
    neat = valid & (phase >= 0.5)
    rgba[foam] = halo_cmap(np.clip(p_grid[foam], 0.0, 1.0))
    rgba[neat] = (*resin_rgb, 1.0)
    rgba[~valid] = (1.0, 1.0, 1.0, 0.0)

    with plt.rc_context(theme.publication_rcparams()):
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        ax.imshow(
            rgba,
            origin="lower",
            extent=[x0, x1, z0, z1],
            aspect="equal",
            interpolation="nearest",
        )
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("z [mm]")
        ax.set_title(
            f"Resin halo — side cut at y = {yc:.0f} mm  "
            f"(cell_size reach ≈ {field.reach:.2g} mm)"
        )

        if field.grooves:
            g_axis, c0, hw, _slope, depth = field.grooves[0]
            if g_axis == 0 and depth > 0:
                z_mid = depth * 0.5
                ax.annotate(
                    "neat kerf\n(machined slit)",
                    xy=(c0, z_mid),
                    xytext=(c0 + field.reach * 2.2, z_mid + Lz * 0.22),
                    fontsize=8,
                    arrowprops={
                        "arrowstyle": "->",
                        "color": theme.edge_color,
                        "lw": 1.0,
                    },
                )
                ax.annotate(
                    "halo\n(opened foam cells)",
                    xy=(c0 + hw + field.reach * 0.45, z_mid),
                    xytext=(c0 + hw + field.reach * 2.6, z_mid - Lz * 0.18),
                    fontsize=8,
                    color=theme.halo_resin_color(),
                    arrowprops={
                        "arrowstyle": "->",
                        "color": theme.halo_resin_color(),
                        "lw": 1.0,
                    },
                )
                ax.annotate(
                    "intact foam",
                    xy=(x1 - Lx * 0.12, z_mid),
                    xytext=(x1 - Lx * 0.38, z_mid + Lz * 0.28),
                    fontsize=8,
                    arrowprops={"arrowstyle": "->", "color": "#2166ac", "lw": 1.0},
                )

        sm = plt.cm.ScalarMappable(cmap=halo_cmap, norm=plt.Normalize(0.0, 1.0))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("P(resin)  blue=foam → red=resin")
        face_on = field.surfaces.get("face", {}).get("enabled", False)
        legend_handles = [
            Patch(facecolor=theme.halo_resin_color(), label="neat resin (kerf volume)"),
            Patch(facecolor=halo_cmap(0.75), label="saw-cut halo (opened cells)"),
            Patch(facecolor=halo_cmap(0.0), label="intact foam (P → 0)"),
        ]
        if face_on:
            legend_handles.insert(
                2,
                Patch(
                    facecolor=halo_cmap(0.35), label="face halo (closed cells, thinner)"
                ),
            )
        ax.legend(handles=legend_handles, loc="upper right", fontsize=7, framealpha=0.9)

    return fig, ax


def plot_halo_sharp_vs_scored(
    sharp_inp: dict,
    scored_inp: dict,
    *,
    theme: CoreTheme = DEFAULT_THEME,
    figsize: tuple[float, float] = (10.5, 4.5),
) -> tuple[plt.Figure, np.ndarray]:
    """Side-by-side: sharp kerf only vs scored core with resin halo."""
    from b3_core.pipeline import mesh_for
    from b3_core.viz import geometry

    def _phase_cut(inp: dict, ax, title: str):
        mesh = mesh_for(inp)
        mat = geometry.cell_material(mesh)
        yc = float(inp["dy"]) * 0.5
        xv, zv = np.unique(mesh.x), np.unique(mesh.z)
        ux = np.linspace(xv[0] + 1e-4, xv[-1] - 1e-4, 200)
        uz = np.linspace(zv[0] + 1e-4, zv[-1] - 1e-4, 120)
        uu, vv = np.meshgrid(ux, uz)
        pts = np.zeros((uu.size, 3))
        pts[:, 0] = uu.ravel()
        pts[:, 2] = vv.ravel()
        pts[:, 1] = yc
        cids = np.asarray(mesh.find_containing_cell(pts.astype(float)))
        grid = np.full(uu.shape, np.nan)
        inside = cids >= 0
        grid.ravel()[inside] = mat[cids[inside]]
        cmap, norm = theme.phase_cmap()
        ax.imshow(
            grid,
            origin="lower",
            extent=[xv[0], xv[-1], zv[0], zv[-1]],
            cmap=cmap,
            norm=norm,
            aspect="equal",
            interpolation="nearest",
        )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("z [mm]")

    with plt.rc_context(theme.publication_rcparams()):
        fig, axes = plt.subplots(1, 2, figsize=figsize, layout="constrained")
        _phase_cut(sharp_inp, axes[0], "Sharp kerf only (no halo)")
        fig_sc, ax_sc = plot_halo_side_cut(scored_inp, theme=theme, figsize=(5.0, 4.0))
        fig_sc.canvas.draw()
        buf = np.asarray(fig_sc.canvas.buffer_rgba())[:, :, :3]
        plt.close(fig_sc)
        axes[1].imshow(buf, aspect="auto")
        axes[1].axis("off")
        axes[1].set_title("With stochastic halo (opened foam cells)", fontsize=9)
        fig.suptitle(
            "Nominal 1 mm kerf vs effective resin region after grid-scoring",
            fontsize=10,
        )
    return fig, axes


def plot_halo_intuitive_board(
    inp: dict,
    *,
    theme: CoreTheme = DEFAULT_THEME,
    figsize: tuple[float, float] = (11.0, 8.0),
) -> plt.Figure:
    """Composite figure: side cut, wall-normal strip, and survival curves."""
    from b3_core.core.scoring import ScoreField

    field = ScoreField(inp)
    if not field.active:
        msg = "ScoreField inactive — set core.cell_size and grooves in inp"
        raise ValueError(msg)

    core = inp.get("core") or {}
    resin = inp.get("resin") or {}
    cell_size = core.get("cell_size")
    e_foam = float(core.get("E3") or core.get("E") or 70e6)
    e_resin = float(resin.get("E", 3e9))

    with plt.rc_context(theme.publication_rcparams()):
        fig = plt.figure(figsize=figsize, layout="constrained")
        gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], width_ratios=[1.2, 1.0])
        ax_cut = fig.add_subplot(gs[0, :])
        ax_strip = fig.add_subplot(gs[1, 0])
        ax_curve = fig.add_subplot(gs[1, 1])

        fig_cut, _ = plot_halo_side_cut(inp, theme=theme, figsize=(9.0, 4.2))
        fig_cut.canvas.draw()
        buf = np.asarray(fig_cut.canvas.buffer_rgba())[:, :, :3]
        plt.close(fig_cut)
        ax_cut.imshow(buf, aspect="auto")
        ax_cut.axis("off")
        ax_cut.set_title(
            "Grid-scored core: neat kerf + stochastic resin halo in opened foam cells",
            fontsize=11,
            pad=6,
        )

        groove = field.grooves[0]
        g_axis, c0, hw, _slope, depth = groove
        z = depth * 0.5 if depth > 0 else field.thickness + depth * 0.5
        span = field.reach * 1.2
        t = np.linspace(0.0, span, 300)
        pts = np.zeros((len(t), 3))
        pts[:, 2] = z
        if g_axis == 0:
            pts[:, 0] = c0 + hw + t
            pts[:, 1] = float(inp["dy"]) * 0.5
        else:
            pts[:, 1] = c0 + hw + t
            pts[:, 0] = float(inp["dx"]) * 0.5
        p = field.resin_probability(pts)
        ax_strip.fill_between(t, 0, p, color=theme.resin_color, alpha=0.3)
        ax_strip.plot(t, p, color=theme.resin_color, lw=2.0)
        ax_strip.axvline(field.reach, color="#888888", ls=":", lw=1.0)
        ax_strip.axvspan(0, field.reach, color=theme.resin_color, alpha=0.06)
        ax_strip.text(
            field.reach * 0.5,
            0.55,
            "halo reach",
            ha="center",
            fontsize=8,
            color=theme.resin_color,
        )
        ax_strip.set_xlim(0, span)
        ax_strip.set_ylim(0, 1.05)
        ax_strip.set_xlabel("Distance from groove wall [mm]")
        ax_strip.set_ylabel("P(resin)")
        ax_strip.set_title("Halo decay normal to cut", fontsize=9)
        ax_strip.grid(True, alpha=0.25)

        d, prob, reach = resin_probability_vs_distance(cell_size)
        e_ratio = effective_modulus_ratio(prob, e_foam=e_foam, e_resin=e_resin)
        ax_curve.plot(d, prob, color=theme.resin_color, lw=2.2, label="P(resin)")
        ax_curve.axvline(reach, color="#888888", ls=":", lw=1.0)
        ax_curve.set_xlabel("Distance from cut [mm]")
        ax_curve.set_ylabel("P(resin)", color=theme.resin_color)
        ax_curve.set_ylim(0, 1.05)
        ax_curve.grid(True, alpha=0.25)
        ax2 = ax_curve.twinx()
        ax2.plot(
            d,
            e_ratio,
            color=theme.face_color,
            lw=2.0,
            ls="--",
            label=r"$E_\mathrm{eff}/E_\mathrm{foam}$",
        )
        ax2.set_ylabel(r"$E_\mathrm{eff}/E_\mathrm{foam}$", color=theme.face_color)
        ax2.set_ylim(0, max(e_ratio.max() * 1.08, 1.05))
        ax_curve.set_title(
            f"Cell-size survival  (cell_size = {cell_size!s})",
            fontsize=9,
        )
        lines1, lab1 = ax_curve.get_legend_handles_labels()
        lines2, lab2 = ax2.get_legend_handles_labels()
        ax_curve.legend(lines1 + lines2, lab1 + lab2, loc="upper right", fontsize=7)

    return fig
