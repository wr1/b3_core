"""Resin-halo probability and stiffness-grading curves (P(resin) vs cut distance)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from b3_core.core.scoring import survival
from b3_core.viz.theme import DEFAULT_THEME, CoreTheme


def resin_probability_vs_distance(
    cell_size: float | dict[str, Any] | None,
    *,
    n: int = 300,
    reach_pad: float = 0.08,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Sample ``P(resin) = S(d)`` from distance ``d`` [mm] to the nearest cut.

    Returns ``(d, P, reach)`` where ``reach`` is the nominal max cell size.
    """
    s_fn, reach = survival(cell_size)
    if reach <= 0.0:
        d = np.array([0.0])
        return d, np.zeros_like(d), 0.0
    d = np.linspace(0.0, reach * (1.0 + reach_pad), n)
    return d, s_fn(d), float(reach)


def effective_modulus_ratio(
    p: np.ndarray,
    *,
    e_foam: float,
    e_resin: float,
) -> np.ndarray:
    """Isotropic rule-of-mixtures: ``E_eff / E_foam`` from ``P(resin)``."""
    ratio = e_resin / e_foam
    return p * ratio + (1.0 - p)


def plot_halo_degradation(
    cell_sizes: list[float | dict[str, Any] | None],
    labels: list[str] | None = None,
    *,
    e_foam: float | None = None,
    e_resin: float | None = None,
    highlight_index: int = 0,
    modulus_label: str = "E",
    theme: CoreTheme = DEFAULT_THEME,
    title: str | None = None,
    figsize: tuple[float, float] = (9.0, 5.0),
) -> tuple[plt.Figure, np.ndarray]:
    """Plot ``P(resin)`` and optional effective-modulus grading vs cut distance.

    Parameters
    ----------
    cell_sizes
        Foam ``cell_size`` specs (scalar, distribution dict, or ``None``).
    labels
        Curve labels; defaults to stringified ``cell_size`` values.
    e_foam, e_resin
        When both are set, the lower panel shows ``E_eff(d) / E_foam`` for the
        first (highlighted) ``cell_size`` entry.
    """
    if labels is None:
        labels = [str(cs) for cs in cell_sizes]
    if len(labels) != len(cell_sizes):
        msg = "labels must match cell_sizes length"
        raise ValueError(msg)

    with plt.rc_context(theme.publication_rcparams()):
        fig, axes = plt.subplots(
            1, 2 if e_foam and e_resin else 1,
            figsize=figsize if e_foam and e_resin else (figsize[0] * 0.55, figsize[1]),
            squeeze=False,
            layout="constrained",
        )
        ax_p = axes[0, 0]
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(cell_sizes)))

        max_reach = 0.0
        for cs, lab, col in zip(cell_sizes, labels, colors, strict=True):
            d, p, reach = resin_probability_vs_distance(cs)
            max_reach = max(max_reach, reach)
            ax_p.plot(d, p, lw=2.2, color=col, label=lab)
            if reach > 0:
                ax_p.axvline(reach, color=col, ls=":", lw=1.0, alpha=0.55)

        ax_p.set_xlim(0.0, max_reach * 1.08 if max_reach else 1.0)
        ax_p.set_ylim(0.0, 1.02)
        ax_p.set_xlabel("Distance from cut surface [mm]")
        ax_p.set_ylabel("P(resin)")
        ax_p.set_title("Resin presence (survival function)")
        ax_p.grid(True, alpha=0.25)
        ax_p.legend(loc="upper right", fontsize=8)

        if e_foam and e_resin:
            ax_e = axes[0, 1]
            highlight = cell_sizes[highlight_index]
            d, p, reach = resin_probability_vs_distance(highlight)
            e_ratio = effective_modulus_ratio(p, e_foam=e_foam, e_resin=e_resin)
            foam_frac = 1.0 - p

            ax_e.plot(d, e_ratio, color=theme.resin_color, lw=2.4,
                      label=f"{modulus_label}_eff / {modulus_label}_foam")
            ax_e.plot(d, foam_frac, color=theme.core_color, lw=1.8, ls="--",
                      label="1 − P(resin)  (intact foam fraction)")
            if reach > 0:
                ax_e.axvline(reach, color="#888888", ls=":", lw=1.0, alpha=0.7)
                ax_e.axhline(1.0, color="#cccccc", lw=0.8)
            ax_e.set_xlim(0.0, reach * 1.08 if reach else 1.0)
            ax_e.set_ylim(0.0, max(e_ratio.max() * 1.05, 1.02))
            ax_e.set_xlabel("Distance from cut surface [mm]")
            ax_e.set_ylabel("Normalized property")
            ax_e.set_title(
                f"Stiffness grading ({modulus_label}: "
                f"{e_foam/1e6:.0f} → {e_resin/1e9:.1f} GPa mix)"
            )
            ax_e.grid(True, alpha=0.25)
            ax_e.legend(loc="upper right", fontsize=8)

        if title:
            fig.suptitle(title, fontsize=11)

    return fig, axes


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
    z = (depth * 0.5 if depth > 0
         else field.thickness + depth * 0.5)
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
        ax.axvline(field.reach, color="#888888", ls=":", lw=1.0,
                   label=f"cell_size reach ≈ {field.reach:.2g} mm")
        ax.set_xlim(0.0, span_mm)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("P(resin)")
        ax.set_title("Halo strip — normal to groove wall (mid depth)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

    return fig, ax


def _mesh_and_field(inp: dict):
    from b3_core.core.cprop import halo_reach
    from b3_core.core.mesh import create_grooved_mesh
    from b3_core.core.scoring import ScoreField
    from b3_core.viz import geometry

    field = ScoreField(inp)
    if not field.active:
        msg = "ScoreField inactive — set core.cell_size and grooves in inp"
        raise ValueError(msg)
    mesh = create_grooved_mesh(
        thickness=float(inp["thickness"]),
        dx=float(inp["dx"]),
        dy=float(inp["dy"]),
        xcuts=inp.get("xgr", []),
        ycuts=inp.get("ygr", []),
        madd=tuple(inp.get("madd", [0])),
        tface=(inp.get("face") or {}).get("thickness", 0.0),
        kx=(inp.get("curvature") or {}).get("kx", 0.0),
        ky=(inp.get("curvature") or {}).get("ky", 0.0),
        s_halo=halo_reach(inp),
    )
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

    mesh, mat, field = _mesh_and_field(inp)
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
    resin_rgb = np.array(to_rgb(theme.resin_color))
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
            rgba, origin="lower", extent=[x0, x1, z0, z1],
            aspect="equal", interpolation="nearest",
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
                    arrowprops={"arrowstyle": "->", "color": theme.edge_color, "lw": 1.0},
                )
                ax.annotate(
                    "halo\n(opened foam cells)",
                    xy=(c0 + hw + field.reach * 0.45, z_mid),
                    xytext=(c0 + hw + field.reach * 2.6, z_mid - Lz * 0.18),
                    fontsize=8,
                    color=theme.resin_color,
                    arrowprops={"arrowstyle": "->", "color": theme.resin_color, "lw": 1.0},
                )
                ax.annotate(
                    "intact foam",
                    xy=(x1 - Lx * 0.12, z_mid),
                    xytext=(x1 - Lx * 0.38, z_mid + Lz * 0.28),
                    fontsize=8,
                    arrowprops={"arrowstyle": "->", "color": "#888888", "lw": 1.0},
                )

        sm = plt.cm.ScalarMappable(cmap=halo_cmap, norm=plt.Normalize(0.0, 1.0))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("P(resin) in foam")
        face_on = field.surfaces.get("face", {}).get("enabled", False)
        legend_handles = [
            Patch(facecolor=theme.resin_color, label="neat resin (kerf volume)"),
            Patch(facecolor=halo_cmap(0.55), label="saw-cut halo (opened cells)"),
            Patch(facecolor=theme.core_color, label="intact foam (P → 0)"),
        ]
        if face_on:
            legend_handles.insert(
                2,
                Patch(facecolor=halo_cmap(0.25), label="face halo (closed cells, thinner)"),
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
    from b3_core.core.mesh import create_grooved_mesh
    from b3_core.viz import geometry

    def _phase_cut(inp: dict, ax, title: str):
        mesh = create_grooved_mesh(
            thickness=float(inp["thickness"]),
            dx=float(inp["dx"]),
            dy=float(inp["dy"]),
            xcuts=inp.get("xgr", []),
            ycuts=inp.get("ygr", []),
            madd=tuple(inp.get("madd", [0])),
            tface=(inp.get("face") or {}).get("thickness", 0.0),
        )
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
            grid, origin="lower", extent=[xv[0], xv[-1], zv[0], zv[-1]],
            cmap=cmap, norm=norm, aspect="equal", interpolation="nearest",
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
            field.reach * 0.5, 0.55,
            "halo reach",
            ha="center", fontsize=8, color=theme.resin_color,
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
        ax2.plot(d, e_ratio, color=theme.face_color, lw=2.0, ls="--",
                 label=r"$E_\mathrm{eff}/E_\mathrm{foam}$")
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


def render_halo_3d_png(
    inp: dict,
    path: str | Path,
    *,
    theme: CoreTheme = DEFAULT_THEME,
    window_size: tuple[int, int] = (900, 720),
) -> Path:
    """3D view: neat resin solid, foam coloured by ``P(resin)`` at cell centres."""
    from b3_core.viz._deps import ensure_headless, require_pyvista
    from b3_core.viz import geometry

    mesh, mat, field = _mesh_and_field(inp)
    centers = mesh.cell_centers().points
    p = np.zeros(mesh.n_cells)
    from b3_core.viz.theme import CORE, RESIN

    foam = mat == CORE
    resin = mat == RESIN
    if foam.any():
        p[foam] = field.resin_probability(centers[foam])
    p[resin] = 1.0

    ensure_headless()
    pv = require_pyvista()
    phases = geometry.split_phases(mesh, mat)
    plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
    plotter.set_background(theme.background)

    if phases["core"].n_cells:
        core_view = phases["core"].copy()
        core_ids = np.where(mat == CORE)[0]
        core_view.cell_data["halo_p"] = p[core_ids]
        plotter.add_mesh(
            core_view,
            scalars="halo_p",
            cmap=[theme.core_color, theme.resin_color],
            clim=[0.0, 1.0],
            opacity=0.95,
            show_edges=False,
            scalar_bar_args={"title": "P(resin)", "n_labels": 5},
        )
    if phases["resin"].n_cells:
        plotter.add_mesh(
            phases["resin"],
            color=theme.resin_color,
            opacity=1.0,
            show_edges=True,
            edge_color=theme.edge_color,
            line_width=theme.edge_width,
        )

    plotter.add_text("Resin halo — foam graded by P(resin)", font_size=10)
    plotter.camera_position = "iso"
    plotter.camera.azimuth = -40
    plotter.camera.elevation = -15
    plotter.camera.zoom(1.1)
    out = Path(path)
    plotter.screenshot(str(out))
    plotter.close()
    return out