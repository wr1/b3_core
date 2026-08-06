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
                Patch(facecolor=halo_cmap(0.35), label="face halo (closed cells, thinner)"),
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
        rwb = ["#dbe9f6", "#92c5de", "#f7f7f7", "#f4a582", "#b2182b"]
        plotter.add_mesh(
            core_view,
            scalars="halo_p",
            cmap=rwb,
            clim=[0.0, 1.0],
            opacity=0.95,
            show_edges=False,
            scalar_bar_args={"title": "P(resin)", "n_labels": 5},
        )
    if phases["resin"].n_cells:
        plotter.add_mesh(
            phases["resin"],
            color=theme.halo_resin_color(),
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


# Same open/close κ as examples/curved_panel/render.py (kerf open & close docs).
_CURVED_PANEL_KX_OPEN = 0.012
_CURVED_PANEL_KX_CLOSED = -0.012


def _default_halo_curvature_case(*, cell_size: float = 0.6) -> dict:
    """``examples/curved_panel/base.json`` geometry + resin halo.

    Top-mouth grooves (``depth < 0``): ``kx > 0`` opens, ``kx < 0`` closes —
    identical to the kerf open & close documentation figures.
    """
    # Prefer the live example file so docs and viz cannot drift.
    roots = [
        Path(__file__).resolve().parents[3] / "examples" / "curved_panel" / "base.json",
        Path.cwd() / "examples" / "curved_panel" / "base.json",
    ]
    base: dict | None = None
    for p in roots:
        if p.is_file():
            import json

            base = json.loads(p.read_text())
            break
    if base is None:
        base = {
            "dx": 50,
            "dy": 50,
            "thickness": 30,
            "xgr": [[10, 10, -27, 3]],
            "ygr": [],
            "madd": [-0.4, -0.2, 0, 0.2, 0.4],
            "core": {"E": 130e6, "nu": 0.30, "rho": 100},
            "resin": {"E": 3e9, "nu": 0.35, "rho": 1100},
        }
    core = dict(base.get("core") or {})
    core["cell_size"] = float(cell_size)
    base = dict(base)
    base["core"] = core
    base["ygr"] = base.get("ygr") or []
    base["scoring"] = {
        "damage_cells": 1.0,
        "surfaces": {"face": {"enabled": False}},
    }
    base["curvature"] = {"kx": 0.0, "ky": 0.0}
    # No face layer in the curved-panel FEA strip.
    if "face" in base:
        del base["face"]
    return base


def _with_kx(inp: dict, kx: float) -> dict:
    out = dict(inp)
    cur = dict(out.get("curvature") or {})
    cur["kx"] = float(kx)
    cur.setdefault("ky", 0.0)
    out["curvature"] = cur
    return out


def _halo_band_cmap():
    """Discrete blue→white→red bands so thin halo rims stay readable."""
    from matplotlib.colors import BoundaryNorm, ListedColormap

    # P=0 foam is drawn separately (neutral grey). Bands only for P>0.
    colors = ["#2166ac", "#67a9cf", "#f7f7f7", "#ef8a62", "#b2182b"]
    bounds = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm, bounds


def _paint_halo_cut(
    ax,
    inp: dict,
    *,
    theme: CoreTheme,
    px: int = 280,
    zoom: tuple[float, float, float, float] | None = None,
    draw_walls: bool = True,
    emphasize_halo: bool = False,
) -> None:
    """Paint one x–z halo cut onto *ax* (neat resin + P in foam + wall lines).

    When ``emphasize_halo`` is set (compose figures):
    - intact foam is neutral grey (not a solid blue slab)
    - only the halo rim uses discrete RWB bands
    - kerf wall + outer reach envelopes are drawn so the grade is locatable
    """
    from matplotlib.colors import to_rgb

    from b3_core.core.mesh import _hw_at

    mesh, mat, field = _mesh_and_field(inp)
    xv, zv = np.unique(mesh.x), np.unique(mesh.z)
    x0, x1, z0, z1 = float(xv[0]), float(xv[-1]), float(zv[0]), float(zv[-1])
    if zoom is not None:
        x0, x1, z0, z1 = zoom
    yc = float(inp["dy"]) * 0.5
    Lx, Lz = max(x1 - x0, 1e-9), max(z1 - z0, 1e-9)
    ref = max(Lx, Lz)
    # Dense sampling so a 0.6–2 mm band is many pixels wide.
    nx = int(np.clip(px * Lx / ref, 100, max(px, 500)))
    nz = int(np.clip(px * Lz / ref, 100, max(px, 500)))
    if emphasize_halo:
        nx = max(nx, 220)
        nz = max(nz, 280)
    ux = np.linspace(x0 + 1e-4, x1 - 1e-4, nx)
    uz = np.linspace(z0 + 1e-4, z1 - 1e-4, nz)
    p_grid, phase = sample_halo_plane(mesh, mat, field, 0, 2, 1, yc, ux, uz)

    resin_rgb = np.array(to_rgb(theme.halo_resin_color()))
    foam_rgb = np.array(to_rgb("#e8e8e8"))  # neutral intact foam
    rgba = np.zeros((*p_grid.shape, 4))
    valid = ~np.isnan(p_grid)
    foam = valid & (phase < 0.5)
    neat = valid & (phase >= 0.5)
    rgba[neat] = (*resin_rgb, 1.0)
    rgba[~valid] = (1.0, 1.0, 1.0, 0.0)

    if emphasize_halo:
        cmap, norm, _bounds = _halo_band_cmap()
        rgba[foam] = (*foam_rgb, 1.0)
        # Colour only the graded rim; leave far foam grey.
        in_halo = foam & (p_grid > 0.02)
        if in_halo.any():
            rgba[in_halo] = cmap(norm(np.clip(p_grid[in_halo], 0.0, 1.0)))
    else:
        halo_cmap = theme.halo_cmap()
        rgba[foam] = halo_cmap(np.clip(p_grid[foam], 0.0, 1.0))

    ax.imshow(
        rgba,
        origin="lower",
        extent=[x0, x1, z0, z1],
        aspect="equal",
        interpolation="bilinear" if emphasize_halo else "nearest",
    )

    th = float(inp["thickness"])
    reach = float(getattr(field, "reach", 0.0) or 0.0)
    if draw_walls and field.grooves:
        for g_axis, c0, hw0, slope, depth in field.grooves:
            if g_axis != 0:
                continue
            if depth > 0:
                zs = np.linspace(0.0, float(depth), 120)
            else:
                zs = np.linspace(th + float(depth), th, 120)
            hw = np.array([_hw_at(hw0, depth, slope, float(z), th) for z in zs])
            # Kerf wall (morph + ScoreField surface).
            ax.plot(c0 - hw, zs, color=theme.edge_color, ls="-", lw=1.4, zorder=5)
            ax.plot(c0 + hw, zs, color=theme.edge_color, ls="-", lw=1.4, zorder=5)
            if emphasize_halo and reach > 0:
                # Outer reach of the halo band (P→0 outside this envelope).
                ax.plot(
                    c0 - hw - reach,
                    zs,
                    color="#2166ac",
                    ls=":",
                    lw=1.2,
                    zorder=5,
                    alpha=0.95,
                )
                ax.plot(
                    c0 + hw + reach,
                    zs,
                    color="#2166ac",
                    ls=":",
                    lw=1.2,
                    zorder=5,
                    alpha=0.95,
                )
    ax.set_xlim(x0, x1)
    ax.set_ylim(z0, z1)


def plot_halo_curvature_compose(
    base_inp: dict | None = None,
    *,
    kx_open: float = _CURVED_PANEL_KX_OPEN,
    kx_closed: float = _CURVED_PANEL_KX_CLOSED,
    theme: CoreTheme = DEFAULT_THEME,
    figsize: tuple[float, float] = (11.8, 7.4),
    px: int = 360,
    # Slightly larger than production 0.6 mm so the rim is a few mm across
    # at RVE scale; still the same S(d) law.
    cell_size: float = 1.5,
) -> plt.Figure:
    """Closed | flat | open with halo — same κ as the kerf open & close docs.

    Uses ``examples/curved_panel`` geometry (top-mouth, ``depth < 0``):
    ``kx > 0`` opens, ``kx < 0`` closes.

    Visual emphasis (so the grade is not a 1-pixel rim):
    - grey intact foam, discrete RWB bands only where ``P > 0``
    - solid black = kerf wall ``hw(z)``; blue dotted = outer halo reach
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    base = dict(base_inp or _default_halo_curvature_case(cell_size=cell_size))
    # Ensure face halo off so only saw-cut rim shows.
    scoring = dict(base.get("scoring") or {})
    scoring["damage_cells"] = 1.0
    surfaces = dict(scoring.get("surfaces") or {})
    surfaces["face"] = {"enabled": False}
    scoring["surfaces"] = surfaces
    base["scoring"] = scoring

    panels = (
        (f"Closed  kx={kx_closed:+.3f}  (pinches free face)", kx_closed),
        ("Flat  kx=0", 0.0),
        (f"Open  kx={kx_open:+.3f}  (flares free face)", kx_open),
    )

    # Zoom on an interior kerf (curved_panel: centres at 0,10,…,50 → pick 20).
    row = (base.get("xgr") or [[10, 10, -27, 3]])[0]
    offset, pitch, depth, width = map(float, row)
    c0 = float(offset + pitch)  # first full interior centre (10+10=20)
    th = float(base["thickness"])
    reach = float(cell_size)
    hw0 = 0.5 * width
    if depth < 0:
        z_lo, z_hi = th + depth - 0.5, th + 0.5
    else:
        z_lo, z_hi = -0.5, depth + 0.5
    # Wide enough to show wall + full reach on both sides.
    pad = hw0 + reach * 2.2 + 0.8
    zoom = (c0 - pad, c0 + pad, z_lo, z_hi)

    band_cmap, band_norm, _ = _halo_band_cmap()

    with plt.rc_context(theme.publication_rcparams()):
        fig, axes = plt.subplots(2, 3, figsize=figsize, layout="constrained")
        for col, (title, kx) in enumerate(panels):
            inp = _with_kx(base, kx)
            ax_full, ax_zoom = axes[0, col], axes[1, col]
            _paint_halo_cut(
                ax_full,
                inp,
                theme=theme,
                px=px,
                draw_walls=True,
                emphasize_halo=True,
            )
            _paint_halo_cut(
                ax_zoom,
                inp,
                theme=theme,
                px=px + 120,
                zoom=zoom,
                draw_walls=True,
                emphasize_halo=True,
            )
            ax_full.set_title(title, fontsize=9)
            ax_full.set_xlabel("x [mm]")
            ax_zoom.set_xlabel("x [mm]")
            if col == 0:
                ax_full.set_ylabel("z [mm]")
                ax_zoom.set_ylabel("z [mm]")
            else:
                ax_full.set_ylabel("")
                ax_zoom.set_ylabel("")
            ax_zoom.set_title("interior kerf — wall + discrete halo bands", fontsize=8)

        sm = plt.cm.ScalarMappable(cmap=band_cmap, norm=band_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02, ticks=[0.1, 0.3, 0.5, 0.7, 0.9])
        cbar.set_label("P(resin) in halo rim  (blue→white→red)")
        cbar.ax.set_yticklabels(["0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1"])
        legend_handles = [
            Patch(facecolor=theme.halo_resin_color(), label="neat resin (morphed kerf)"),
            Patch(facecolor="#ef8a62", label="halo rim (graded P)"),
            Patch(facecolor="#e8e8e8", edgecolor="#888", label="intact foam (P ≈ 0)"),
            Line2D([0], [0], color=theme.edge_color, lw=1.4, label="kerf wall hw(z)"),
            Line2D(
                [0], [0], color="#2166ac", lw=1.2, ls=":", label="halo reach (wall + cell_size)"
            ),
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=3,
            fontsize=7.5,
            framealpha=0.92,
            bbox_to_anchor=(0.5, -0.03),
        )
        fig.suptitle(
            "Halo on the kerf open & close RVE  "
            f"(curved_panel base · cell_size = {cell_size:g} mm for band visibility)",
            fontsize=11,
        )
    return fig


def plot_halo_curvature_wall_strip(
    base_inp: dict | None = None,
    *,
    kx_open: float = _CURVED_PANEL_KX_OPEN,
    kx_closed: float = _CURVED_PANEL_KX_CLOSED,
    theme: CoreTheme = DEFAULT_THEME,
    figsize: tuple[float, float] = (9.0, 4.2),
) -> tuple[plt.Figure, plt.Axes]:
    """P(resin) vs stand-off from the *local* wall at mouth and root.

    Same survival decay from open vs closed walls — the wall moves with κ;
    ``S(d)`` does not. Uses curved_panel open/closed κ by default.
    """
    from b3_core.core.mesh import _hw_at
    from b3_core.core.scoring import ScoreField

    base = dict(base_inp or _default_halo_curvature_case())
    th = float(base["thickness"])
    cs = float((base.get("core") or {}).get("cell_size") or 0.6)
    span = cs * 1.4
    t = np.linspace(0.0, span, 250)
    yc = float(base["dy"]) * 0.5

    series = (
        ("open mouth", kx_open, "mouth"),
        ("open root", kx_open, "root"),
        ("closed mouth", kx_closed, "mouth"),
        ("closed root", kx_closed, "root"),
    )
    styles = {
        ("open", "mouth"): {"color": theme.resin_color, "ls": "-", "lw": 2.3},
        ("open", "root"): {"color": theme.resin_color, "ls": "--", "lw": 1.6},
        ("closed", "mouth"): {"color": "#c45c26", "ls": "-", "lw": 2.3},
        ("closed", "root"): {"color": "#c45c26", "ls": "--", "lw": 1.6},
    }

    with plt.rc_context(theme.publication_rcparams()):
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        for label, kx, where in series:
            inp = _with_kx(base, kx)
            field = ScoreField(inp)
            # Interior x-groove (skip domain-edge partials).
            g_axis, c0, hw0, slope, depth = next(
                g
                for g in field.grooves
                if g[0] == 0 and 5.0 < g[1] < float(base["dx"]) - 5.0
            )
            if depth > 0:
                z_mouth, z_root = 0.15, float(depth) - 0.15
            else:
                z_mouth, z_root = th - 0.15, th + float(depth) + 0.15
            z = z_mouth if where == "mouth" else z_root
            hw = _hw_at(hw0, depth, slope, z, th)
            pts = np.zeros((len(t), 3))
            pts[:, 0] = c0 + hw + t
            pts[:, 1] = yc
            pts[:, 2] = z
            p = field.resin_probability(pts)
            # Top-mouth: open has kx>0, closed kx<0.
            key = ("open" if kx > 0 else "closed", where)
            ax.plot(t, p, label=label, **styles[key])

        ax.axvline(cs, color="#888888", ls=":", lw=1.0, label="cell_size reach")
        ax.set_xlim(0.0, span)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("Wall-normal distance from tapered face [mm]")
        ax.set_ylabel("P(resin)")
        ax.set_title(
            "Halo decay rides the wall: same S(d) at open/closed mouth and root"
        )
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=7, ncol=2)
    return fig, ax

def _resin_probability_flat_walls(field, points: np.ndarray) -> np.ndarray:
    """P(resin) as if kerf walls stayed rectangular (slope forced to 0).

    Same survival reach as *field*, but distance uses constant ``hw0`` — the
    pre-curvature wall. Used only to contrast the correct tapered-wall halo.
    """
    from b3_core.core.mesh import _MIN_HW

    pts = np.asarray(points, dtype=float)
    z = pts[:, 2]
    d_min = np.full(len(pts), np.inf)
    th = field.thickness
    for axis, c0, hw0, _slope, depth in field.grooves:
        if depth > 0:
            z0, z1 = 0.0, depth
        else:
            z0, z1 = th + depth, th
        inside = (z >= z0) & (z <= z1)
        hw = np.where(inside, max(_MIN_HW, hw0), hw0)
        du = np.maximum(0.0, np.abs(pts[:, axis] - c0) - hw)
        dz = np.maximum(0.0, np.maximum(z0 - z, z - z1))
        d_min = np.minimum(d_min, np.hypot(du, dz))
    saw = field.surfaces["saw_cut"]
    if not saw.get("enabled", True):
        return np.zeros(len(pts))
    return np.asarray(saw["S"](d_min), dtype=float)


def _sample_p_plane(
    mesh,
    mat: np.ndarray,
    p_fn,
    u_axis: int,
    v_axis: int,
    fixed_axis: int,
    coord: float,
    u_vals: np.ndarray,
    v_vals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Like :func:`sample_halo_plane` but with a custom ``p_fn(points)``."""
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
    p_field = np.asarray(p_fn(pts[inside]), dtype=float)
    for j, flat in enumerate(np.flatnonzero(inside)):
        cid = int(cids[flat])
        if mat[cid] == RESIN:
            p[flat] = 1.0
            phase[flat] = 1.0
        else:
            p[flat] = p_field[j]
            phase[flat] = 0.0
    return p.reshape(uu.shape), phase.reshape(uu.shape)


def _draw_wall_pair(ax, field, *, th: float, style: dict, label: str | None = None):
    """Plot left/right analytical walls for the first x-groove."""
    from b3_core.core.mesh import _hw_at

    for g_axis, c0, hw0, slope, depth in field.grooves:
        if g_axis != 0:
            continue
        if depth > 0:
            zs = np.linspace(0.0, float(depth), 120)
        else:
            zs = np.linspace(th + float(depth), th, 120)
        hw = np.array([_hw_at(hw0, depth, slope, float(z), th) for z in zs])
        ax.plot(c0 - hw, zs, label=label, **style)
        ax.plot(c0 + hw, zs, **{**style, "label": None})
        return


def plot_halo_follows_angled_walls(
    base_inp: dict | None = None,
    *,
    kx: float = -0.012,
    theme: CoreTheme = DEFAULT_THEME,
    figsize: tuple[float, float] = (11.2, 8.0),
    px: int = 360,
) -> plt.Figure:
    """Show that the halo band updates onto the *angled* (morphed) kerf walls.

    Layout
    ------
    Top: open morph with correct tapered-wall ``P(resin)``; ghost rectangular
    walls vs solid tapered walls; zoom on the right face with ``P`` contours.
    Bottom: same open mesh — halo if walls stayed rectangular (wrong) vs halo
    on ``hw(z)`` (correct) vs difference (where the band moved with the wall).
    """
    from matplotlib.colors import to_rgb
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from b3_core.core.mesh import _hw_at

    base = dict(base_inp or _default_halo_curvature_case())
    # Thicker reach so the band is obvious next to the angled face.
    core = dict(base.get("core") or {})
    core["cell_size"] = max(float(core.get("cell_size") or 0.6), 1.0)
    base["core"] = core
    scoring = dict(base.get("scoring") or {})
    scoring["damage_cells"] = max(float(scoring.get("damage_cells") or 1.0), 1.5)
    surfaces = dict(scoring.get("surfaces") or {})
    face = dict(surfaces.get("face") or {})
    face["enabled"] = False
    surfaces["face"] = face
    scoring["surfaces"] = surfaces
    base["scoring"] = scoring

    inp = _with_kx(base, kx)
    mesh, mat, field = _mesh_and_field(inp)
    th = float(inp["thickness"])
    yc = float(inp["dy"]) * 0.5
    g_axis, c0, hw0, slope, depth = next(g for g in field.grooves if g[0] == 0)
    assert g_axis == 0

    # Full-frame and right-wall zoom extents.
    xv, zv = np.unique(mesh.x), np.unique(mesh.z)
    x0, x1 = float(xv[0]), float(xv[-1])
    z0, z1 = float(zv[0]), float(zv[-1])
    z_depth = abs(float(depth))
    reach = float(field.reach)
    # Span both the rectangular ghost wall and the extreme tapered wall so
    # wrong-vs-correct halo bands are both in frame.
    z_lo_w = 0.2 if depth > 0 else th + float(depth) + 0.2
    z_hi_w = float(depth) - 0.2 if depth > 0 else th - 0.2
    hw_a = _hw_at(hw0, depth, slope, z_lo_w, th)
    hw_b = _hw_at(hw0, depth, slope, z_hi_w, th)
    wall_r_min = c0 + min(hw0, hw_a, hw_b)
    wall_r_max = c0 + max(hw0, hw_a, hw_b)
    zoom = (
        wall_r_min - 0.6 * reach,
        wall_r_max + 1.4 * reach,
        -0.15,
        z_depth + 0.4,
    )

    def _grid(zoom_box=None):
        xa, xb, za, zb = zoom_box if zoom_box else (x0, x1, z0, z1)
        Lx, Lz = max(xb - xa, 1e-9), max(zb - za, 1e-9)
        ref = max(Lx, Lz)
        nx = int(np.clip(px * Lx / ref, 80, px))
        nz = int(np.clip(px * Lz / ref, 80, px))
        ux = np.linspace(xa + 1e-4, xb - 1e-4, nx)
        uz = np.linspace(za + 1e-4, zb - 1e-4, nz)
        return ux, uz, (xa, xb, za, zb)

    def _rgba(p_grid, phase):
        resin_rgb = np.array(to_rgb(theme.resin_color))
        halo_cmap = theme.halo_cmap()
        rgba = np.zeros((*p_grid.shape, 4))
        valid = ~np.isnan(p_grid)
        foam = valid & (phase < 0.5)
        neat = valid & (phase >= 0.5)
        rgba[foam] = halo_cmap(np.clip(p_grid[foam], 0.0, 1.0))
        rgba[neat] = (*resin_rgb, 1.0)
        rgba[~valid] = (1.0, 1.0, 1.0, 0.0)
        return rgba

    def _paint(ax, p_fn, *, zoom_box=None, contours: bool = False):
        ux, uz, ext = _grid(zoom_box)
        p_grid, phase = _sample_p_plane(
            mesh, mat, p_fn, 0, 2, 1, yc, ux, uz
        )
        ax.imshow(
            _rgba(p_grid, phase),
            origin="lower",
            extent=[ext[0], ext[1], ext[2], ext[3]],
            aspect="equal",
            interpolation="nearest",
        )
        if contours:
            foam = (~np.isnan(p_grid)) & (phase < 0.5)
            pc = np.where(foam, p_grid, np.nan)
            if np.nanmax(pc) > 0.05:
                cs = ax.contour(
                    ux,
                    uz,
                    pc,
                    levels=[0.25, 0.5, 0.75],
                    colors=["#0d5c57", "#0d5c57", "#0d5c57"],
                    linewidths=[0.7, 1.0, 0.7],
                    alpha=0.95,
                )
                ax.clabel(cs, fmt="P=%.2f", fontsize=6, inline=True)
        # Ghost rectangular walls (pre-curvature).
        if depth > 0:
            zs = np.linspace(0.0, float(depth), 80)
        else:
            zs = np.linspace(th + float(depth), th, 80)
        ax.plot(
            np.full_like(zs, c0 - hw0),
            zs,
            color="#888888",
            ls=":",
            lw=1.4,
            alpha=0.95,
        )
        ax.plot(
            np.full_like(zs, c0 + hw0),
            zs,
            color="#888888",
            ls=":",
            lw=1.4,
            alpha=0.95,
        )
        # Tapered walls (morph + ScoreField).
        hw = np.array([_hw_at(hw0, depth, slope, float(z), th) for z in zs])
        ax.plot(c0 - hw, zs, color=theme.edge_color, ls="-", lw=1.6)
        ax.plot(c0 + hw, zs, color=theme.edge_color, ls="-", lw=1.6)
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        return p_grid, phase, ux, uz, ext

    p_correct = field.resin_probability
    p_flat = lambda pts: _resin_probability_flat_walls(field, pts)

    with plt.rc_context(theme.publication_rcparams()):
        fig = plt.figure(figsize=figsize, layout="constrained")
        gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0], width_ratios=[1.0, 1.0, 1.0])
        ax_full = fig.add_subplot(gs[0, 0])
        ax_zoom = fig.add_subplot(gs[0, 1:])
        ax_wrong = fig.add_subplot(gs[1, 0])
        ax_right = fig.add_subplot(gs[1, 1])
        ax_diff = fig.add_subplot(gs[1, 2])

        _paint(ax_full, p_correct, contours=False)
        ax_full.set_title("Open morph + correct halo\n(full RVE)", fontsize=9)
        ax_full.set_xlabel("x [mm]")
        ax_full.set_ylabel("z [mm]")

        _paint(ax_zoom, p_correct, zoom_box=zoom, contours=True)
        ax_zoom.set_title(
            "Right wall zoom — halo contours hug the *angled* face",
            fontsize=9,
        )
        ax_zoom.set_xlabel("x [mm]")
        ax_zoom.set_ylabel("z [mm]")
        # Callouts.
        z_mid = 0.35 * z_depth
        hw_mid = _hw_at(hw0, depth, slope, z_mid, th)
        ax_zoom.annotate(
            "tapered wall\n(c₀ ± hw(z))",
            xy=(c0 + hw_mid, z_mid),
            xytext=(c0 + hw_mid + reach * 0.55, z_mid + z_depth * 0.25),
            fontsize=7.5,
            color=theme.edge_color,
            arrowprops={
                "arrowstyle": "->",
                "color": theme.edge_color,
                "lw": 1.0,
            },
        )
        ax_zoom.annotate(
            "rectangular\n(pre-κ ghost)",
            xy=(c0 + hw0, z_mid),
            xytext=(c0 + hw0 - reach * 0.9, z_mid + z_depth * 0.35),
            fontsize=7.5,
            color="#666666",
            arrowprops={"arrowstyle": "->", "color": "#666666", "lw": 1.0},
        )
        ax_zoom.annotate(
            "halo band\nmoves with wall",
            xy=(c0 + hw_mid + reach * 0.35, z_mid * 0.5),
            xytext=(c0 + hw_mid + reach * 0.7, z_mid * 0.15),
            fontsize=7.5,
            color=theme.resin_color,
            arrowprops={
                "arrowstyle": "->",
                "color": theme.resin_color,
                "lw": 1.0,
            },
        )

        _paint(ax_wrong, p_flat, zoom_box=zoom, contours=True)
        ax_wrong.set_title(
            "Wrong: halo from *rectangular* walls\n(ignores κ taper)",
            fontsize=9,
        )
        ax_wrong.set_xlabel("x [mm]")
        ax_wrong.set_ylabel("z [mm]")

        _paint(ax_right, p_correct, zoom_box=zoom, contours=True)
        ax_right.set_title(
            "Correct: halo from *angled* walls\n(same hw(z) as morph)",
            fontsize=9,
        )
        ax_right.set_xlabel("x [mm]")

        # Difference on foam only: where the band moved with the wall.
        ux, uz, ext = _grid(zoom)
        p_bad, phase_bad = _sample_p_plane(
            mesh, mat, p_flat, 0, 2, 1, yc, ux, uz
        )
        p_ok2, phase_ok2 = _sample_p_plane(
            mesh, mat, p_correct, 0, 2, 1, yc, ux, uz
        )
        foam = (
            (~np.isnan(p_ok2))
            & (~np.isnan(p_bad))
            & (phase_ok2 < 0.5)
            & (phase_bad < 0.5)
        )
        diff = np.full_like(p_ok2, np.nan)
        diff[foam] = p_ok2[foam] - p_bad[foam]
        # Neat resin mask for context.
        neat = (~np.isnan(phase_ok2)) & (phase_ok2 >= 0.5)
        rgba_d = np.ones((*diff.shape, 4))
        rgba_d[..., 3] = 0.0
        if foam.any():
            vmax = max(float(np.nanmax(np.abs(diff[foam]))), 0.15)
            # Diverging: blue = correct has less, red = correct has more (band moved out).
            from matplotlib import colormaps

            dcmap = colormaps["RdBu_r"]
            normed = np.clip((diff[foam] + vmax) / (2 * vmax), 0.0, 1.0)
            rgba_d[foam] = dcmap(normed)
            rgba_d[foam, 3] = 0.95
        rgba_d[neat] = (*to_rgb(theme.resin_color), 0.35)
        ax_diff.imshow(
            rgba_d,
            origin="lower",
            extent=[ext[0], ext[1], ext[2], ext[3]],
            aspect="equal",
            interpolation="nearest",
        )
        if depth > 0:
            zs = np.linspace(0.0, float(depth), 80)
        else:
            zs = np.linspace(th + float(depth), th, 80)
        ax_diff.plot(np.full_like(zs, c0 + hw0), zs, color="#888888", ls=":", lw=1.4)
        hw = np.array([_hw_at(hw0, depth, slope, float(z), th) for z in zs])
        ax_diff.plot(c0 + hw, zs, color=theme.edge_color, ls="-", lw=1.6)
        ax_diff.set_xlim(ext[0], ext[1])
        ax_diff.set_ylim(ext[2], ext[3])
        ax_diff.set_title(
            "ΔP = correct − rectangular\n(red: halo moved *out* with open wall)",
            fontsize=9,
        )
        ax_diff.set_xlabel("x [mm]")
        if foam.any():
            sm = plt.cm.ScalarMappable(
                cmap="RdBu_r",
                norm=plt.Normalize(-vmax, vmax),
            )
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax_diff, fraction=0.046, pad=0.04)
            cbar.set_label("ΔP(resin)")

        halo_cmap = theme.halo_cmap()
        legend = [
            Patch(facecolor=theme.resin_color, label="neat resin (morphed)"),
            Patch(facecolor=halo_cmap(0.55), label="P(resin) halo in foam"),
            Line2D([0], [0], color=theme.edge_color, lw=1.6, label="angled wall hw(z)"),
            Line2D(
                [0], [0], color="#888888", lw=1.4, ls=":", label="rectangular wall (ghost)"
            ),
        ]
        fig.legend(
            handles=legend,
            loc="lower center",
            ncol=4,
            fontsize=8,
            framealpha=0.92,
            bbox_to_anchor=(0.5, -0.02),
        )
        fig.suptitle(
            "Halo updates onto the angled kerf walls  "
            f"(kx = {kx:+.3f}, cell_size = {core['cell_size']:.2g} mm)",
            fontsize=11,
        )
    return fig


# ---------------------------------------------------------------------------
# Parametric stiffness: κ × halo width (cell_size)
# ---------------------------------------------------------------------------
_PARAM_FOAM = {
    "E1": 32e6,
    "E2": 32e6,
    "E3": 70e6,
    "G12": 19e6,
    "G13": 19e6,
    "G23": 19e6,
    "nu12": 0.3,
    "nu13": 0.3,
    "nu23": 0.3,
    "rho": 60,
}
_PARAM_RESIN = {"E": 3e9, "nu": 0.3, "rho": 1100}


def _parametric_base_case() -> dict:
    """Compact top-mouth RVE for κ × cell_size sweeps (numpy, ~0.2 s/point).

    Same sign convention as curved_panel: ``depth < 0``, ``kx > 0`` opens.
    """
    return {
        "dx": 30.0,
        "dy": 12.0,
        "thickness": 20.0,
        "xgr": [[5.0, 10.0, -17.0, 2.0]],
        "ygr": [],
        "madd": [0.0],
        "core": dict(_PARAM_FOAM),
        "resin": dict(_PARAM_RESIN),
        "scoring": {"damage_cells": 1.0, "surfaces": {"face": {"enabled": False}}},
        "curvature": {"kx": 0.0, "ky": 0.0},
    }


def homogenize_halo_curvature(
    kx: float,
    cell_size: float | None,
    *,
    base: dict | None = None,
) -> dict[str, float]:
    """One homogenization at given ``kx`` and halo width (``cell_size`` mm).

    ``cell_size is None`` or ``<= 0`` → sharp kerf (no ScoreField).
    Returns moduli [Pa], resin volume fractions, and inputs echoed.
    """
    from b3_core.core.analysis import geom_analysis
    from b3_core.core.cprop import halo_reach
    from b3_core.core.mesh import create_grooved_mesh
    from b3_core.core.scoring import ScoreField, effective_resin_vf
    from b3_core.io import aniso

    inp = dict(base or _parametric_base_case())
    core = dict(inp.get("core") or {})
    resin = dict(inp.get("resin") or _PARAM_RESIN)
    if cell_size is None or float(cell_size) <= 0.0:
        core.pop("cell_size", None)
        use_halo = False
    else:
        core["cell_size"] = float(cell_size)
        use_halo = True
    inp["core"] = core
    inp["resin"] = resin
    inp["curvature"] = {"kx": float(kx), "ky": 0.0}

    th = float(inp["thickness"])
    dx, dy = float(inp["dx"]), float(inp["dy"])
    madd = tuple(inp.get("madd") or [0.0])
    s_halo = halo_reach(inp) if use_halo else 0.0
    mesh = create_grooved_mesh(
        thickness=th,
        dx=dx,
        dy=dy,
        xcuts=inp.get("xgr") or [],
        ycuts=inp.get("ygr") or [],
        madd=madd,
        tface=0.0,
        kx=float(kx),
        ky=0.0,
        s_halo=s_halo,
    )
    geom = geom_analysis(mesh)
    sf = ScoreField(inp) if use_halo else None
    C = aniso.runnumpy(
        mesh,
        resin,
        core,
        score_field=sf,
        scoring=inp.get("scoring") if use_halo else None,
    ).stiffness
    props = aniso._properties_from_stiffness(C)[0]
    resin_vf = float(geom["resin_vf"])
    if sf is not None:
        eff, halo_vf = effective_resin_vf(mesh, sf, resin_vf)
    else:
        eff, halo_vf = resin_vf, 0.0
    return {
        "kx": float(kx),
        "cell_size": float(cell_size or 0.0),
        "Exx": float(props["Exx"]),
        "Eyy": float(props["Eyy"]),
        "Ezz": float(props["Ezz"]),
        "Gxy": float(props["Gxy"]),
        "Gxz": float(props["Gxz"]),
        "Gyz": float(props["Gyz"]),
        "resin_vf": resin_vf,
        "halo_vf": float(halo_vf),
        "effective_resin_vf": float(eff),
    }


def sweep_halo_curvature_grid(
    *,
    kx_values: list[float] | np.ndarray | None = None,
    cell_sizes: list[float] | np.ndarray | None = None,
    base: dict | None = None,
    cache_path: str | Path | None = None,
) -> list[dict[str, float]]:
    """Cartesian product of ``kx`` × ``cell_size`` (0 = sharp kerf).

    Results are cached to ``cache_path`` JSON when provided so re-renders
    do not re-homogenize.
    """
    import json

    kx_values = list(
        kx_values
        if kx_values is not None
        else np.linspace(-0.010, 0.010, 9)
    )
    cell_sizes = list(
        cell_sizes if cell_sizes is not None else [0.0, 0.3, 0.6, 1.0, 1.5]
    )
    cache_path = Path(cache_path) if cache_path else None
    if cache_path is not None and cache_path.is_file():
        rows = json.loads(cache_path.read_text())
        # Accept cache only if it covers the requested grid.
        have = {(round(r["kx"], 6), round(r["cell_size"], 6)) for r in rows}
        need = {
            (round(float(kx), 6), round(float(cs), 6))
            for kx in kx_values
            for cs in cell_sizes
        }
        if need.issubset(have):
            return [
                r
                for r in rows
                if (round(r["kx"], 6), round(r["cell_size"], 6)) in need
            ]

    rows: list[dict[str, float]] = []
    base = base or _parametric_base_case()
    for cs in cell_sizes:
        for kx in kx_values:
            rows.append(
                homogenize_halo_curvature(
                    float(kx),
                    None if float(cs) <= 0 else float(cs),
                    base=base,
                )
            )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(rows, indent=2))
    return rows


def plot_stiffness_vs_curvature_halo(
    rows: list[dict[str, float]] | None = None,
    *,
    theme: CoreTheme = DEFAULT_THEME,
    figsize: tuple[float, float] = (11.0, 8.2),
) -> plt.Figure:
    """Publication board: Eyy & resin Vf vs κ and halo width.

    Primary modulus is ``Eyy`` (channel-direction for uniaxial x-grooves): it
    tracks open/close and halo width cleanly on the fast numpy RVE. Through-
    thickness ``Ezz`` is omitted here — morphed thin meshes understate it.
    """
    if rows is None:
        rows = sweep_halo_curvature_grid()

    kx_vals = np.array(sorted({r["kx"] for r in rows}))
    cs_vals = np.array(sorted({r["cell_size"] for r in rows}))
    by = {(round(r["kx"], 6), round(r["cell_size"], 6)): r for r in rows}

    def series(cs: float, key: str) -> np.ndarray:
        return np.array(
            [by[(round(float(kx), 6), round(float(cs), 6))][key] for kx in kx_vals]
        )

    def series_cs(kx: float, key: str) -> np.ndarray:
        return np.array(
            [by[(round(float(kx), 6), round(float(cs), 6))][key] for cs in cs_vals]
        )

    cs_colors = plt.cm.viridis(np.linspace(0.15, 0.9, max(len(cs_vals), 2)))
    kx_pick = [
        float(kx_vals[0]),
        float(kx_vals[len(kx_vals) // 2]),
        float(kx_vals[-1]),
    ]
    kx_styles = {
        kx_pick[0]: {"color": "#2166ac", "label": f"closed kx={kx_pick[0]:+.3f}"},
        kx_pick[1]: {"color": "#666666", "label": f"flat kx={kx_pick[1]:+.3f}"},
        kx_pick[2]: {"color": "#b2182b", "label": f"open kx={kx_pick[2]:+.3f}"},
    }

    with plt.rc_context(theme.publication_rcparams()):
        fig, axes = plt.subplots(2, 2, figsize=figsize, layout="constrained")
        ax_e, ax_cs, ax_vf, ax_hm = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

        # (a) Eyy vs kx for each cell_size
        for i, cs in enumerate(cs_vals):
            col = cs_colors[i]
            lab = "sharp (no halo)" if cs <= 0 else f"cell_size = {cs:g} mm"
            eyy = series(cs, "Eyy") / 1e9
            ax_e.plot(kx_vals * 1e3, eyy, "-o", color=col, ms=3.5, lw=1.8, label=lab)
        ax_e.axvline(0.0, color="#bbbbbb", lw=0.8)
        ax_e.set_xlabel(r"curvature $k_x$ [$10^{-3}$/mm]")
        ax_e.set_ylabel(r"$E_{yy}$ [GPa]")
        ax_e.set_title(r"$E_{yy}$ vs curvature (halo width as colour)")
        ax_e.grid(True, alpha=0.25)
        ax_e.legend(fontsize=6.5, loc="best", ncol=1)

        # (b) Eyy vs cell_size at closed / flat / open
        for kx in kx_pick:
            sty = kx_styles[kx]
            eyy = series_cs(kx, "Eyy") / 1e9
            ax_cs.plot(cs_vals, eyy, "-o", ms=4, lw=1.8, **sty)
        ax_cs.set_xlabel("halo width cell_size [mm]")
        ax_cs.set_ylabel(r"$E_{yy}$ [GPa]")
        ax_cs.set_title(r"$E_{yy}$ vs halo width (fixed curvature)")
        ax_cs.grid(True, alpha=0.25)
        ax_cs.legend(fontsize=7, loc="best")

        # (c) resin volume fractions vs kx
        for i, cs in enumerate(cs_vals):
            col = cs_colors[i]
            lab = "sharp" if cs <= 0 else f"cs={cs:g}"
            ax_vf.plot(
                kx_vals * 1e3,
                series(cs, "resin_vf"),
                "-",
                color=col,
                lw=1.5,
                label=f"neat ({lab})",
            )
            if cs > 0:
                ax_vf.plot(
                    kx_vals * 1e3,
                    series(cs, "effective_resin_vf"),
                    "--",
                    color=col,
                    lw=1.3,
                    alpha=0.85,
                )
        ax_vf.axvline(0.0, color="#bbbbbb", lw=0.8)
        ax_vf.set_xlabel(r"curvature $k_x$ [$10^{-3}$/mm]")
        ax_vf.set_ylabel("volume fraction")
        ax_vf.set_title("neat resin_vf (solid) · effective_resin_vf (dashed)")
        ax_vf.grid(True, alpha=0.25)
        ax_vf.legend(fontsize=6, ncol=2, loc="best")

        # (d) heatmap Eyy(kx, cell_size)
        Z = np.zeros((len(cs_vals), len(kx_vals)))
        for i, cs in enumerate(cs_vals):
            for j, kx in enumerate(kx_vals):
                Z[i, j] = by[(round(float(kx), 6), round(float(cs), 6))]["Eyy"] / 1e9
        im = ax_hm.imshow(
            Z,
            origin="lower",
            aspect="auto",
            extent=[
                float(kx_vals[0]) * 1e3,
                float(kx_vals[-1]) * 1e3,
                float(cs_vals[0]),
                float(cs_vals[-1]),
            ],
            cmap="RdYlBu_r",
            interpolation="bilinear",
        )
        # Mark sample points
        for cs in cs_vals:
            for kx in kx_vals:
                ax_hm.plot(float(kx) * 1e3, float(cs), "k.", ms=2, alpha=0.35)
        ax_hm.set_xlabel(r"curvature $k_x$ [$10^{-3}$/mm]")
        ax_hm.set_ylabel("halo width cell_size [mm]")
        ax_hm.set_title(r"$E_{yy}$ [GPa]  heatmap")
        cbar = fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
        cbar.set_label(r"$E_{yy}$ [GPa]")

        fig.suptitle(
            "Parametric stiffness: mould curvature × resin-halo width\n"
            r"(top-mouth: $k_x>0$ opens · cell_size $=0$ = sharp kerf · "
            r"$E_{yy}$ tracks resin lattice)",
            fontsize=11,
        )
    return fig


def plot_stiffness_moduli_vs_curvature(
    rows: list[dict[str, float]] | None = None,
    *,
    cell_size: float = 0.6,
    theme: CoreTheme = DEFAULT_THEME,
    figsize: tuple[float, float] = (9.5, 4.2),
) -> plt.Figure:
    """Exx, Eyy, Gxy, Gxz vs kx at one halo width (+ sharp twin)."""
    if rows is None:
        rows = sweep_halo_curvature_grid(
            cell_sizes=[0.0, float(cell_size)],
        )
    kx_vals = np.array(sorted({r["kx"] for r in rows}))
    by = {(round(r["kx"], 6), round(r["cell_size"], 6)): r for r in rows}
    keys = ["Exx", "Eyy", "Gxy", "Gxz"]
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]

    with plt.rc_context(theme.publication_rcparams()):
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        for key, col in zip(keys, colors, strict=True):
            y_h = np.array(
                [
                    by[(round(float(kx), 6), round(float(cell_size), 6))][key] / 1e9
                    for kx in kx_vals
                ]
            )
            y_s = np.array(
                [by[(round(float(kx), 6), 0.0)][key] / 1e9 for kx in kx_vals]
            )
            ax.plot(
                kx_vals * 1e3,
                y_h,
                "-o",
                color=col,
                ms=3.5,
                lw=1.8,
                label=f"{key} + halo",
            )
            ax.plot(
                kx_vals * 1e3,
                y_s,
                "--",
                color=col,
                lw=1.2,
                alpha=0.7,
                label=f"{key} sharp",
            )
        ax.axvline(0.0, color="#bbbbbb", lw=0.8)
        ax.set_xlabel(r"curvature $k_x$ [$10^{-3}$/mm]")
        ax.set_ylabel("modulus [GPa]")
        ax.set_title(
            f"Engineering moduli vs curvature  "
            f"(solid = cell_size {cell_size:g} mm, dashed = sharp)"
        )
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=6.5, ncol=4, loc="best")
    return fig


def render_halo_curvature_figures(
    out_dir: str | Path,
    *,
    base_inp: dict | None = None,
    kx_open: float = _CURVED_PANEL_KX_OPEN,
    kx_closed: float = _CURVED_PANEL_KX_CLOSED,
    dpi: int = 200,
    run_parametric: bool = True,
) -> list[Path]:
    """Write halo on curved_panel open/closed configs to *out_dir*."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    fig = plot_halo_curvature_compose(
        base_inp, kx_open=kx_open, kx_closed=kx_closed
    )
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
        rows = sweep_halo_curvature_grid(cache_path=cache)
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
    import matplotlib.pyplot as plt

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
            "uniform cell_size = 0.6 mm (DIAB H60 scale)",
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
        ("halo_strip_diab_gs30.png", lambda: plot_halo_cross_section_strip(scored_inp)),
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