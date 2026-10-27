"""Curvature halo figures and stiffness boards."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from b3_core.viz.halo.sections import mesh_and_field, sample_halo_plane
from b3_core.viz.theme import DEFAULT_THEME, CoreTheme


def _sweep_grid(*args, **kwargs):
    from b3_core.sweep.curvature_grid import sweep_halo_curvature_grid

    return sweep_halo_curvature_grid(*args, **kwargs)


# Same open/close κ as examples/curved_panel/render.py (kerf open & close docs).
CURVED_PANEL_KX_OPEN = 0.012
CURVED_PANEL_KX_CLOSED = -0.012


def _default_halo_curvature_case(*, cell_size: float = 0.6) -> dict:
    """``examples/curved_panel/base.json`` geometry + resin halo.

    Top-mouth grooves (``depth < 0``): ``kx > 0`` opens, ``kx < 0`` closes —
    identical to the kerf open & close documentation figures.
    """
    # Prefer the live example file so docs and viz cannot drift.
    roots = [
        Path(__file__).resolve().parents[4] / "examples" / "curved_panel" / "base.json",
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

    from b3_core.core.mesh import hw_at

    mesh, mat, field = mesh_and_field(inp)
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
            hw = np.array([hw_at(hw0, depth, slope, float(z), th) for z in zs])
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
    kx_open: float = CURVED_PANEL_KX_OPEN,
    kx_closed: float = CURVED_PANEL_KX_CLOSED,
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
        cbar = fig.colorbar(
            sm, ax=axes, fraction=0.025, pad=0.02, ticks=[0.1, 0.3, 0.5, 0.7, 0.9]
        )
        cbar.set_label("P(resin) in halo rim  (blue→white→red)")
        cbar.ax.set_yticklabels(["0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1"])
        legend_handles = [
            Patch(
                facecolor=theme.halo_resin_color(), label="neat resin (morphed kerf)"
            ),
            Patch(facecolor="#ef8a62", label="halo rim (graded P)"),
            Patch(facecolor="#e8e8e8", edgecolor="#888", label="intact foam (P ≈ 0)"),
            Line2D([0], [0], color=theme.edge_color, lw=1.4, label="kerf wall hw(z)"),
            Line2D(
                [0],
                [0],
                color="#2166ac",
                lw=1.2,
                ls=":",
                label="halo reach (wall + cell_size)",
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
    kx_open: float = CURVED_PANEL_KX_OPEN,
    kx_closed: float = CURVED_PANEL_KX_CLOSED,
    theme: CoreTheme = DEFAULT_THEME,
    figsize: tuple[float, float] = (9.0, 4.2),
) -> tuple[plt.Figure, plt.Axes]:
    """P(resin) vs stand-off from the *local* wall at mouth and root.

    Same survival decay from open vs closed walls — the wall moves with κ;
    ``S(d)`` does not. Uses curved_panel open/closed κ by default.
    """
    from b3_core.core.mesh import hw_at
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
            hw = hw_at(hw0, depth, slope, z, th)
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
    from b3_core.core.mesh import MIN_HW

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
        hw = np.where(inside, max(MIN_HW, hw0), hw0)
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
    from b3_core.core.mesh import hw_at

    for g_axis, c0, hw0, slope, depth in field.grooves:
        if g_axis != 0:
            continue
        if depth > 0:
            zs = np.linspace(0.0, float(depth), 120)
        else:
            zs = np.linspace(th + float(depth), th, 120)
        hw = np.array([hw_at(hw0, depth, slope, float(z), th) for z in zs])
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

    from b3_core.core.mesh import hw_at

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
    mesh, mat, field = mesh_and_field(inp)
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
    hw_a = hw_at(hw0, depth, slope, z_lo_w, th)
    hw_b = hw_at(hw0, depth, slope, z_hi_w, th)
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
        p_grid, phase = _sample_p_plane(mesh, mat, p_fn, 0, 2, 1, yc, ux, uz)
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
        hw = np.array([hw_at(hw0, depth, slope, float(z), th) for z in zs])
        ax.plot(c0 - hw, zs, color=theme.edge_color, ls="-", lw=1.6)
        ax.plot(c0 + hw, zs, color=theme.edge_color, ls="-", lw=1.6)
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        return p_grid, phase, ux, uz, ext

    p_correct = field.resin_probability

    def p_flat(pts):
        return _resin_probability_flat_walls(field, pts)

    with plt.rc_context(theme.publication_rcparams()):
        fig = plt.figure(figsize=figsize, layout="constrained")
        gs = fig.add_gridspec(
            2, 3, height_ratios=[1.05, 1.0], width_ratios=[1.0, 1.0, 1.0]
        )
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
        hw_mid = hw_at(hw0, depth, slope, z_mid, th)
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
        p_bad, phase_bad = _sample_p_plane(mesh, mat, p_flat, 0, 2, 1, yc, ux, uz)
        p_ok2, phase_ok2 = _sample_p_plane(mesh, mat, p_correct, 0, 2, 1, yc, ux, uz)
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
        hw = np.array([hw_at(hw0, depth, slope, float(z), th) for z in zs])
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
                [0],
                [0],
                color="#888888",
                lw=1.4,
                ls=":",
                label="rectangular wall (ghost)",
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
        rows = _sweep_grid()

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
        rows = _sweep_grid(
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


def plot_sign_schematic(
    *,
    kx: float = 0.004,
    pitch: float = 10.0,
    thickness: float = 20.0,
    width: float = 3.0,
    depth: float = 8.0,
    theme: CoreTheme = DEFAULT_THEME,
    figsize: tuple[float, float] = (7.2, 3.2),
) -> plt.Figure:
    """Two families at one positive k. Each title is that mouth's ``opens_for``.

    Wall polylines use the same ``hw(z)`` law as the halo curvature figures.
    ``k > 0`` puts the centre of curvature on the mould side of ``z = 0``.
    """
    from b3_core.core.mesh import hw_at

    hw0 = 0.5 * float(width)
    panels = (
        ("bottom", "k<0", float(depth)),
        ("top", "k>0", -float(depth)),
    )
    with plt.rc_context(theme.publication_rcparams()):
        fig, axes = plt.subplots(
            1, 2, figsize=figsize, layout="constrained", sharey=True
        )
        for ax, (mouth, opens_for, signed_depth) in zip(axes, panels, strict=True):
            slope = -float(np.sign(signed_depth)) * float(kx) * float(pitch) / 2.0
            if signed_depth > 0.0:
                zs = np.linspace(0.0, signed_depth, 80)
            else:
                zs = np.linspace(thickness + signed_depth, thickness, 80)
            half = np.array(
                [hw_at(hw0, signed_depth, slope, float(z), thickness) for z in zs]
            )
            centre = float(pitch)
            ax.plot(centre - half, zs, color=theme.edge_color, lw=1.6)
            ax.plot(centre + half, zs, color=theme.edge_color, lw=1.6)
            ax.axhline(0.0, color="#888888", lw=0.6, ls=":")
            ax.set_title(f"{mouth} mouth · opens for {opens_for}", fontsize=9)
            ax.set_xlabel("x [mm]")
        axes[0].set_ylabel("z [mm]  (0 = mould)")
        fig.suptitle(
            f"k = {float(kx):g} /mm  ·  centre on the mould side of z = 0",
            fontsize=10,
        )
    return fig
