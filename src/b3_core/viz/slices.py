"""Orthogonal cross-section cuts (plan / top / side) with the mesh overlaid.

Matplotlib, so it is robust on any machine (no GL). Samples the phase on each cut
plane via ``find_containing_cell`` — exact for the tapered/curved grooves — and
draws the structured-grid lines as the mesh.
"""

from __future__ import annotations

import numpy as np

from b3_core.viz import geometry
from b3_core.viz.theme import DEFAULT_THEME, FACE, CoreTheme


def _lin(lo, hi, length, ref, px):
    n = int(np.clip(px * (hi - lo) / ref, 40, px))
    eps = 1e-4 * max(hi - lo, ref)
    return np.linspace(lo + eps, hi - eps, n)


def _draw_panel(ax, grid, extent, cmap, norm, lines_u, lines_v, title):
    ax.imshow(
        grid,
        origin="lower",
        extent=extent,
        cmap=cmap,
        norm=norm,
        aspect="equal",
        interpolation="nearest",
    )
    for u in lines_u:
        ax.axvline(u, color="0.35", lw=0.15, alpha=0.45)
    for v in lines_v:
        ax.axhline(v, color="0.35", lw=0.15, alpha=0.45)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)


def plot_orthogonal_cuts(model, *, px: int = 240, theme: CoreTheme = DEFAULT_THEME):
    """Render plan + two side cuts coloured by phase, with the mesh overlaid.

    Returns ``(fig, aspect)`` where ``aspect`` is width/height (for layout).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    mesh = model.mesh
    mat = model.material_codes
    centers = mesh.cell_centers().points
    xv, yv, zv = model.axis_vectors
    x0, x1, y0, y1, z0, z1 = xv[0], xv[-1], yv[0], yv[-1], zv[0], zv[-1]
    Lx, Ly, Lz = x1 - x0, y1 - y0, z1 - z0
    ref = max(Lx, Ly, Lz)

    zc = geometry.best_slab(mesh, mat, 2, centers)
    yc = geometry.best_slab(mesh, mat, 1, centers)
    xc = geometry.best_slab(mesh, mat, 0, centers)
    ux = _lin(x0, x1, Lx, ref, px)
    uy = _lin(y0, y1, Ly, ref, px)
    uz = _lin(z0, z1, Lz, ref, px)

    plan = geometry.sample_plane(mesh, mat, 0, 1, 2, zc, ux, uy)  # z=zc : x,y
    top = geometry.sample_plane(mesh, mat, 0, 2, 1, yc, ux, uz)  # y=yc : x,z
    side = geometry.sample_plane(mesh, mat, 2, 1, 0, xc, uz, uy)  # x=xc : z,y

    cmap, norm = theme.phase_cmap()
    fig = plt.figure(figsize=((Lz + Lx) / ref * 4.2, (Lz + Ly) / ref * 4.2))
    gs = fig.add_gridspec(
        2, 2, width_ratios=[Lz, Lx], height_ratios=[Lz, Ly], hspace=0.28, wspace=0.22
    )
    fig.add_subplot(gs[0, 0]).axis("off")
    ax_top = fig.add_subplot(gs[0, 1])
    ax_side = fig.add_subplot(gs[1, 0])
    ax_plan = fig.add_subplot(gs[1, 1])

    _draw_panel(ax_top, top, (x0, x1, z0, z1), cmap, norm, xv, zv, f"top  y={yc:.3g}")
    _draw_panel(
        ax_side, side, (z0, z1, y0, y1), cmap, norm, zv, yv, f"side  x={xc:.3g}"
    )
    _draw_panel(
        ax_plan, plan, (x0, x1, y0, y1), cmap, norm, xv, yv, f"plan  z={zc:.3g}"
    )

    ax_plan.axvline(xc, color=theme.cut_line, lw=0.9, ls="--")
    ax_plan.axhline(yc, color=theme.cut_line, lw=0.9, ls="--")
    ax_plan.plot([xc], [yc], "+", color=theme.cut_line, ms=8, mew=1.4)

    handles = [
        Patch(facecolor=theme.core_color, edgecolor="0.4", label="core"),
        Patch(facecolor=theme.resin_color, edgecolor="0.4", label="resin"),
    ]
    if (mat == FACE).any():
        handles.append(Patch(facecolor=theme.face_color, edgecolor="0.4", label="face"))
    fig.legend(handles=handles, loc="lower left", fontsize=6, frameon=False, ncol=3)

    w, h = fig.get_size_inches()
    return fig, float(w / h)
