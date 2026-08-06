#!/usr/bin/env python3
"""Visualise kerf open/close and curved → flattened modelling step (2D side view).

Kinematics:

* **Flattened for FEA** — interval-affine wall morph on the structured RVE:
  kerfs taper with ``hw(z)`` and foam bays become **trapezoidal**. This is the
  mesh homogenization actually solves.
* **Curved (viz)** — that same FEA mesh rolled onto a cylinder
  (``θ = κ·(x − x_mid)``, ``r = R + (z − z_ref)``). Kerf tip bluntness and
  wall taper match the flat panel exactly; only the global shape changes.

    uv run python examples/curved_panel/render.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyvista as pv

from b3_core.core.mesh import create_grooved_mesh

HERE = Path(__file__).parent
BASE = json.loads((HERE / "base.json").read_text())
STATES = [("closed", -0.012), ("flat", 0.0), ("opened", 0.012)]
PAIR_KX_OPEN = 0.012
PAIR_KX_CLOSED = -0.012
SLAB_HALF = 0.5


def _mesh_flat_fea(kx: float) -> pv.UnstructuredGrid:
    """Flattened RVE used by homogenization (morphed walls, trapezoidal foam)."""
    mesh = create_grooved_mesh(
        thickness=BASE["thickness"],
        dx=BASE["dx"],
        dy=BASE["dy"],
        xcuts=BASE["xgr"],
        ycuts=BASE["ygr"],
        madd=tuple(BASE["madd"]),
        tface=0.0,
        kx=kx,
        ky=0.0,
    )
    grid = mesh.cast_to_unstructured_grid()
    for name in mesh.cell_data.keys():
        grid.cell_data[name] = mesh.cell_data[name]
    return grid


def _roll_points_to_cylinder(
    points: np.ndarray,
    kappa: float,
    *,
    x_mid: float,
    z_ref: float,
) -> np.ndarray:
    """Roll material ``(x, z)`` onto a cylinder (preserves kerf taper / tips).

    ``θ = κ (x − x_mid)``, ``r = R + (z − z_ref)`` with ``R = 1/|κ|``.
    Per-block rigid hinging is *not* used — it pinches land gaps harder than
    the interval-affine ``hw(z)`` law and makes kerf tips look needle-sharp
    next to the flat FEA panel. ``κ → 0`` is the identity.
    """
    pts = np.asarray(points, dtype=float)
    if abs(kappa) < 1e-12:
        return pts.copy()
    R = 1.0 / abs(kappa)
    out = pts.copy()
    x = out[:, 0]
    z = out[:, 2]
    theta = kappa * (x - x_mid)
    r = R + (z - z_ref)
    out[:, 0] = x_mid + r * np.sin(theta)
    out[:, 2] = z_ref - R + r * np.cos(theta)
    return out


def _bend_fea_to_cylinder(fea: pv.UnstructuredGrid, kx: float) -> pv.UnstructuredGrid:
    """Roll the flattened FEA mesh onto a cylinder (same tags, moved points)."""
    if abs(kx) < 1e-12:
        return fea
    th = float(BASE["thickness"])
    dx = float(BASE["dx"])
    out = fea.copy(deep=True)
    out.points = _roll_points_to_cylinder(
        np.asarray(out.points, dtype=float),
        kx,
        x_mid=0.5 * dx,
        z_ref=0.5 * th,
    )
    return out


def _mesh_curved_fea(kx: float) -> pv.UnstructuredGrid:
    """Curved-on-mould view: FEA RVE with identical wall taper, rolled onto arc."""
    return _bend_fea_to_cylinder(_mesh_flat_fea(kx), kx)


def _mid_y_slab(mesh: pv.DataSet, half: float = SLAB_HALF) -> pv.DataSet:
    y0 = 0.5 * (mesh.bounds[2] + mesh.bounds[3])
    kept = mesh.clip(normal="y", origin=(0.0, y0 - half, 0.0), invert=False)
    return kept.clip(normal="-y", origin=(0.0, y0 + half, 0.0), invert=False)


def _set_side_camera(plotter: pv.Plotter, mesh: pv.DataSet) -> None:
    plotter.enable_parallel_projection()
    b = mesh.bounds
    cx = 0.5 * (b[0] + b[1])
    cy = 0.5 * (b[2] + b[3])
    cz = 0.5 * (b[4] + b[5])
    sx = max(b[1] - b[0], 1.0)
    sz = max(b[5] - b[4], 1.0)
    dist = max(sx, sz) * 2.2
    plotter.camera.position = (cx, cy - dist, cz)
    plotter.camera.focal_point = (cx, cy, cz)
    plotter.camera.up = (0.0, 0.0, 1.0)
    plotter.camera.parallel_scale = 0.55 * max(sx, sz)


def _add_side_view(plotter: pv.Plotter, mesh: pv.DataSet, title: str) -> None:
    slab = _mid_y_slab(mesh)
    if slab.n_cells == 0:
        slab = mesh
    if "resin" in slab.array_names:
        resin = slab.threshold(0.5, scalars="resin")
        foam = slab.threshold(0.5, scalars="resin", invert=True)
    else:
        resin, foam = None, slab
    if foam is not None and foam.n_cells:
        plotter.add_mesh(
            foam,
            color="gainsboro",
            opacity=1.0,
            show_edges=True,
            edge_color="silver",
            line_width=0.5,
        )
    if resin is not None and resin.n_cells:
        plotter.add_mesh(
            resin,
            color="firebrick",
            opacity=1.0,
            show_edges=True,
            edge_color="darkred",
            line_width=0.6,
        )
    plotter.add_text(title, font_size=11)
    plotter.background_color = "white"
    _set_side_camera(plotter, slab if slab.n_cells else mesh)


def _write_open_flat_closed(out: Path) -> None:
    grid = pv.Plotter(shape=(1, 3), off_screen=True, window_size=(1680, 520))
    for col, (name, kx) in enumerate(STATES):
        mesh = _mesh_flat_fea(kx)
        single = pv.Plotter(off_screen=True, window_size=(640, 480))
        _add_side_view(single, mesh, f"{name}  kx={kx:+.3f}  (flattened RVE)")
        single.screenshot(str(out / f"groove_{name}.png"))
        single.close()
        grid.subplot(0, col)
        _add_side_view(grid, mesh, f"{name}  kx={kx:+.3f}")
    grid.screenshot(str(out / "groove_strip.png"))
    grid.close()


def _write_curved_vs_flat(
    out: Path,
    kx: float,
    *,
    mode: str,
    write_singles: bool = False,
) -> None:
    curved = _mesh_curved_fea(kx)
    flat = _mesh_flat_fea(kx)
    kerf_word = "open" if mode == "open" else "closed"

    if write_singles:
        for tag, mesh, title in (
            ("curved", curved, f"on mould  kx={kx:+.3f}  (FEA mesh rolled onto arc)"),
            ("flattened", flat, f"flattened RVE  kx={kx:+.3f}  (trapezoidal foam)"),
        ):
            pl = pv.Plotter(off_screen=True, window_size=(720, 480))
            _add_side_view(pl, mesh, title)
            pl.screenshot(str(out / f"groove_{tag}.png"))
            pl.close()

    board = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1400, 520))
    board.subplot(0, 0)
    _add_side_view(
        board,
        curved,
        f"1 · curved   kx={kx:+.3f}   same FEA taper, rolled onto mould arc",
    )
    board.subplot(0, 1)
    _add_side_view(
        board,
        flat,
        f"2 · flattened for FEA   trapezoidal foam + {kerf_word} kerf taper",
    )
    name = (
        "groove_curved_vs_flat.png"
        if mode == "open"
        else "groove_curved_vs_flat_closed.png"
    )
    board.screenshot(str(out / name))
    board.close()


def main() -> int:
    try:
        pv.start_xvfb()
    except Exception:
        pass
    out = HERE / "img"
    out.mkdir(parents=True, exist_ok=True)
    _write_open_flat_closed(out)
    _write_curved_vs_flat(out, PAIR_KX_OPEN, mode="open", write_singles=True)
    _write_curved_vs_flat(out, PAIR_KX_CLOSED, mode="closed", write_singles=False)
    for name in (
        "groove_closed",
        "groove_flat",
        "groove_opened",
        "groove_strip",
        "groove_curved",
        "groove_flattened",
        "groove_curved_vs_flat",
        "groove_curved_vs_flat_closed",
    ):
        print(f"wrote {out / f'{name}.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
