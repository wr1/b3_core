#!/usr/bin/env python3
"""Render the tapered RVE so the opened-vs-pinched grooves are visible.

For three curvatures (closing, flat, opening) it builds the base RVE's mesh,
shows the foam translucent and the infused resin solid, and writes a PNG per
state plus a combined strip.

    uv run python examples/curved_panel/render.py

Off-screen rendering; falls back to a virtual framebuffer if no display.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyvista as pv

from b3_core.core.mesh import create_grooved_mesh

HERE = Path(__file__).parent
BASE = json.loads((HERE / "base.json").read_text())
STATES = [("closed", -0.012), ("flat", 0.0), ("opened", 0.012)]


def _mesh_for(kx: float):
    return create_grooved_mesh(
        thickness=BASE["thickness"], dx=BASE["dx"], dy=BASE["dy"],
        xcuts=BASE["xgr"], ycuts=BASE["ygr"], madd=tuple(BASE["madd"]),
        tface=0.0, kx=kx, ky=0.0,
    )


def _add(plotter, mesh, title):
    resin = mesh.threshold(0.5, scalars="resin")
    foam = mesh.threshold(0.5, scalars="resin", invert=True)
    if foam.n_cells:
        plotter.add_mesh(foam, color="lightgray", opacity=0.12, show_edges=False)
    if resin.n_cells:
        plotter.add_mesh(resin, color="firebrick", opacity=1.0, show_edges=True)
    plotter.add_text(title, font_size=10)
    # Orthographic look along y (xz-plane): the x-grooves show their tapered
    # trapezoid section without perspective foreshortening.
    plotter.enable_parallel_projection()
    plotter.camera_position = "xz"
    plotter.camera.zoom(1.6)


def main() -> int:
    try:
        pv.start_xvfb()
    except Exception:
        pass  # a real display is fine too
    out = HERE / "img"
    out.mkdir(parents=True, exist_ok=True)

    grid = pv.Plotter(shape=(1, 3), off_screen=True, window_size=(1500, 560))
    for col, (name, kx) in enumerate(STATES):
        mesh = _mesh_for(kx)
        single = pv.Plotter(off_screen=True, window_size=(560, 560))
        _add(single, mesh, f"{name}  kx={kx:+.3f}")
        single.screenshot(str(out / f"groove_{name}.png"))
        single.close()

        grid.subplot(0, col)
        _add(grid, mesh, f"{name}  kx={kx:+.3f}")
    grid.screenshot(str(out / "groove_strip.png"))
    grid.close()

    for name, _ in STATES:
        print(f"wrote {out / f'groove_{name}.png'}")
    print(f"wrote {out / 'groove_strip.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
