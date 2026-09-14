#!/usr/bin/env python3
"""Render each groove pattern: translucent foam core + solid resin channels.

Builds the mesh for every pattern case, shows the foam at low opacity and the
infused resin channels solid, and writes a PNG per pattern plus a combined 2x2
gallery used in the README.

    uv run python examples/mfem_patterns/render.py

Off-screen rendering; falls back to a virtual framebuffer if no display.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyvista as pv

from b3_core.core.mesh import create_grooved_mesh

HERE = Path(__file__).parent
CASES = ["plain", "uniaxial", "crossed", "two_sided"]


def _mesh_for(name: str):
    case = json.loads((HERE / f"{name}.json").read_text())
    return create_grooved_mesh(
        thickness=case["thickness"], dx=case["dx"], dy=case["dy"],
        xcuts=case["xgr"], ycuts=case["ygr"], madd=tuple(case["madd"]),
        tface=case.get("face", {}).get("thickness", 0.0),
    )


def _add(plotter, mesh, title):
    foam = mesh.threshold(0.5, scalars="resin", invert=True)
    resin = mesh.threshold(0.5, scalars="resin")
    if foam.n_cells:
        plotter.add_mesh(foam, color="lightgray", opacity=0.08, show_edges=False)
    if resin.n_cells:
        plotter.add_mesh(resin, color="firebrick", opacity=1.0, show_edges=True)
    plotter.add_text(title, font_size=10)
    plotter.camera_position = "iso"
    plotter.camera.azimuth = -45
    plotter.camera.elevation = -10


def main() -> int:
    try:
        pv.start_xvfb()
    except Exception:
        pass
    out = HERE / "img"
    out.mkdir(parents=True, exist_ok=True)

    gallery = pv.Plotter(shape=(2, 2), off_screen=True, window_size=(1100, 1000))
    for i, name in enumerate(CASES):
        mesh = _mesh_for(name)

        single = pv.Plotter(off_screen=True, window_size=(620, 560))
        _add(single, mesh, name)
        single.screenshot(str(out / f"pattern_{name}.png"))
        single.close()

        gallery.subplot(i // 2, i % 2)
        _add(gallery, mesh, name)
    gallery.screenshot(str(out / "gallery.png"))
    gallery.close()

    for name in CASES:
        print(f"wrote {out / f'pattern_{name}.png'}")
    print(f"wrote {out / 'gallery.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
