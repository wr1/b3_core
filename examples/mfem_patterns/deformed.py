#!/usr/bin/env python3
"""Deformed-shape viz to review the MFEM periodic boundary conditions.

For one pattern it solves the six unit-strain corrector problems with MFEM,
warps the RVE by the true periodic displacement u = E.x + w, and renders:

  * deformed_modes.png  - the six load cases (foam translucent, resin solid)
  * periodic_tiling.png - the xy-shear cell tiled 2x2 by the *deformed* lattice
    vectors; because the fluctuation w is periodic, the tiles abut seamlessly,
    which is the visual check that the periodic BC is doing its job.

    uv run python examples/mfem_patterns/deformed.py [pattern]

Requires the optional MFEM stack (`uv sync --extra mfem`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

from b3_core.core.mesh import create_grooved_mesh
from b3_core.io.mfem_backend import _macro_strain, runmfem

HERE = Path(__file__).parent
LOAD_CASES = ["xx", "yy", "zz", "yz", "xz", "xy"]
WARP = 0.3  # unit strain is 100%; scale down so the shape stays readable


def _grid_with_disp(pattern: str):
    case = json.loads((HERE / f"{pattern}.json").read_text())
    mesh = create_grooved_mesh(
        thickness=case["thickness"], dx=case["dx"], dy=case["dy"],
        xcuts=case["xgr"], ycuts=case["ygr"], madd=tuple(case["madd"]),
        tface=case.get("face", {}).get("thickness", 0.0),
    )
    result = runmfem(mesh, case["resin"], case["core"], None, return_details=True)
    # Same vertex order the backend used; grid stays in mm, displacement in metres.
    grid = mesh.cast_to_unstructured_grid()
    return grid, result


def _deformed(grid, disp_m):
    g = grid.copy()
    g["u"] = disp_m * 1000.0  # metres -> mm to match the grid
    return g.warp_by_vector("u", factor=WARP)


def _add(plotter, mesh, title=""):
    foam = mesh.threshold(0.5, scalars="resin", invert=True)
    resin = mesh.threshold(0.5, scalars="resin")
    if foam.n_cells:
        plotter.add_mesh(foam, color="lightgray", opacity=0.1)
    if resin.n_cells:
        plotter.add_mesh(resin, color="firebrick", show_edges=True)
    if title:
        plotter.add_text(title, font_size=10)
    plotter.camera_position = "iso"


def main() -> int:
    pattern = sys.argv[1] if len(sys.argv) > 1 else "crossed"
    try:
        pv.start_xvfb()
    except Exception:
        pass
    out = HERE / "img"
    out.mkdir(parents=True, exist_ok=True)
    grid, result = _grid_with_disp(pattern)

    # The six deformation modes.
    modes = pv.Plotter(shape=(2, 3), off_screen=True, window_size=(1500, 1000))
    for i, case in enumerate(LOAD_CASES):
        modes.subplot(i // 3, i % 3)
        _add(modes, _deformed(grid, result.displacements[case]), f"{pattern}: {case}")
    modes.screenshot(str(out / "deformed_modes.png"))
    modes.close()

    # Periodicity check: tile the xy-shear cell by the deformed lattice vectors.
    dx = grid.bounds[1] - grid.bounds[0]
    dy = grid.bounds[3] - grid.bounds[2]
    E = _macro_strain("xy")
    warped = _deformed(grid, result.displacements["xy"])
    # deformed in-plane lattice vectors: L + WARP * (E . L), in mm
    Lx = np.array([dx, 0, 0]) + WARP * (E @ np.array([dx, 0, 0]))
    Ly = np.array([0, dy, 0]) + WARP * (E @ np.array([0, dy, 0]))
    tiling = pv.Plotter(off_screen=True, window_size=(900, 900))
    for a in (0, 1):
        for b in (0, 1):
            shift = a * Lx + b * Ly
            _add(tiling, warped.copy().translate(shift, inplace=True))
    tiling.add_text("xy shear - 2x2 deformed-lattice tiling (periodic w => seamless)", font_size=9)
    tiling.camera_position = "xy"
    tiling.screenshot(str(out / "periodic_tiling.png"))
    tiling.close()

    print(f"wrote {out / 'deformed_modes.png'}")
    print(f"wrote {out / 'periodic_tiling.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
