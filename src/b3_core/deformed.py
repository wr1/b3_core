"""Periodic deformed-shape view of the six unit-strain load cases.

Separate from the datasheet: solves the RVE with the MFEM backend and warps the
mesh by the true periodic displacement ``u = E.x + w`` for each macroscopic
strain (xx, yy, zz, yz, xz, xy), rendering a 2x3 montage. Because the
fluctuation ``w`` is periodic, opposite faces deform compatibly — the visual
check that the periodic BC is doing its job.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from b3_core.core.mesh import create_grooved_mesh
from b3_core.io import mfem_backend

logger = logging.getLogger(__name__)

# Backend load-case order (see io/fenicsx.LOAD_CASES).
LOAD_CASES = ("xx", "yy", "zz", "yz", "xz", "xy")
_DEFAULT_WARP = 0.3  # unit strain is 100%; scale down so the shape stays readable


def render_deformed_modes(
    json_path: str | Path,
    out_png: str | Path,
    *,
    warp: float = _DEFAULT_WARP,
    window: tuple[int, int] = (1500, 1000),
) -> Path:
    """Render the six periodic deformation modes of a case to ``out_png``.

    Each subplot shows the RVE warped by its load case's displacement field —
    resin grooves solid (coloured by displacement magnitude), core translucent.
    Returns the output path.
    """
    import pyvista as pv

    json_path = Path(json_path)
    inp = json.loads(json_path.read_text())
    mesh = create_grooved_mesh(
        thickness=inp["thickness"], dx=inp["dx"], dy=inp["dy"],
        xcuts=inp["xgr"], ycuts=inp["ygr"], madd=tuple(inp.get("madd", [0])),
        tface=(inp.get("face") or {}).get("thickness", 0.0),
        kx=(inp.get("curvature") or {}).get("kx", 0.0),
        ky=(inp.get("curvature") or {}).get("ky", 0.0),
    )
    logger.info("running MFEM backend for the periodic displacement fields")
    details = mfem_backend.runmfem(
        mesh, inp["resin"], inp["core"], inp.get("face"), return_details=True
    )

    # Same vertex order the backend used; grid in mm, displacement in metres.
    grid = mesh.cast_to_unstructured_grid()
    try:
        pv.start_xvfb()
    except Exception:  # pragma: no cover - display already present / unsupported
        pass

    plotter = pv.Plotter(shape=(2, 3), off_screen=True, window_size=window)
    plotter.set_background("white")
    for i, case in enumerate(LOAD_CASES):
        plotter.subplot(i // 3, i % 3)
        g = grid.copy()
        g["u"] = np.asarray(details.displacements[case]) * 1000.0  # m -> mm
        g["umag_mm"] = np.linalg.norm(g["u"], axis=1)
        warped = g.warp_by_vector("u", factor=warp)
        core = warped.threshold(0.5, scalars="resin", invert=True)
        resin = warped.threshold(0.5, scalars="resin")
        if core.n_cells:
            plotter.add_mesh(core, color="#d9d9d9", opacity=0.1)
        if resin.n_cells:
            plotter.add_mesh(
                resin, scalars="umag_mm", cmap="viridis", show_edges=True,
                edge_color="#333333", line_width=0.3, show_scalar_bar=False,
            )
        plotter.add_text(f"strain {case}", font_size=10, color="black")
        plotter.camera_position = "iso"

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(out_png))
    plotter.close()
    logger.info("wrote %s", out_png)
    return out_png
