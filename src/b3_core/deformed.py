"""Periodic deformed-shape view of the six unit-strain load cases.

Separate from the datasheet: warps the RVE by the true periodic displacement
``u = E.x + w`` for each macroscopic strain (xx, yy, zz, yz, xz, xy) and renders
a 2x3 montage. Because the fluctuation ``w`` is periodic, opposite faces deform
compatibly — the visual check that the periodic BC is doing its job. Built on the
shared :mod:`b3_core.viz` layer (CoreModel, CoreTheme, headless bootstrap).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from b3_core.viz._deps import ensure_headless, require_pyvista
from b3_core.viz.model import CoreModel
from b3_core.viz.theme import DEFAULT_THEME, CoreTheme

logger = logging.getLogger(__name__)

LOAD_CASES = ("xx", "yy", "zz", "yz", "xz", "xy")
_DEFAULT_WARP = 0.3  # unit strain is 100%; scale down so the shape stays readable


def render_deformed_modes(
    case: str | Path | CoreModel,
    out_png: str | Path,
    *,
    warp: float = _DEFAULT_WARP,
    window: tuple[int, int] = (1500, 1000),
    theme: CoreTheme = DEFAULT_THEME,
) -> Path:
    """Render the six periodic deformation modes of a case to ``out_png``.

    ``case`` is a JSON path or a prepared :class:`~b3_core.viz.CoreModel`. Each
    subplot shows the RVE warped by its load case, resin grooves coloured by
    displacement magnitude and the core translucent. Returns the output path.
    """
    pv = require_pyvista()
    model = case if isinstance(case, CoreModel) else CoreModel.from_json(case)
    logger.info("rendering periodic deformation modes for %s", model.name)

    grid = model.mesh.cast_to_unstructured_grid()
    ensure_headless()
    plotter = pv.Plotter(shape=(2, 3), off_screen=True, window_size=window)
    plotter.set_background(theme.background)
    for i, lc in enumerate(LOAD_CASES):
        plotter.subplot(i // 3, i % 3)
        g = grid.copy()
        g["u"] = model.displacements(lc) * 1000.0  # m -> mm
        g["umag_mm"] = np.linalg.norm(g["u"], axis=1)
        warped = g.warp_by_vector("u", factor=warp)
        core = warped.threshold(0.5, scalars="resin", invert=True)
        resin = warped.threshold(0.5, scalars="resin")
        if core.n_cells:
            plotter.add_mesh(core, color=theme.core_color, opacity=theme.core_opacity)
        if resin.n_cells:
            plotter.add_mesh(
                resin,
                scalars="umag_mm",
                cmap=theme.cmap_displacement,
                show_edges=True,
                edge_color=theme.edge_color,
                line_width=theme.edge_width,
                show_scalar_bar=False,
            )
        plotter.add_text(f"strain {lc}", font_size=10, color="black")
        plotter.camera_position = "iso"

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(out_png))
    plotter.close()
    logger.info("wrote %s", out_png)
    return out_png
