"""Off-screen 3D halo render."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from b3_core.viz.halo.sections import mesh_and_field
from b3_core.viz.theme import DEFAULT_THEME, CoreTheme


def render_halo_3d_png(
    inp: dict,
    path: str | Path,
    *,
    theme: CoreTheme = DEFAULT_THEME,
    window_size: tuple[int, int] = (900, 720),
) -> Path:
    """3D view: neat resin solid, foam coloured by ``P(resin)`` at cell centres."""
    from b3_core.viz import geometry
    from b3_core.viz.deps import ensure_headless, require_pyvista

    mesh, mat, field = mesh_and_field(inp)
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
