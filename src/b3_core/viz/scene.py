"""CoreScene — a fluent pyvista 3D scene for grooved cores.

Each ``add_*`` returns ``self`` for chaining and reads its colours from the
shared :class:`CoreTheme`::

    CoreScene(model).add_phases().add_mesh_edges().screenshot("core.png")

The plotter is created lazily, off-screen by default (headless / publication), so
``screenshot`` works without a display. Pass ``off_screen=False`` for a native
window (``.show()``) or use ``.serve()`` to export an interactive HTML viewer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from b3_core.viz import geometry, tensor
from b3_core.viz._deps import ensure_headless, require_pyvista, require_trame
from b3_core.viz.theme import DEFAULT_THEME, CoreTheme


def _plane_origin(mesh, axis: str, value: float):
    c = list(mesh.center)
    c["xyz".index(axis)] = value
    return c


class CoreScene:
    def __init__(
        self,
        model,
        *,
        theme: CoreTheme = DEFAULT_THEME,
        off_screen: bool = True,
        window_size: tuple[int, int] = (1000, 850),
    ):
        self.model = model
        self.theme = theme
        self.off_screen = off_screen
        self.window_size = window_size
        self._plotter = None
        self.actors: dict[str, object] = {}

    @property
    def plotter(self):
        if self._plotter is None:
            pv = require_pyvista()
            if self.off_screen:
                ensure_headless()
            self._plotter = pv.Plotter(
                off_screen=self.off_screen, window_size=list(self.window_size)
            )
            self._plotter.set_background(self.theme.background)
        return self._plotter

    # -- phase geometry -----------------------------------------------------
    def _add_phase_meshes(self, phases: dict, *, edges: bool, prefix: str = "") -> None:
        t = self.theme
        if phases["core"].n_cells:
            self.actors[prefix + "core"] = self.plotter.add_mesh(
                phases["core"], color=t.core_color, opacity=t.core_opacity,
                show_edges=edges, edge_color=t.edge_color,
            )
        if phases["face"].n_cells:
            self.actors[prefix + "face"] = self.plotter.add_mesh(
                phases["face"], color=t.face_color, opacity=t.face_opacity,
            )
        if phases["resin"].n_cells:
            self.actors[prefix + "resin"] = self.plotter.add_mesh(
                phases["resin"], color=t.resin_color, show_edges=True,
                edge_color=t.edge_color, line_width=t.edge_width,
            )

    def add_phases(self, *, edges: bool = False) -> "CoreScene":
        """Core (translucent), resin grooves (solid) and face skin."""
        self._add_phase_meshes(
            geometry.split_phases(self.model.mesh, self.model.material_codes), edges=edges
        )
        return self

    def add_mesh_edges(self, *, color: str | None = None) -> "CoreScene":
        """Overlay the full FE mesh as a wireframe."""
        self.actors["edges"] = self.plotter.add_mesh(
            self.model.mesh, style="wireframe",
            color=color or self.theme.edge_color,
            line_width=self.theme.edge_width, opacity=0.3,
        )
        return self

    def add_slice_planes(self, *, x=None, y=None, z=None) -> "CoreScene":
        """Orthogonal slices coloured by phase (x/y/z give the cut coordinate)."""
        view = self.model.mesh.copy()
        view.cell_data["__phase"] = self.model.material_codes
        for axis, val in (("x", x), ("y", y), ("z", z)):
            if val is None:
                continue
            sl = view.slice(normal=axis, origin=_plane_origin(view, axis, val))
            self.actors[f"slice_{axis}"] = self.plotter.add_mesh(
                sl, scalars="__phase", cmap=self.theme.phase_colors(), clim=[0, 2],
                show_scalar_bar=False, show_edges=True,
                edge_color=self.theme.edge_color, line_width=0.2,
            )
        return self

    def add_cutaway(self, axis: str = "x", frac: float = 0.5, *, edges: bool = False) -> "CoreScene":
        """Phases clipped at a plane to reveal the internal groove nesting."""
        mesh = self.model.mesh
        lo, hi = mesh.bounds[2 * "xyz".index(axis)], mesh.bounds[2 * "xyz".index(axis) + 1]
        origin = _plane_origin(mesh, axis, lo + frac * (hi - lo))
        phases = geometry.split_phases(mesh, self.model.material_codes)
        clipped = {k: (v.clip(normal=axis, origin=origin) if v.n_cells else v)
                   for k, v in phases.items()}
        self._add_phase_meshes(clipped, edges=edges, prefix="cut_")
        return self

    def add_deformation(self, case: str, *, warp: float = 0.3) -> "CoreScene":
        """Warp the RVE by load case ``case``'s periodic displacement u = E.x + w."""
        grid = self.model.mesh.cast_to_unstructured_grid()
        grid["u"] = self.model.displacements(case) * 1000.0  # m -> mm
        grid["umag_mm"] = np.linalg.norm(grid["u"], axis=1)
        warped = grid.warp_by_vector("u", factor=warp)
        core = warped.threshold(0.5, scalars="resin", invert=True)
        resin = warped.threshold(0.5, scalars="resin")
        if core.n_cells:
            self.plotter.add_mesh(core, color=self.theme.core_color, opacity=self.theme.core_opacity)
        if resin.n_cells:
            self.actors["deformed"] = self.plotter.add_mesh(
                resin, scalars="umag_mm", cmap=self.theme.cmap_displacement,
                show_edges=True, edge_color=self.theme.edge_color,
                line_width=self.theme.edge_width, show_scalar_bar=False,
            )
        return self

    def add_modulus_surface(self, C=None, *, kind: str = "E") -> "CoreScene":
        """Directional Young's-modulus surface from the effective stiffness."""
        C = self.model.stiffness if C is None else C
        surf = tensor.modulus_surface(C, kind=kind)
        self.actors["modulus"] = self.plotter.add_mesh(
            surf, scalars="value_GPa", cmap=self.theme.cmap_modulus,
            scalar_bar_args={"title": "E [GPa]"},
        )
        return self

    # -- camera / interaction ----------------------------------------------
    def isometric(self) -> "CoreScene":
        self.plotter.view_isometric()
        return self

    def add_axes(self) -> "CoreScene":
        self.plotter.add_axes(line_width=2)
        return self

    def add_phase_toggles(self) -> "CoreScene":
        """Checkbox widgets to toggle each phase (interactive sessions)."""
        for i, name in enumerate(("core", "resin", "face")):
            actor = self.actors.get(name)
            if actor is None:
                continue

            def _cb(state, a=actor):
                a.SetVisibility(state)

            self.plotter.add_checkbox_button_widget(
                _cb, value=True, position=(10.0, 10.0 + 35 * i), size=25,
            )
        return self

    # -- terminals ----------------------------------------------------------
    def show(self):
        if self.off_screen:
            raise RuntimeError("scene is off_screen; pass off_screen=False to .show()")
        return self.plotter.show()

    def serve(self, html_path: str | Path):
        """Export a self-contained interactive HTML viewer (needs the trame extra)."""
        require_trame()
        html_path = Path(html_path)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        self.plotter.export_html(str(html_path))
        return html_path

    def screenshot(self, path: str | Path, *, scale: int = 1):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.plotter.screenshot(str(path), scale=scale)
        return path

    def close(self) -> None:
        if self._plotter is not None:
            self._plotter.close()
            self._plotter = None
