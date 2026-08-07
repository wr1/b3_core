"""GroovedCoreView — the high-level, one-call entry point for grooved-core viz.

Wraps a :class:`CoreModel` and exposes publication helpers (one figure per
aspect) plus a composite ``gallery`` board and interactive ``show`` / ``serve``::

    GroovedCoreView.from_json("case.json").gallery("board.pdf")
    GroovedCoreView.from_json("case.json").show()        # native window
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib

from b3_core.viz import slices, tensorplot
from b3_core.viz.model import CoreModel
from b3_core.viz.scene import CoreScene
from b3_core.viz.theme import DEFAULT_THEME, CoreTheme

matplotlib.use("Agg")


class GroovedCoreView:
    def __init__(self, model: CoreModel, *, theme: CoreTheme = DEFAULT_THEME):
        self.model = model
        self.theme = theme

    @classmethod
    def from_json(cls, path, *, theme: CoreTheme = DEFAULT_THEME) -> "GroovedCoreView":
        return cls(CoreModel.from_json(path), theme=theme)

    @classmethod
    def from_dict(
        cls, inp: dict, *, theme: CoreTheme = DEFAULT_THEME, **kw
    ) -> "GroovedCoreView":
        return cls(CoreModel.from_dict(inp, **kw), theme=theme)

    def scene(self, *, off_screen: bool = True, **kw) -> CoreScene:
        return CoreScene(self.model, theme=self.theme, off_screen=off_screen, **kw)

    # -- single-aspect publication figures ---------------------------------
    def geometry_png(self, path, *, cutaway: bool = False, edges: bool = False):
        s = self.scene().add_phases(edges=edges)
        if cutaway:
            s.add_cutaway("x", 0.5)
        s.add_axes().isometric().screenshot(path)
        s.close()
        return Path(path)

    def mesh_png(self, path):
        s = self.scene().add_phases().add_mesh_edges().add_axes().isometric()
        s.screenshot(path)
        s.close()
        return Path(path)

    def slices_png(self, path):
        import matplotlib.pyplot as plt

        fig, _ = slices.plot_orthogonal_cuts(self.model, theme=self.theme)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return Path(path)

    def modulus_surface_png(self, path, *, kind: str = "E"):
        s = self.scene().add_modulus_surface(kind=kind).add_axes().isometric()
        s.screenshot(path)
        s.close()
        return Path(path)

    def deformation_png(self, path, case: str = "xy", *, warp: float = 0.3):
        s = self.scene().add_deformation(case, warp=warp).add_axes().isometric()
        s.screenshot(path)
        s.close()
        return Path(path)

    def stiffness_heatmap_png(self, path):
        import matplotlib.pyplot as plt

        ax = tensorplot.plot_stiffness_heatmap(self.model.stiffness, theme=self.theme)
        ax.figure.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(ax.figure)
        return Path(path)

    def modulus_polar_png(self, path):
        import matplotlib.pyplot as plt

        axes = tensorplot.plot_modulus_polar(self.model.stiffness, theme=self.theme)
        fig = axes[0].figure
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return Path(path)

    # -- composite board ----------------------------------------------------
    def gallery(self, out, *, fmt: str | None = None, workdir=None):
        """Composite multi-panel board: geometry, cutaway, modulus surface,
        slices, stiffness heatmap and directional-modulus polar, with a title
        banner of the engineering constants. Writes ``out`` (PNG or PDF)."""
        import matplotlib.pyplot as plt

        out = Path(out)
        root = (
            Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="b3core_viz_"))
        )
        root.mkdir(parents=True, exist_ok=True)

        panels = [
            ("3D geometry", self.geometry_png(root / "geom.png")),
            (
                "cutaway (internal architecture)",
                self.geometry_png(root / "cut.png", cutaway=True),
            ),
            ("directional modulus  E(n)", self.modulus_surface_png(root / "mod.png")),
            ("orthogonal cuts + mesh", self.slices_png(root / "cuts.png")),
            ("effective stiffness", self.stiffness_heatmap_png(root / "heat.png")),
            (
                "directional modulus  E(theta)",
                self.modulus_polar_png(root / "polar.png"),
            ),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
        for ax, (title, img) in zip(axes.ravel(), panels, strict=True):
            ax.imshow(plt.imread(img))
            ax.set_title(title, fontsize=10)
            ax.axis("off")

        ec = self.model.engineering_constants
        g = self.model.geom
        banner = (
            f"Grooved core — {self.model.name}        "
            f"E = ({ec['E_x'] / 1e9:.2f}, {ec['E_y'] / 1e9:.2f}, {ec['E_z'] / 1e9:.2f}) GPa     "
            f"G = ({ec['G_xy'] / 1e9:.2f}, {ec['G_xz'] / 1e9:.2f}, {ec['G_yz'] / 1e9:.2f}) GPa     "
            f"nu = ({ec['nu_xy']:.2f}, {ec['nu_xz']:.2f}, {ec['nu_yz']:.2f})     "
            f"resin Vf = {g['resin_vf']:.3f}     rho = {g['rho_infused']:.0f} kg/m3"
        )
        fig.suptitle(banner, fontsize=11, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fmt = fmt or out.suffix.lstrip(".") or "png"
        fig.savefig(out, dpi=160, format=fmt)
        plt.close(fig)
        return out

    # -- interactive --------------------------------------------------------
    def show(self, **scene_kw):
        """Open a native interactive window (needs a display)."""
        s = self.scene(off_screen=False, **scene_kw)
        s.add_phases().add_axes().add_phase_toggles()
        return s.show()

    def serve(self, html_path, **scene_kw):
        """Export a self-contained interactive HTML viewer (needs trame extra)."""
        s = self.scene(**scene_kw).add_phases().add_axes()
        return s.serve(html_path)
