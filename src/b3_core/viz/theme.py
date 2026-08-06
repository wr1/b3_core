"""Single source of truth for grooved-core visualization styling.

Every renderer in :mod:`b3_core.viz` — the 3D ``CoreScene``, the 2D matplotlib
slice/tensor panels and the deformed-mode montage — reads its colours, colormaps
and opacities from one :class:`CoreTheme`, so the look never drifts between the
datasheet, the interactive scene and the publication gallery.
"""

from __future__ import annotations

from dataclasses import dataclass

# Per-cell material phase codes (order matches CoreTheme.phase_colors).
CORE, RESIN, FACE = 0, 1, 2
PHASE_NAMES = ("core", "resin", "face")


@dataclass(frozen=True)
class CoreTheme:
    """Named colours / colormaps shared across all grooved-core renderers."""

    core_color: str = "#d9d9d9"        # lightweight foam / balsa core
    resin_color: str = "#2ca7a0"       # resin-infused grooves (teal)
    face_color: str = "#d8b274"        # optional face skin (tan)
    edge_color: str = "#333333"        # FE mesh / cell edges
    cut_line: str = "#39d0ff"          # slice-plane indicator (cyan)
    background: str = "white"

    cmap_displacement: str = "viridis"  # |u| on the deformed modes
    cmap_modulus: str = "turbo"         # directional Young's modulus surface
    cmap_stiffness: str = "coolwarm"    # signed 6x6 C_eff heatmap

    core_opacity: float = 0.12          # translucent so internal grooves show
    face_opacity: float = 0.55
    edge_width: float = 0.3

    def phase_colors(self) -> list[str]:
        """Discrete colour list indexed by phase code (core, resin, face)."""
        return [self.core_color, self.resin_color, self.face_color]

    def phase_cmap(self):
        """ListedColormap + BoundaryNorm over the three phase codes (NaN = white)."""
        from matplotlib.colors import BoundaryNorm, ListedColormap

        cmap = ListedColormap(self.phase_colors())
        cmap.set_bad("white")
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
        return cmap, norm

    def halo_cmap(self, *, n: int = 256):
        """Clear red–white–blue scale for halo ``P(resin)`` in [0, 1].

        Pale blue = intact foam (``P → 0``), white mid-band, red = resin-rich
        (``P → 1``). Pale foam keeps the thin halo rim readable; deep navy
        would swallow the white→red band at RVE scale.
        """
        from matplotlib.colors import LinearSegmentedColormap

        return LinearSegmentedColormap.from_list(
            "halo_rwb",
            [
                (0.00, "#dbe9f6"),  # foam far from cut
                (0.15, "#92c5de"),
                (0.40, "#f7f7f7"),  # mid
                (0.70, "#f4a582"),
                (1.00, "#b2182b"),  # resin-rich / neat
            ],
            N=n,
        )

    def halo_resin_color(self) -> str:
        """Solid neat-kerf colour matching the high end of :meth:`halo_cmap`."""
        return "#b2182b"

    def publication_rcparams(self) -> dict:
        """matplotlib rcParams for clean, publication-quality figures.

        Serif text + mathtext so symbols and axis labels read like a paper;
        adopted from the b3 house style. Apply via ``plt.rcParams.update(...)``
        or a ``plt.rc_context``.
        """
        return {
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "figure.facecolor": self.background,
            "savefig.facecolor": self.background,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.grid": False,
            "axes.edgecolor": "#444444",
        }


DEFAULT_THEME = CoreTheme()
