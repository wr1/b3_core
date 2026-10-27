"""b3_core.viz — unified visualization layer for grooved cores.

The high-level entry point is :class:`GroovedCoreView`; :class:`CoreScene` is the
fluent 3D builder, :class:`CoreModel` the cached mesh/homogenisation handle, and
:class:`CoreTheme` the shared styling. Tensor utilities expose the directional
elastic properties behind the modulus surface and polar plots.
"""

from b3_core.viz.model import CoreModel
from b3_core.viz.scene import CoreScene
from b3_core.viz.slices import plot_orthogonal_cuts
from b3_core.viz.tensor import (
    engineering_constants,
    modulus_surface,
    polar_modulus,
    youngs_modulus,
)
from b3_core.viz.tensorplot import plot_modulus_polar, plot_stiffness_heatmap
from b3_core.viz.theme import DEFAULT_THEME, CoreTheme
from b3_core.viz.view import GroovedCoreView

__all__ = [
    "DEFAULT_THEME",
    "CoreModel",
    "CoreScene",
    "CoreTheme",
    "GroovedCoreView",
    "engineering_constants",
    "modulus_surface",
    "plot_modulus_polar",
    "plot_orthogonal_cuts",
    "plot_stiffness_heatmap",
    "polar_modulus",
    "youngs_modulus",
]
