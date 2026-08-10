"""b3_core — homogenized properties for grooved sandwich-panel cores.

Periodic-BC FEA homogenisation of sawcut/grooved foam or balsa cores. Prefer
**textile-as-code** (``b3_core.cases`` / ``CpropInput``) then ``homogenize``;
JSON/YAML files remain optional CLI interchange. ``homogenize`` returns a
`b3_mat.OrthotropicMaterial` for the wider b3 pipeline. Backends: MFEM
(default), CalculiX, optional FEniCSx, and numpy.
"""

from b3_core.cases import (
    Textile,
    crossed,
    curved_panel,
    grid_scored,
    plain,
    two_sided,
    uniaxial,
)
from b3_core.core.cprop import CpropInput, Material, cprop, homogenize, normalize_case
from b3_core.physics_surrogate import (
    CorePhysicsSurrogate,
    fit_from_homogenization,
    fit_physics_surrogate,
)
from b3_core.result import CoreResult

__version__ = "0.1.0"

__all__ = [
    "CoreResult",
    "CpropInput",
    "Material",
    "Textile",
    "cprop",
    "homogenize",
    "normalize_case",
    "plain",
    "uniaxial",
    "crossed",
    "two_sided",
    "curved_panel",
    "grid_scored",
    "CorePhysicsSurrogate",
    "fit_physics_surrogate",
    "fit_from_homogenization",
]
