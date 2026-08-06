"""b3_core — homogenized properties for grooved sandwich-panel cores.

Periodic-BC FEA homogenisation of sawcut/grooved foam or balsa cores. The
`cprop` pipeline writes a full JSON result; `homogenize` wraps it as a
`b3_mat.OrthotropicMaterial` so results plug into the wider b3 pipeline.
Backends: MFEM (default), CalculiX, optional FEniCSx, and numpy.
"""

from b3_core.core.cprop import CpropInput, Material, cprop, homogenize
from b3_core.result import CoreResult

__version__ = "0.1.0"

__all__ = [
    "CoreResult",
    "CpropInput",
    "Material",
    "cprop",
    "homogenize",
]
