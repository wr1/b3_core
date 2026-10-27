"""b3_core — homogenized properties for grooved sandwich-panel cores.

Periodic-BC homogenisation of sawcut/grooved foam or balsa cores. Build a
:class:`~b3_core.cases.CoreCase` (the historical name is ``Textile``) and pass
it to :func:`homogenize`. The call is pure: it writes nothing unless
``write=True``.
"""

from __future__ import annotations

from b3_core import _version as _version_mod
from b3_core.api import (
    cprop,
    homogenize,
    homogenize_to_disk,
    run_case,
    stiffness_tensor,
    sweep,
)
from b3_core.cases import (
    CoreCase,
    Textile,
    crossed,
    curved_panel,
    grid_scored,
    plain,
    two_sided,
    uniaxial,
)
from b3_core.loaders import normalize_case
from b3_core.models import CaseInput, CpropInput, Material
from b3_core.result import CoreResult, RunRecord
from b3_core.skill import skill_path

__version__ = _version_mod.__version__

__all__ = [
    "__version__",
    "CaseInput",
    "CoreCase",
    "CoreResult",
    "CpropInput",
    "Material",
    "RunRecord",
    "Textile",
    "cprop",
    "crossed",
    "curved_panel",
    "grid_scored",
    "homogenize",
    "homogenize_to_disk",
    "normalize_case",
    "plain",
    "run_case",
    "skill_path",
    "stiffness_tensor",
    "sweep",
    "two_sided",
    "uniaxial",
]


def __getattr__(name: str):
    if name in {
        "CorePhysicsSurrogate",
        "fit_from_homogenization",
        "fit_physics_surrogate",
    }:
        from b3_core import physics_surrogate as surrogate

        return getattr(surrogate, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
