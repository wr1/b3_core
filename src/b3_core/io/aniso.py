"""Deprecated. Import from b3_core.solvers.numpy_fe."""

import warnings

from b3_core.solvers.elasticity import (
    isotropic_C,
    material_C,
    orthotropic_C,
)
from b3_core.solvers.elasticity import (
    properties_from_stiffness as _properties_from_stiffness,
)
from b3_core.solvers.numpy_fe.assembly import homogenize_aniso
from b3_core.solvers.numpy_fe.backend import AnisoResult, runnumpy
from b3_core.solvers.numpy_fe.failure import ALLOWABLE_RESIN_STRAIN, resin_failure_index
from b3_core.solvers.numpy_fe.halo_sampling import gauss_point_resin_P

warnings.warn(
    "b3_core.io.aniso is deprecated; use b3_core.solvers.numpy_fe",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "ALLOWABLE_RESIN_STRAIN",
    "AnisoResult",
    "_properties_from_stiffness",
    "gauss_point_resin_P",
    "homogenize_aniso",
    "isotropic_C",
    "material_C",
    "orthotropic_C",
    "resin_failure_index",
    "runnumpy",
]
