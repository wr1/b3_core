"""Deprecated. Import from b3_core.solvers.mfem."""

import warnings

from b3_core.solvers import mfem as mfem_solver

warnings.warn(
    "b3_core.io.mfem_backend is deprecated; use b3_core.solvers.mfem",
    DeprecationWarning,
    stacklevel=2,
)

MfemResult = mfem_solver.MfemResult
MfemUnavailableError = mfem_solver.MfemUnavailableError
is_mfem_available = mfem_solver.is_mfem_available
runmfem = mfem_solver.runmfem
_macro_strain = mfem_solver._macro_strain

__all__ = [
    "MfemResult",
    "MfemUnavailableError",
    "_macro_strain",
    "is_mfem_available",
    "runmfem",
]
