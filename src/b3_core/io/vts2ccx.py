"""Deprecated. Import from b3_core.solvers.calculix.writer."""

import warnings

from b3_core.solvers.calculix.writer import nset, periodic_bcs, vtstoccx

warnings.warn(
    "b3_core.io.vts2ccx is deprecated; use b3_core.solvers.calculix.writer",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "nset",
    "periodic_bcs",
    "vtstoccx",
]
