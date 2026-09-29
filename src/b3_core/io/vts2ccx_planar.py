"""Deprecated. Import from b3_core.solvers.calculix.planar."""

import warnings

from b3_core.solvers.calculix.planar import (
    planar_mpcs,
    shearbcs,
    vtstoccx_planar,
    write_mpc,
    write_nodeset,
)

warnings.warn(
    "b3_core.io.vts2ccx_planar is deprecated; use b3_core.solvers.calculix.planar",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "planar_mpcs",
    "shearbcs",
    "vtstoccx_planar",
    "write_mpc",
    "write_nodeset",
]
