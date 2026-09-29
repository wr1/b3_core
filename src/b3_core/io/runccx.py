"""Deprecated. Import from b3_core.solvers.calculix.runner."""

import warnings

from b3_core.solvers.calculix.runner import run_single, runccx

warnings.warn(
    "b3_core.io.runccx is deprecated; use b3_core.solvers.calculix.runner",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "run_single",
    "runccx",
]
