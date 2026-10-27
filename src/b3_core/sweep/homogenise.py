"""Deprecated spelling. Use :mod:`b3_core.sweep.homogenize`."""

from __future__ import annotations

import warnings

warnings.warn(
    "b3_core.sweep.homogenise is deprecated; use b3_core.sweep.homogenize",
    DeprecationWarning,
    stacklevel=2,
)

from b3_core.sweep.homogenize import (  # noqa: E402
    run_all_homogenise,
    run_curvature,
    run_patterns,
    run_thickness,
)

__all__ = [
    "run_all_homogenise",
    "run_curvature",
    "run_patterns",
    "run_thickness",
]
