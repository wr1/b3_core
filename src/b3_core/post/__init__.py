"""Deprecated post-processing shims."""

import warnings

warnings.warn(
    "b3_core.post is deprecated; use b3_core.solvers.calculix",
    DeprecationWarning,
    stacklevel=2,
)
