"""Deprecated. Import from b3_core.solvers.calculix.base."""

import warnings

from b3_core.solvers.calculix.base import postprocess

warnings.warn(
    "b3_core.post.base is deprecated; use b3_core.solvers.calculix.base",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "postprocess",
]
