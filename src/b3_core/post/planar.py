"""Deprecated. Import from b3_core.solvers.calculix.planar_post."""

import warnings

from b3_core.solvers.calculix.planar_post import postprocess_planar

warnings.warn(
    "b3_core.post.planar is deprecated; use b3_core.solvers.calculix.planar_post",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "postprocess_planar",
]
