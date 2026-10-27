"""Deprecated. Import from b3_core.solvers.calculix.post."""

import warnings

from b3_core.solvers.calculix.post import postprocess

warnings.warn(
    "b3_core.post.skins is deprecated; use b3_core.solvers.calculix.post",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "postprocess",
]
