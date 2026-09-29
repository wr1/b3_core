"""Deprecated. Import from b3_core.export.ccx."""

import warnings

from b3_core.export.ccx import ccx_ortho_card, ortho_dijkl

warnings.warn(
    "b3_core.io.ccx_card is deprecated; use b3_core.export.ccx",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "ccx_ortho_card",
    "ortho_dijkl",
]
