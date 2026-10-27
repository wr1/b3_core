"""Deprecated orchestration module.

Models live in :mod:`b3_core.models`, loading in :mod:`b3_core.loaders`,
and the solve in :mod:`b3_core.api`. This module re-exports the 0.2 names.
"""

from __future__ import annotations

import warnings

from b3_core.api import cprop, homogenize
from b3_core.loaders import load_case, normalize_case
from b3_core.models import CpropInput, Material

warnings.warn(
    "b3_core.core.cprop is deprecated; use b3_core.api and b3_core.models",
    DeprecationWarning,
    stacklevel=2,
)


def halo_reach(dct) -> float:
    from b3_core.core.scoring import halo_reach as reach

    return reach(dct)


def _score_field(dct):
    from b3_core.core.scoring import score_field_for

    return score_field_for(dct)


def _is_orthotropic(dct) -> bool:
    if hasattr(dct, "is_orthotropic"):
        return bool(dct.is_orthotropic)
    return any(
        (dct.get(part) or {}).get("E1") is not None
        or (dct.get(part) or {}).get("Ex") is not None
        for part in ("core", "resin")
    )


def _needs_numpy(dct) -> bool:
    return _is_orthotropic(dct) or halo_reach(dct) > 0.0


__all__ = [
    "CpropInput",
    "Material",
    "cprop",
    "halo_reach",
    "homogenize",
    "load_case",
    "normalize_case",
]
