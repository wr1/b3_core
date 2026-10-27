"""Deprecated. Import from b3_core.solvers.fenicsx."""

import warnings

from b3_core.solvers.elasticity import LOAD_CASES, PROPERTY_KEYS
from b3_core.solvers.elasticity import (
    properties_from_stiffness as _properties_from_stiffness,
)
from b3_core.solvers.fenicsx import (
    FenicsxResult,
    FenicsxUnavailableError,
    is_fenicsx_available,
    runfenicsx,
)
from b3_core.solvers.validation import validate_against_ccx

warnings.warn(
    "b3_core.io.fenicsx is deprecated; use b3_core.solvers.fenicsx",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "LOAD_CASES",
    "PROPERTY_KEYS",
    "FenicsxResult",
    "FenicsxUnavailableError",
    "_properties_from_stiffness",
    "is_fenicsx_available",
    "runfenicsx",
    "validate_against_ccx",
]
