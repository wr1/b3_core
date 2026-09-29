"""Numpy finite-element backend."""

from b3_core.solvers.numpy_fe.backend import AnisoResult, NumpyBackend, runnumpy
from b3_core.solvers.numpy_fe.failure import (
    ALLOWABLE_RESIN_STRAIN,
    resin_failure_index,
)

__all__ = [
    "ALLOWABLE_RESIN_STRAIN",
    "AnisoResult",
    "NumpyBackend",
    "resin_failure_index",
    "runnumpy",
]
