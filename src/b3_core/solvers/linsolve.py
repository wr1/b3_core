"""Sparse factorization with an optional PARDISO path.

``B3_CORE_SOLVER`` selects the direct solver: ``auto`` (default) uses
PARDISO through :mod:`pypardiso` when it imports and the system has at
least ``AUTO_PARDISO_MIN_ROWS`` rows, and SciPy SuperLU otherwise,
``pardiso`` requires it, ``splu`` forces SuperLU. Install the ``fast`` extra
for PARDISO. Both return a solve that takes a vector or an ``(n, k)`` block,
so one factorization serves all six unit load cases.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import numpy as np

_CHOICES = ("auto", "pardiso", "splu")
# Below this size SuperLU beats PARDISO setup cost (measured on the flat RVE).
AUTO_PARDISO_MIN_ROWS = 5_000


def _have_pardiso() -> bool:
    try:
        import pypardiso  # noqa: F401
    except Exception:
        return False
    return True


def resolve_solver(choice: str | None = None) -> str:
    """Return ``"pardiso"`` or ``"splu"`` for ``choice`` or ``B3_CORE_SOLVER``."""
    raw = choice if choice is not None else os.environ.get("B3_CORE_SOLVER", "")
    name = (raw or "auto").strip().lower()
    if name not in _CHOICES:
        raise ValueError(f"B3_CORE_SOLVER must be one of {_CHOICES}, got {raw!r}")
    if name == "splu":
        return "splu"
    if _have_pardiso():
        return "pardiso"
    if name == "pardiso":
        raise ImportError(
            "B3_CORE_SOLVER=pardiso needs pypardiso (pip install b3_core[fast])"
        )
    return "splu"


def factorize(A, *, solver: str | None = None) -> Callable[[np.ndarray], np.ndarray]:
    """Factor sparse ``A`` once; the returned callable solves ``A x = b``."""
    name = resolve_solver(solver)
    explicit = (solver or os.environ.get("B3_CORE_SOLVER", "")).strip().lower()
    if (
        name == "pardiso"
        and explicit != "pardiso"
        and A.shape[0] < AUTO_PARDISO_MIN_ROWS
    ):
        name = "splu"
    if name == "pardiso":
        import pypardiso

        ps = pypardiso.PyPardisoSolver(mtype=11)  # real nonsymmetric: any assembled K
        Acsr = A.tocsr()
        Acsr.sort_indices()
        ps.factorize(Acsr)

        def _solve_pardiso(b: np.ndarray) -> np.ndarray:
            b = np.ascontiguousarray(b, dtype=np.float64)
            x = ps.solve(Acsr, b)
            return np.asarray(x).reshape(b.shape)

        return _solve_pardiso

    from scipy.sparse.linalg import splu

    lu = splu(A.tocsc())

    def _solve_splu(b: np.ndarray) -> np.ndarray:
        return lu.solve(np.asarray(b, dtype=np.float64))

    return _solve_splu
