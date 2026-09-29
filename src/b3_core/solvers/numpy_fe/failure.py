"""Resin-grid failure check (Laustsen et al. 2014)."""

from __future__ import annotations

import numpy as np

from b3_core.solvers.numpy_fe.backend import AnisoResult

# Allowable in-situ resin failure strain from uniaxial grid-scored tension tests
# (Laustsen et al. 2014, Table 3); the resin grid fails brittle far below bulk.
ALLOWABLE_RESIN_STRAIN = {
    "H60_resinA": 8443e-6,
    "H130_resinA": 13120e-6,
    "resinB": 5194e-6,
}


def _max_principal_strain(eps_voigt: np.ndarray) -> np.ndarray:
    """Max (most tensile) principal strain per row of engineering-Voigt strains."""
    e = np.asarray(eps_voigt, dtype=float).reshape(-1, 6)
    T = np.zeros((len(e), 3, 3))
    T[:, 0, 0], T[:, 1, 1], T[:, 2, 2] = e[:, 0], e[:, 1], e[:, 2]
    T[:, 1, 2] = T[:, 2, 1] = 0.5 * e[:, 3]  # gamma/2
    T[:, 0, 2] = T[:, 2, 0] = 0.5 * e[:, 4]
    T[:, 0, 1] = T[:, 1, 0] = 0.5 * e[:, 5]
    return np.linalg.eigvalsh(T)[:, -1]


def resin_failure_index(
    result: AnisoResult,
    *,
    macro_strain=None,
    macro_stress=None,
    allowable: float = ALLOWABLE_RESIN_STRAIN["H60_resinA"],
) -> dict:
    """Strain-based resin-grid failure check (Laustsen et al. 2014).

    Applies a macroscopic strain (engineering Voigt, order xx,yy,zz,yz,xz,xy) or
    stress to the homogenised RVE, reconstructs the per-element strain in the
    resin cells, and compares the maximum tensile principal strain to the
    allowable in-situ resin strain. `failure_index >= 1` predicts resin fracture.
    """
    if result.elem_strain is None:
        raise ValueError("need a return_details=True result for the failure check")
    if macro_strain is None:
        if macro_stress is None:
            raise ValueError("provide macro_strain or macro_stress")
        macro_strain = result.compliance @ np.asarray(macro_stress, dtype=float)
    eps0 = np.asarray(macro_strain, dtype=float).reshape(6)

    # Linear superposition of the six unit-case element strains.
    elem_eps = np.einsum("i,eiv->ev", eps0, result.elem_strain)
    resin = result.elem_attr == 2
    if not resin.any():
        return {"failure_index": 0.0, "max_principal_strain": 0.0, "n_resin": 0}
    maxp = _max_principal_strain(elem_eps[resin])
    worst = int(np.argmax(maxp))
    return {
        "failure_index": float(maxp.max() / allowable),
        "max_principal_strain": float(maxp.max()),
        "allowable": float(allowable),
        "n_resin": int(resin.sum()),
        "worst_resin_element": int(np.flatnonzero(resin)[worst]),
    }
