"""Gauss-point sampling of a ScoreField (numpy).

Thin adapter over :mod:`b3_core.solvers.sampling`: it evaluates the trilinear
hex Gauss-point coordinates, then calls the shared halo sampler so the numpy and
MFEM backends integrate the same ``P(resin)``.
"""

from __future__ import annotations

import numpy as np

from b3_core.solvers.numpy_fe.assembly import SHAPE_N as N_SHAPE
from b3_core.solvers.sampling import cell_aabb_mm, halo_probability


def gauss_point_resin_P(
    points_m, cells, score_field, *, strategy="exact", resolution=3, idw_power=2.0
) -> np.ndarray:
    """P(resin) at each element's 8 Gauss points -> (n_elem, 8).

    ``strategy="exact"`` samples the field at the Gauss point; ``"local_cloud"``
    samples a ``resolution**3`` cloud of material sub-points per element and
    inverse-distance-weights them to each Gauss point (sub-element averaging).
    """
    xe = np.asarray(points_m)[cells]  # (n, 8, 3) metres
    gp_mm = np.einsum("gn,enj->egj", N_SHAPE, xe) * 1000.0  # (n, 8, 3) mm
    lo_mm = hi_mm = None
    if strategy == "local_cloud":
        lo_mm, hi_mm = cell_aabb_mm(points_m, cells)
    return halo_probability(
        gp_mm,
        score_field=score_field,
        strategy=strategy,
        resolution=resolution,
        idw_power=idw_power,
        lo_mm=lo_mm,
        hi_mm=hi_mm,
    )
