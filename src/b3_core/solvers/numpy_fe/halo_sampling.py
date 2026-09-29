"""Gauss-point sampling of a ScoreField."""

from __future__ import annotations

import numpy as np

from b3_core.solvers.numpy_fe.assembly import SHAPE_N as N_SHAPE


def _unit_grid(resolution: int) -> np.ndarray:
    t = (np.arange(resolution) + 0.5) / resolution  # cell-centred in [0,1]
    return np.stack(np.meshgrid(t, t, t, indexing="ij"), -1).reshape(-1, 3)


def gauss_point_resin_P(
    points_m, cells, score_field, *, strategy="exact", resolution=3, idw_power=2.0
) -> np.ndarray:
    """P(resin) at each element's 8 Gauss points -> (n_elem, 8).

    ``strategy="exact"`` samples the field at the Gauss point; ``"local_cloud"``
    samples a ``resolution**3`` cloud of material sub-points per element and
    inverse-distance-weights them to each Gauss point (sub-element averaging).
    """
    Xe = points_m[cells]  # (n, 8, 3) metres
    gp = np.einsum("gn,enj->egj", N_SHAPE, Xe) * 1000.0  # (n, 8, 3) mm Gauss coords
    n = len(cells)
    if strategy == "exact":
        return score_field.resin_probability(gp.reshape(-1, 3)).reshape(n, 8)
    ref = _unit_grid(resolution)  # (M, 3) in [0,1]
    lo, hi = Xe.min(axis=1), Xe.max(axis=1)  # (n, 3) element AABB
    cloud = (
        lo[:, None, :] + ref[None, :, :] * (hi - lo)[:, None, :]
    ) * 1000.0  # (n,M,3) mm
    Pc = score_field.resin_probability(cloud.reshape(-1, 3)).reshape(n, -1)  # (n, M)
    dist = np.linalg.norm(
        gp[:, :, None, :] - cloud[:, None, :, :], axis=-1
    )  # (n, 8, M)
    w = 1.0 / np.maximum(dist, 1e-9) ** idw_power
    return (w * Pc[:, None, :]).sum(-1) / w.sum(-1)  # (n, 8)
