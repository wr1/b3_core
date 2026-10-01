"""Solver-neutral Gauss-point material sampling.

Both the numpy and the MFEM backends integrate the *same* constitutive map:
neat phase stiffness per element (foam / resin / face skin), optionally graded
by the stochastic resin halo at each Gauss point. Keeping the map here — the
halo probability ``P(resin)``, ``local_cloud`` averaging, the phase/attribute
assignment and the rule of mixtures — guarantees the two backends see identical
material, so their 6x6 stiffnesses agree.

Backends only differ in how they get the Gauss-point coordinates (numpy from
the trilinear shape functions, MFEM from its element transformations) and in
how they assemble/solve. Everything constitutive lives in this module.
"""

from __future__ import annotations

import numpy as np

from b3_core.solvers.elasticity import (
    constituent_dict,
    isotropic_C,
    material_C,
)

# Hard-coded skin material, shared by the numpy and MFEM backends.
FACE_E = 12_000_000_000.0
FACE_NU = 0.3

# Gauss points per linear hexahedron (2x2x2 rule).
NQ_HEX = 8


def unit_grid(resolution: int) -> np.ndarray:
    """``resolution**3`` cell-centred sub-points in ``[0, 1]^3``."""
    t = (np.arange(resolution) + 0.5) / resolution
    return np.stack(np.meshgrid(t, t, t, indexing="ij"), -1).reshape(-1, 3)


def phase_attributes(grid, face=None) -> np.ndarray:
    """Tag cells: foam = 1, resin = 2, face skin = 3."""
    resin_cells = np.asarray(grid.cell_data["resin"], dtype=bool)
    face_cells = np.asarray(grid.cell_data["face"], dtype=bool)
    attr = np.ones(grid.n_cells, dtype=np.int64)
    attr[resin_cells] = 2
    if face is not None and face_cells.any():
        attr[face_cells] = 3
    return attr


def cell_aabb_mm(points_m, cells) -> tuple[np.ndarray, np.ndarray]:
    """Per-element axis-aligned bounds in millimetres: ``(lo, hi)`` each (n, 3)."""
    xe = np.asarray(points_m, dtype=float)[cells] * 1000.0
    return xe.min(axis=1), xe.max(axis=1)


def halo_probability(
    gp_mm,
    *,
    score_field,
    strategy: str = "exact",
    resolution: int = 3,
    idw_power: float = 2.0,
    lo_mm=None,
    hi_mm=None,
) -> np.ndarray:
    """``P(resin)`` at Gauss points -> ``(n_elem, nq)``.

    ``gp_mm`` is ``(n_elem, nq, 3)`` in millimetres. ``exact`` samples the field
    at each Gauss point; ``local_cloud`` samples a ``resolution**3`` cloud per
    element (AABB ``lo_mm`` / ``hi_mm``, millimetres) and inverse-distance-weights
    it to each Gauss point.
    """
    gp = np.asarray(gp_mm, dtype=float)
    n_elem, nq = gp.shape[:2]
    if strategy != "local_cloud":
        p = score_field.resin_probability(gp.reshape(-1, 3))
        return np.asarray(p, dtype=float).reshape(n_elem, nq)
    if lo_mm is None or hi_mm is None:
        raise ValueError("local_cloud sampling needs element lo/hi bounds")
    ref = unit_grid(resolution)
    cloud = lo_mm[:, None, :] + ref[None, :, :] * (hi_mm - lo_mm)[:, None, :]
    p_cloud = np.asarray(
        score_field.resin_probability(cloud.reshape(-1, 3)), dtype=float
    ).reshape(n_elem, -1)
    dist = np.linalg.norm(gp[:, :, None, :] - cloud[:, None, :, :], axis=-1)
    w = 1.0 / np.maximum(dist, 1e-9) ** idw_power
    return (w * p_cloud[:, None, :]).sum(-1) / w.sum(-1)


def blend_halo(c_core: np.ndarray, c_resin: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Rule-of-mixtures stiffness ``P*C_resin + (1-P)*C_core`` at ``P``."""
    return p * c_resin + (1.0 - p) * c_core


def phase_stiffness(attr, nq: int, core, resin, face=None) -> np.ndarray:
    """Neat per-Gauss-point 6x6 stiffness by cell attribute -> (n_elem, nq, 6, 6)."""
    n_elem = len(attr)
    c_core, c_resin = material_C(core), material_C(resin)
    out = np.broadcast_to(c_core, (n_elem, nq, 6, 6)).copy()
    out[attr == 2] = c_resin
    if (attr == 3).any():
        if face:
            fd = constituent_dict(face)
            if fd.get("E1") is None:
                fd.setdefault("E", FACE_E)
                fd.setdefault("nu", FACE_NU)
            c_face = material_C(fd)
        else:
            c_face = isotropic_C(FACE_E, FACE_NU)
        out[attr == 3] = c_face
    return out


def per_gp_stiffness(
    attr,
    gp_mm,
    *,
    core,
    resin,
    face=None,
    score_field=None,
    scoring=None,
    points_m=None,
    cells=None,
) -> np.ndarray:
    """Full constitutive map: neat phases + graded resin halo.

    ``gp_mm`` is ``(n_elem, nq, 3)`` Gauss-point coordinates in millimetres.
    ``points_m`` / ``cells`` (metres) are only needed for the ``local_cloud``
    sampling strategy. Returns ``(n_elem, nq, 6, 6)``.
    """
    attr = np.asarray(attr)
    out = phase_stiffness(attr, gp_mm.shape[1], core, resin, face)
    if score_field is None or not getattr(score_field, "active", False):
        return out

    foam = np.flatnonzero(attr == 1)
    if not len(foam):
        return out
    sampling = (scoring or {}).get("sampling") or {}
    strategy = sampling.get("strategy", "exact")
    lo = hi = None
    if strategy == "local_cloud":
        if points_m is None or cells is None:
            raise ValueError("local_cloud sampling needs points_m and cells")
        lo_all, hi_all = cell_aabb_mm(points_m, cells)
        lo, hi = lo_all[foam], hi_all[foam]
    p = halo_probability(
        gp_mm[foam],
        score_field=score_field,
        strategy=strategy,
        resolution=int(sampling.get("resolution", 3)),
        idw_power=float(sampling.get("idw_power", 2.0)),
        lo_mm=lo,
        hi_mm=hi,
    )[:, :, None, None]
    out[foam] = blend_halo(material_C(core), material_C(resin), p)
    return out


__all__ = [
    "FACE_E",
    "FACE_NU",
    "NQ_HEX",
    "blend_halo",
    "cell_aabb_mm",
    "halo_probability",
    "per_gp_stiffness",
    "phase_attributes",
    "phase_stiffness",
    "unit_grid",
]
