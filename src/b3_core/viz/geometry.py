"""Shared mesh/phase helpers for the 2D and 3D grooved-core renderers.

Centralises the per-cell phase classification, slab selection and plane sampling
that were duplicated inside ``datasheet.py``.
"""

from __future__ import annotations

import numpy as np

from b3_core.viz.theme import CORE, FACE, RESIN


def cell_material(mesh) -> np.ndarray:
    """Per-cell phase code: 0 core, 1 resin, 2 face (from mesh cell_data tags)."""
    resin = np.asarray(mesh.cell_data["resin"]).astype(bool)
    face = np.asarray(mesh.cell_data["face"]).astype(bool)
    mat = np.full(mesh.n_cells, CORE, dtype=int)
    mat[resin] = RESIN
    mat[face] = FACE
    return mat


def axis_vectors(mesh):
    """Sorted unique grid-line coordinates along x, y, z."""
    return np.unique(mesh.x), np.unique(mesh.y), np.unique(mesh.z)


def split_phases(mesh, mat: np.ndarray | None = None) -> dict:
    """Threshold the mesh into ``{"core", "resin", "face"}`` sub-grids.

    Tags a scratch ``__phase`` cell array so each phase can be extracted by code;
    missing phases map to empty grids.
    """
    if mat is None:
        mat = cell_material(mesh)
    view = mesh.copy()
    view.cell_data["__phase"] = mat
    return {
        "core": view.threshold([CORE - 0.5, CORE + 0.5], scalars="__phase"),
        "resin": view.threshold([RESIN - 0.5, RESIN + 0.5], scalars="__phase"),
        "face": view.threshold([FACE - 0.5, FACE + 0.5], scalars="__phase"),
    }


def best_slab(mesh, mat: np.ndarray, axis: int, centers: np.ndarray) -> float:
    """Cell-centre coordinate of the slab along ``axis`` richest in resin.

    Guarantees a cut that actually intersects grooves; falls back to the median
    plane when there is no resin.
    """
    coords = np.round(centers[:, axis], 6)
    uniq = np.unique(coords)
    resin = mat == RESIN
    counts = np.array([resin[coords == c].sum() for c in uniq])
    if counts.max() == 0:
        return float(uniq[len(uniq) // 2])
    return float(uniq[int(counts.argmax())])


def sample_plane(mesh, mat, u_axis, v_axis, fixed_axis, coord, u_vals, v_vals):
    """Phase code on a plane as a float grid (NaN outside the RVE)."""
    uu, vv = np.meshgrid(u_vals, v_vals)
    pts = np.zeros((uu.size, 3))
    pts[:, u_axis] = uu.ravel()
    pts[:, v_axis] = vv.ravel()
    pts[:, fixed_axis] = coord
    cids = np.asarray(mesh.find_containing_cell(pts.astype(float)))
    out = np.full(uu.size, np.nan)
    inside = cids >= 0
    out[inside] = mat[cids[inside]]
    return out.reshape(uu.shape)
