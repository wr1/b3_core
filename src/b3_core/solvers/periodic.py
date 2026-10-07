"""Periodic z-face constraints for a kerf-tapered RVE.

``morph_kerf_walls`` remaps x and y as a function of z. The x = dx face stays
an exact translate of x = 0, and the same for y. The two z faces still span
the rectangle ``[0, dx] × [0, dy]`` — each slice is a cartesian tensor grid —
but they no longer share (x, y) locations, so a coordinate tie drops the
through-thickness pairs.

The periodic condition on that face is the trace of the fluctuation:
``w(x, y, H) = w(x, y, 0)``. A top node is the bilinear interpolant of the
bottom-face cell that contains its (x, y). When every top node already sits
on a bottom node, the weights are a permutation and the solvers keep the
node-to-node tie.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ZFaceTies:
    """One row per ``z = H`` node.

    ``masters`` are ``z = 0`` node indices. ``weights`` are the bilinear
    values at the slave's (x, y); they sum to 1. ``identity`` is true when
    every slave lands on a single bottom node in the same coordinate hash the
    node-to-node tie uses.
    """

    identity: bool
    slave: np.ndarray
    masters: np.ndarray
    weights: np.ndarray


def _span(points: np.ndarray) -> float:
    return float((points.max(axis=0) - points.min(axis=0)).max())


def face_tol(points: np.ndarray) -> float:
    """Same face tolerance as the coordinate periodic hash."""
    return 1e-9 + 1e-6 * _span(points)


def _cluster_lines(vals: np.ndarray, tol: float) -> np.ndarray:
    v = np.sort(np.asarray(vals, dtype=float))
    if len(v) == 0:
        return v
    keep = np.empty(len(v), dtype=bool)
    keep[0] = True
    keep[1:] = np.diff(v) > tol
    return v[keep]


def _interval(val: float, lines: np.ndarray) -> tuple[int, float]:
    """Cell index and local coordinate of ``val`` on a sorted 1-D grid."""
    if val <= lines[0]:
        return 0, 0.0
    if val >= lines[-1]:
        return len(lines) - 2, 1.0
    i = int(np.searchsorted(lines, val, side="right") - 1)
    i = min(max(i, 0), len(lines) - 2)
    span = float(lines[i + 1] - lines[i])
    if span == 0.0:
        return i, 0.0
    return i, float((val - lines[i]) / span)


def _xy_hash(points: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Rounded (x, y) keys, the in-plane part of ``_periodic_masters``."""
    lo = points.min(axis=0)
    span = max(_span(points), 1.0)
    return np.round((np.asarray(xy, dtype=float) - lo[:2]) / span, 6)


def z_face_ties(points: np.ndarray) -> ZFaceTies:
    """Bilinear ties from every ``z = H`` node onto the ``z = 0`` tensor grid."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    tol = face_tol(points)
    on_bot = np.abs(points[:, 2] - lo[2]) < tol
    on_top = np.abs(points[:, 2] - hi[2]) < tol
    bot_ids = np.flatnonzero(on_bot)
    top_ids = np.flatnonzero(on_top)
    if len(bot_ids) == 0 or len(top_ids) == 0:
        raise ValueError("mesh has no nodes on a z face")

    xs = _cluster_lines(points[bot_ids, 0], tol)
    ys = _cluster_lines(points[bot_ids, 1], tol)
    if len(xs) < 2 or len(ys) < 2:
        raise ValueError("z = 0 face is not a 2-D grid")

    def _axis_index(vals: np.ndarray, lines: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(lines, vals)
        idx = np.clip(idx, 0, len(lines) - 1)
        prev = np.clip(idx - 1, 0, len(lines) - 1)
        use_prev = np.abs(vals - lines[prev]) <= np.abs(vals - lines[idx])
        idx = np.where(use_prev, prev, idx)
        if np.any(np.abs(vals - lines[idx]) > tol):
            raise ValueError("z = 0 node does not lie on the tensor grid")
        return idx.astype(np.int64)

    grid = np.full((len(xs), len(ys)), -1, dtype=np.int64)
    grid[_axis_index(points[bot_ids, 0], xs), _axis_index(points[bot_ids, 1], ys)] = (
        bot_ids
    )
    if np.any(grid < 0):
        raise ValueError("z = 0 face is missing a tensor-grid node")

    n = len(top_ids)
    masters = np.empty((n, 4), dtype=np.int64)
    weights = np.empty((n, 4), dtype=float)
    for s, node in enumerate(top_ids):
        i, tx = _interval(float(points[node, 0]), xs)
        j, ty = _interval(float(points[node, 1]), ys)
        masters[s, 0] = grid[i, j]
        masters[s, 1] = grid[i + 1, j]
        masters[s, 2] = grid[i, j + 1]
        masters[s, 3] = grid[i + 1, j + 1]
        weights[s, 0] = (1.0 - tx) * (1.0 - ty)
        weights[s, 1] = tx * (1.0 - ty)
        weights[s, 2] = (1.0 - tx) * ty
        weights[s, 3] = tx * ty

    bot_keys = {tuple(row) for row in _xy_hash(points, points[bot_ids, :2])}
    top_keys = _xy_hash(points, points[top_ids, :2])
    identity = all(tuple(row) in bot_keys for row in top_keys)
    return ZFaceTies(identity, top_ids, masters, weights)
