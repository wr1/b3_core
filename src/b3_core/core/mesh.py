#!/usr/bin/env python3
"""Structured RVE mesh with wall-aligned kerf open/close.

Flat rectangular grooves are built first (walls = grid faces, resin tags from
rectangular cell-centre tests). When mould curvature is nonzero, node
coordinates are **morphed** so those same wall faces track the root-hinged
taper ``hw(z)`` — open/close is geometry motion, not voxel property painting.
"""

from __future__ import annotations

import argparse

import numpy as np
import pyvista as pv

# Minimum half-width after pinch (mm) so cells keep positive volume.
_MIN_HW = 1e-3


def create_grooves(cuts, bnd, meshadd=(-0.5, -0.2, 0, 0.2, 0.5), tol=1e-6, kappa=0.0):
    """Per groove-instance left/right walls, signed depth, and curvature taper slope.

    ``kappa`` (1/length) sets the per-instance taper ``slope`` =
    ``-sign(depth) * kappa * pitch / 2`` so a single curvature opens grooves
    mouthing on one face and closes those on the other. Slope is consumed by
    the wall morph (not by resin painting). At ``kappa == 0`` slopes are zero.
    """
    if len(cuts) > 0:
        bot = np.minimum(
            bnd,
            np.maximum(
                0,
                np.concatenate(
                    [
                        np.arange(i[0] - i[1] - 0.5 * i[3], bnd + i[1] + tol, i[1]) + j
                        for i in cuts
                        for j in meshadd
                    ]
                ),
            ),
        )
        top = np.minimum(
            bnd,
            np.maximum(
                0,
                np.concatenate(
                    [
                        np.arange(i[0] - i[1] + 0.5 * i[3], bnd + i[1] + i[3], i[1]) + j
                        for i in cuts
                        for j in meshadd
                    ]
                ),
            ),
        )
        height = np.concatenate(
            [
                np.ones(
                    len(np.arange(i[0] - i[1] - 0.5 * i[3], bnd + i[1] + tol, i[1]))
                )
                * i[2]
                for i in cuts
                for j in meshadd
            ]
        )
        slope = np.concatenate(
            [
                np.ones(
                    len(np.arange(i[0] - i[1] - 0.5 * i[3], bnd + i[1] + tol, i[1]))
                )
                * (-np.sign(i[2]) * kappa * i[1] / 2.0)
                for i in cuts
                for j in meshadd
            ]
        )
    else:
        bot = np.maximum(0, np.array(meshadd))
        top = np.minimum(bnd, np.array(meshadd) + bnd)
        height = np.zeros_like(bot)
        slope = np.zeros_like(bot)

    # Boundary-clamped bot≈top pairs are mesh-line artifacts, not grooves.
    degen = np.abs(top - bot) < tol
    height = np.where(degen, 0.0, height)
    slope = np.where(degen, 0.0, slope)

    return bot, top, height, slope


def create_z_mesh(cuts, th, meshadd=(-0.5, -0.2, 0, 0.2, 0.5), absadd=()):
    fz = np.concatenate([0 + np.array(meshadd), th + np.array(meshadd)])
    for i in cuts:
        for j in meshadd:
            fz = np.append(fz, [i[2] + j, th + i[2] + j])
    fz = np.append(fz, absadd)
    return np.unique(np.minimum(th, np.maximum(0, fz)))


def _mark_resin(resin, coord, c, thickness, centres, halfwidths, depths, slopes=None):
    """Mark cells whose centre falls inside each **rectangular** groove.

    Taper is applied later by morphing wall nodes; slopes are ignored here
    (kept as an optional arg for call-site compatibility).
    """
    del slopes  # wall-aligned morph owns taper
    for c0, hw, d in zip(centres, halfwidths, depths, strict=True):
        if hw <= 0 or d == 0:
            continue
        if d > 0:  # mouth at z = 0, root at z = d
            in_z = c[:, 2] < d
        else:  # mouth at z = thickness, root at z = thickness + d
            in_z = c[:, 2] > thickness + d
        resin = np.maximum(resin, (np.abs(c[:, coord] - c0) < hw) & in_z)
    return resin


def _mark_halo(frac, coord, c, thickness, centres, halfwidths, depths, slopes, s_halo):
    """Geometric kerf-damage halo: cells just outside each groove wall.

    Uses the same ``hw(z)`` law as the wall morph (incl. pinch clamp), so the
    band tracks open/closed kerfs when marked on **post-morph** cell centres.
    Viz only — graded stiffness comes from ``ScoreField`` at integration points.
    """
    for c0, hw, d, s in zip(centres, halfwidths, depths, slopes, strict=True):
        if hw <= 0 or d == 0:
            continue
        if d > 0:
            in_z = c[:, 2] < d
            zeta = d - c[:, 2]
        else:
            in_z = c[:, 2] > thickness + d
            zeta = c[:, 2] - (thickness + d)
        # Same clamp as _hw_at / morph (no negative half-width).
        hw_z = np.maximum(_MIN_HW, hw + s * zeta)
        gap = np.abs(c[:, coord] - c0) - hw_z
        g = np.where(
            (gap >= 0) & (gap < s_halo) & in_z,
            np.clip(1.0 - gap / s_halo, 0.0, 1.0),
            0.0,
        )
        frac = np.maximum(frac, g)
    return frac


def _collapse_lines(vals, tol=1e-6):
    """Sorted unique grid lines, merging any closer than ``tol``."""
    v = np.sort(np.asarray(vals, dtype=float))
    keep = np.concatenate(([True], np.diff(v) > tol))
    return v[keep]


def _physical_grooves(cuts, bnd, kappa: float, tol: float = 1e-6):
    """Groove instances that intersect the domain (including edge partials).

    Uses the *ideal* unclipped centre and half-width so edge kerfs get the same
    taper law as interior ones; wall positions are clipped to ``[0, bnd]`` later
    in ``_ordered_breaks``. Meshadd ghosts are not included (``meshadd=[0]``
    lattice only).

    Returns list of ``(c0, hw0, depth, slope)`` sorted by ``c0``.
    """
    if not cuts:
        return []
    out = []
    for row in cuts:
        offset, pitch, depth, width = (
            float(row[0]),
            float(row[1]),
            float(row[2]),
            float(row[3]),
        )
        if depth == 0.0 or width <= 0.0 or pitch <= 0.0:
            continue
        hw = 0.5 * width
        slope = -np.sign(depth) * kappa * pitch / 2.0
        # Same instance lattice as create_grooves(..., meshadd=[0]) left walls.
        lefts = np.arange(offset - pitch - hw, bnd + pitch + tol, pitch)
        for left in lefts:
            right = left + width
            # Skip only if completely outside the domain.
            if right <= tol or left >= bnd - tol:
                continue
            c0 = 0.5 * (left + right)
            out.append((float(c0), float(hw), float(depth), float(slope)))
    out.sort(key=lambda g: g[0])
    return out


def _hw_at(hw0: float, depth: float, slope: float, z: float, thickness: float) -> float:
    """Half-width at height z; flat nominal outside the groove z-band."""
    if depth > 0:
        if z < 0.0 or z > depth:
            return hw0
        zeta = depth - z
    else:
        z_root = thickness + depth
        if z < z_root or z > thickness:
            return hw0
        zeta = z - z_root
    return float(max(_MIN_HW, hw0 + slope * zeta))


def _ordered_breaks(grooves, length: float, hw_of) -> np.ndarray:
    """Breakpoints ``[0, L1, R1, L2, R2, …, L]`` — length ``2 + 2·n_grooves``.

    Edge grooves use ideal centres clipped to the domain so the first/last
    kerfs morph like interior ones. Each groove always contributes two breaks
    (strictly increasing) so flat vs tapered maps share topology.
    """
    length = float(length)
    walls: list[tuple[float, float]] = []
    for c0, hw0, d, s in grooves:
        hw = float(hw_of(c0, hw0, d, s))
        left = float(np.clip(c0 - hw, 0.0, length))
        right = float(np.clip(c0 + hw, 0.0, length))
        if right < left + 1e-6:
            # Fully pinched against a bound — keep a tiny positive interval.
            if left <= 1e-12:
                left, right = 0.0, min(length, 1e-6)
            else:
                left, right = max(0.0, length - 1e-6), length
        walls.append((left, right))
    walls.sort(key=lambda lr: (lr[0], lr[1]))
    pts = [0.0]
    for left, right in walls:
        pts.append(left)
        pts.append(right)
    pts.append(length)
    arr = np.asarray(pts, dtype=float)
    for i in range(1, len(arr)):
        if arr[i] <= arr[i - 1]:
            arr[i] = min(length, arr[i - 1] + 1e-6)
    arr[-1] = length
    return arr


def _breaks_flat(grooves, length: float) -> np.ndarray:
    return _ordered_breaks(grooves, length, lambda c0, hw0, d, s: hw0)


def _breaks_at_z(grooves, length: float, z: float, thickness: float) -> np.ndarray:
    return _ordered_breaks(
        grooves,
        length,
        lambda c0, hw0, d, s: _hw_at(hw0, d, s, z, thickness),
    )


def _affine_remap_1d(u: np.ndarray, breaks_src: np.ndarray, breaks_dst: np.ndarray) -> np.ndarray:
    """Map each u from intervals breaks_src → breaks_dst (monotone, no crossing)."""
    u = np.asarray(u, dtype=float)
    n = len(breaks_src) - 1
    if n < 1:
        return u.copy()
    idx = np.searchsorted(breaks_src, u, side="right") - 1
    idx = np.clip(idx, 0, n - 1)
    left_s, right_s = breaks_src[idx], breaks_src[idx + 1]
    left_d, right_d = breaks_dst[idx], breaks_dst[idx + 1]
    span_s = right_s - left_s
    t = np.zeros_like(u, dtype=float)
    good = span_s > 1e-15
    t[good] = (u[good] - left_s[good]) / span_s[good]
    return left_d + t * (right_d - left_d)


def morph_kerf_walls(
    grid: pv.StructuredGrid,
    xcuts,
    ycuts,
    thickness: float,
    dx: float,
    dy: float,
    kx: float = 0.0,
    ky: float = 0.0,
    madd=(),
) -> pv.StructuredGrid:
    """Warp the flat RVE so wall faces track hw(z) (interval-affine morph).

    **Flattened-for-FEA kinematics:** every bay between walls is remapped, so
    foam between open kerfs becomes trapezoidal and kerfs taper. That is the
    intended flat RVE (not hinged rectangles). Topology/tags unchanged.

    (Curved-on-mould visualization uses rigid rectangular foam blocks separately.)
    """
    del madd  # reserved for callers; interval morph uses physical walls only
    if abs(kx) < 1e-15 and abs(ky) < 1e-15:
        return grid

    pts = np.array(grid.points, dtype=float, copy=True)

    def _morph_axis(axis: int, cuts, length: float, kappa: float) -> None:
        grooves = _physical_grooves(cuts, length, kappa)
        if not grooves:
            return
        bf = _breaks_flat(grooves, length)
        z_vals = pts[:, 2]
        for z0 in np.unique(np.round(z_vals, 10)):
            mask = np.abs(z_vals - z0) < 5e-10
            bd = _breaks_at_z(grooves, length, float(z0), thickness)
            assert len(bd) == len(bf), (len(bd), len(bf), z0)
            pts[mask, axis] = _affine_remap_1d(pts[mask, axis], bf, bd)

    _morph_axis(0, xcuts, dx, kx)
    _morph_axis(1, ycuts, dy, ky)
    grid.points = pts
    return grid


def create_grooved_mesh(
    thickness,
    dx,
    dy,
    xcuts,
    ycuts,
    madd=(-0.4, -0.2, 0, 0.2, 0.4),
    tface=2.0,
    kx=0.0,
    ky=0.0,
    s_halo=0.0,
):
    """Build a structured RVE; open/close via wall-aligned morph when κ ≠ 0.

    1. Rectangular flat grid (walls = grid faces, rectangular resin tags).
    2. If ``kx``/``ky`` nonzero, interval-affine morph so walls track ``hw(z)``.
       Foam bays become trapezoidal on the flat RVE (intended FEA geometry).
    """
    # Flat groove instances for grid lines + rectangular resin tags
    bx, tx, hx, _sx0 = create_grooves(xcuts, dx, meshadd=madd, kappa=0.0)
    by, ty, hy, _sy0 = create_grooves(ycuts, dy, meshadd=madd, kappa=0.0)
    _bx, _tx, _hx, sx = create_grooves(xcuts, dx, meshadd=madd, kappa=kx)
    _by, _ty, _hy, sy = create_grooves(ycuts, dy, meshadd=madd, kappa=ky)
    del _bx, _tx, _hx, _by, _ty, _hy

    tol = 1e-3
    fz = create_z_mesh(
        xcuts + ycuts, thickness + tface, meshadd=madd, absadd=[thickness + tol]
    )
    if s_halo > 0:
        hz = np.linspace(0.0, 1.0, 3)[1:]
        roots = [
            d + s_halo * hz if d > 0 else (thickness + d) - s_halo * hz
            for cut in list(xcuts) + list(ycuts)
            if (d := cut[2]) != 0
        ]
        if roots:
            fz = np.append(fz, np.clip(np.concatenate(roots), 0.0, thickness + tface))

    # Through-thickness stations so morphed walls are piecewise-linear.
    if abs(kx) > 0.0 or abs(ky) > 0.0:
        n_z_taper = 9
        taper_z = []
        for cut in list(xcuts) + list(ycuts):
            d = float(cut[2])
            if d == 0.0:
                continue
            if d > 0:
                z0, z1 = 0.0, d
            else:
                z0, z1 = thickness + d, thickness
            taper_z.append(np.linspace(z0, z1, n_z_taper))
        if taper_z:
            fz = np.append(fz, np.clip(np.concatenate(taper_z), 0.0, thickness + tface))

    xc, xhw = 0.5 * (bx + tx), 0.5 * (tx - bx)
    yc, yhw = 0.5 * (by + ty), 0.5 * (ty - by)

    # Local refinement near walls. Halo band lines at nominal hw + s_halo, and
    # (when κ opens the mouth) at max |hw(z)| + s_halo so the open-mouth foam
    # strip is resolved. Morph owns the wall motion; no voxel flare tags.
    xoff = xhw[:, None]
    yoff = yhw[:, None]
    if s_halo > 0:
        hf = np.linspace(0.0, 1.0, 3)[1:]
        xoff = np.concatenate([xoff, xhw[:, None] + s_halo * hf[None, :]], axis=1)
        yoff = np.concatenate([yoff, yhw[:, None] + s_halo * hf[None, :]], axis=1)
    if s_halo > 0 and (abs(kx) > 0.0 or abs(ky) > 0.0):
        # Farthest wall from root-hinged taper (open mouth or closed root).
        def _flare_hw(hw0, depth, slope):
            z_span = abs(float(depth))
            return float(max(hw0, hw0 + slope * z_span, hw0 + slope * 0.0))

        x_flare = np.array(
            [
                _flare_hw(float(h), float(d), float(s)) + s_halo
                for h, d, s in zip(xhw, hx, sx, strict=True)
            ]
        )
        y_flare = np.array(
            [
                _flare_hw(float(h), float(d), float(s)) + s_halo
                for h, d, s in zip(yhw, hy, sy, strict=True)
            ]
        )
        if len(x_flare):
            xoff = np.concatenate([xoff, x_flare[:, None]], axis=1)
        if len(y_flare):
            yoff = np.concatenate([yoff, y_flare[:, None]], axis=1)

    xlines = np.clip(
        np.concatenate(
            [bx, tx, (xc[:, None] - xoff).ravel(), (xc[:, None] + xoff).ravel()]
        ),
        0,
        dx,
    )
    ylines = np.clip(
        np.concatenate(
            [by, ty, (yc[:, None] - yoff).ravel(), (yc[:, None] + yoff).ravel()]
        ),
        0,
        dy,
    )
    X, Y, Z = np.meshgrid(
        _collapse_lines(xlines), _collapse_lines(ylines), _collapse_lines(fz)
    )
    grd = pv.StructuredGrid(X, Y, Z)
    c = grd.cell_centers().points
    resin = np.zeros(len(c))
    resin = _mark_resin(resin, 0, c, thickness, xc, xhw, hx)
    resin = _mark_resin(resin, 1, c, thickness, yc, yhw, hy)
    face = c[:, 2] > thickness + tol
    is_resin = resin.astype(bool) & (~face)
    grd.cell_data["face"] = face
    grd.cell_data["resin"] = is_resin

    # Material coordinates before morph (for roll-to-cylinder viz / recovery).
    grd.point_data["x_mat"] = np.ascontiguousarray(grd.points[:, 0])
    grd.point_data["y_mat"] = np.ascontiguousarray(grd.points[:, 1])
    grd.point_data["z_mat"] = np.ascontiguousarray(grd.points[:, 2])

    # Flattened-for-FEA morph: walls track hw(z); foam bays become trapezoidal.
    if abs(kx) > 0.0 or abs(ky) > 0.0:
        morph_kerf_walls(
            grd, xcuts, ycuts, thickness, dx, dy, kx=kx, ky=ky, madd=madd
        )

    # Halo after morph so the band sits outside the *physical* tapered walls
    # (same hw(z) as ScoreField / morph). Pre-morph tagging misses open mouths.
    halo = np.zeros(grd.n_cells)
    if s_halo > 0:
        c_phys = grd.cell_centers().points
        halo = _mark_halo(halo, 0, c_phys, thickness, xc, xhw, hx, sx, s_halo)
        halo = _mark_halo(halo, 1, c_phys, thickness, yc, yhw, hy, sy, s_halo)
    is_resin = np.asarray(grd.cell_data["resin"], dtype=bool)
    is_face = np.asarray(grd.cell_data["face"], dtype=bool)
    grd.cell_data["halo"] = (halo > 0) & (~is_resin) & (~is_face)

    return grd


def main():
    p = argparse.ArgumentParser(
        description="Create mesh for core material domain with grooves marked as resin"
    )
    p.add_argument("--thickness", default=30.0, type=float)
    p.add_argument("--xcuts", default="[]")
    p.add_argument("--ycuts", default="[[-2, 25, 17., 2.0],[2, 25, - 17.0, 2.0]]")
    p.add_argument("--dx", default=50, type=float)
    p.add_argument("--dy", default=50, type=float)
    p.add_argument("--name", default="__temp.vts", help=".vts structured grid output")
    p.add_argument("--tface", default=4.0, help="Thickness of the face plate")
    p.add_argument("--kx", default=0.0, type=float)
    p.add_argument("--ky", default=0.0, type=float)
    args = p.parse_args()
    of = create_grooved_mesh(
        thickness=args.thickness,
        dx=args.dx,
        dy=args.dy,
        xcuts=eval(args.xcuts),
        ycuts=eval(args.ycuts),
        madd=[0],
        tface=args.tface,
        kx=args.kx,
        ky=args.ky,
    )
    of.save(args.name)
    print(f"Written grid to {args.name}")


if __name__ == "__main__":
    main()
