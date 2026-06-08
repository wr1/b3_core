#!/usr/bin/env python3

import numpy as np
import pyvista as pv
import argparse


def create_grooves(cuts, bnd, meshadd=(-0.5, -0.2, 0, 0.2, 0.5), tol=1e-6, kappa=0.0):
    """Per groove-instance left/right walls, signed depth, and curvature taper slope.

    ``kappa`` (1/length) bends the panel so the grooves open or close. The walls
    rotate about the groove root corner: the per-instance taper ``slope`` is
    ``sigma * kappa * pitch / 2`` with ``sigma = -sign(depth)`` so a single
    curvature opens grooves mouthing on one face and closes those on the other.
    ``slope`` is consumed by ``create_grooved_mesh`` to evaluate a z-dependent
    half-width; at ``kappa == 0`` it is all zeros and the grooves stay
    rectangular (bit-for-bit the original mesh).
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

    return bot, top, height, slope


def create_z_mesh(cuts, th, meshadd=(-0.5, -0.2, 0, 0.2, 0.5), absadd=()):
    fz = np.concatenate([0 + np.array(meshadd), th + np.array(meshadd)])
    for i in cuts:
        for j in meshadd:
            fz = np.append(fz, [i[2] + j, th + i[2] + j])
    fz = np.append(fz, absadd)
    return np.unique(np.minimum(th, np.maximum(0, fz)))


def _mark_resin(resin, coord, c, thickness, centres, halfwidths, depths, slopes):
    """Mark cells whose centre falls inside each (possibly tapered) groove.

    ``coord`` selects the in-plane axis (0 for x-grooves, 1 for y-grooves). The
    half-width is evaluated at the cell's z via the curvature ``slope``, hinged
    at the groove root corner, so opened grooves flare toward the mouth and
    closed ones pinch shut (clamped to zero). At ``slope == 0`` this reduces to
    the original constant-width ``bot < c < top`` test.
    """
    for c0, hw, d, s in zip(centres, halfwidths, depths, slopes, strict=True):
        if d > 0:  # mouth at z = 0, root at z = d
            in_z = c[:, 2] < d
            zeta = d - c[:, 2]
        else:  # mouth at z = thickness, root at z = thickness + d
            in_z = c[:, 2] > thickness + d
            zeta = c[:, 2] - (thickness + d)
        hw_z = np.clip(hw + s * zeta, 0, None)
        resin = np.maximum(resin, (np.abs(c[:, coord] - c0) < hw_z) & in_z)
    return resin


def _mark_halo(frac, coord, c, thickness, centres, halfwidths, depths, slopes, s_halo):
    """Geometric kerf-damage halo: cells just outside each groove wall.

    Returns a value >0 for cells within ``s_halo`` of a (z-dependent) kerf wall,
    used only to flag the band for visualisation; the actual graded material is
    evaluated from the ScoreField at integration points by the numpy backend.
    """
    for c0, hw, d, s in zip(centres, halfwidths, depths, slopes, strict=True):
        if d > 0:
            in_z = c[:, 2] < d
            zeta = d - c[:, 2]
        else:
            in_z = c[:, 2] > thickness + d
            zeta = c[:, 2] - (thickness + d)
        hw_z = np.clip(hw + s * zeta, 0, None)
        gap = np.abs(c[:, coord] - c0) - hw_z
        g = np.where((gap >= 0) & (gap < s_halo) & in_z,
                     np.clip(1.0 - gap / s_halo, 0.0, 1.0), 0.0)
        frac = np.maximum(frac, g)
    return frac


def _collapse_lines(vals, tol=1e-6):
    """Sorted unique grid lines, merging any closer than ``tol``.

    Concatenating walls, madd shifts and opened-mouth extents produces lines that
    coincide only to within floating-point error; ``np.unique`` keeps both and
    leaves ~1e-15-wide sliver cells whose near-zero Jacobian makes the MFEM
    operator singular. Snapping within ``tol`` (sub-micron) removes them without
    touching real mm-scale geometry.
    """
    v = np.sort(np.asarray(vals, dtype=float))
    keep = np.concatenate(([True], np.diff(v) > tol))
    return v[keep]


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
    bx, tx, hx, sx = create_grooves(xcuts, dx, meshadd=madd, kappa=kx)
    by, ty, hy, sy = create_grooves(ycuts, dy, meshadd=madd, kappa=ky)
    tol = 1e-3
    fz = create_z_mesh(
        xcuts + ycuts, thickness + tface, meshadd=madd, absadd=[thickness + tol]
    )
    if s_halo > 0:
        # z sub-lines just past each groove root (a cut surface) so the root halo
        # is resolved; root at z=depth (depth>0) or z=thickness+depth (depth<0).
        hz = np.linspace(0.0, 1.0, 3)[1:]
        roots = [d + s_halo * hz if d > 0 else (thickness + d) - s_halo * hz
                 for cut in xcuts + ycuts if (d := cut[2]) != 0]
        if roots:
            fz = np.append(fz, np.clip(np.concatenate(roots), 0.0, thickness + tface))
    # Groove centres and nominal half-widths. An opened groove flares from the
    # nominal half-width at the root to ``*_mouth`` at the surface; lay several
    # in-plane grid lines across that taper zone (this axis only, so y/z stay
    # coarse) so cell-centre marking resolves the trapezoid smoothly instead of
    # jumping as the single mouth line shifts with curvature. At kappa=0 the zone
    # has zero width and every extra line collapses onto the nominal wall.
    xc, xhw = 0.5 * (bx + tx), 0.5 * (tx - bx)
    yc, yhw = 0.5 * (by + ty), 0.5 * (ty - by)
    x_mouth = xhw + np.maximum(0.0, sx * np.abs(hx))
    y_mouth = yhw + np.maximum(0.0, sy * np.abs(hy))
    fracs = np.linspace(0.0, 1.0, 5)[1:]  # taper-zone sub-lines, nominal wall excluded
    xoff = xhw[:, None] + np.outer(x_mouth - xhw, fracs)
    yoff = yhw[:, None] + np.outer(y_mouth - yhw, fracs)
    if s_halo > 0:
        # extra in-plane lines across the halo band so the graded resin field is
        # resolved by a few cells (sub-element sampling smooths within).
        hf = np.linspace(0.0, 1.0, 3)[1:]
        xoff = np.concatenate([xoff, xhw[:, None] + s_halo * hf[None, :]], axis=1)
        yoff = np.concatenate([yoff, yhw[:, None] + s_halo * hf[None, :]], axis=1)
    xlines = np.clip(
        np.concatenate([bx, tx, (xc[:, None] - xoff).ravel(), (xc[:, None] + xoff).ravel()]),
        0, dx,
    )
    ylines = np.clip(
        np.concatenate([by, ty, (yc[:, None] - yoff).ravel(), (yc[:, None] + yoff).ravel()]),
        0, dy,
    )
    X, Y, Z = np.meshgrid(
        _collapse_lines(xlines), _collapse_lines(ylines), _collapse_lines(fz)
    )
    grd = pv.StructuredGrid(X, Y, Z)
    c = grd.cell_centers().points
    resin = np.zeros(len(c))
    resin = _mark_resin(resin, 0, c, thickness, xc, xhw, hx, sx)
    resin = _mark_resin(resin, 1, c, thickness, yc, yhw, hy, sy)
    face = c[:, 2] > thickness + tol
    is_resin = resin.astype(bool) & (~face)
    grd.cell_data["face"] = face
    grd.cell_data["resin"] = is_resin

    # Mark the halo band (foam cells within s_halo of a groove wall/root) for viz
    # only; the material itself comes from the ScoreField at integration points.
    halo = np.zeros(len(c))
    if s_halo > 0:
        halo = _mark_halo(halo, 0, c, thickness, xc, xhw, hx, sx, s_halo)
        halo = _mark_halo(halo, 1, c, thickness, yc, yhw, hy, sy, s_halo)
    grd.cell_data["halo"] = (halo > 0) & (~is_resin) & (~face)
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
    args = p.parse_args()
    of = create_grooved_mesh(
        thickness=args.thickness,
        dx=args.dx,
        dy=args.dy,
        xcuts=eval(args.xcuts),
        ycuts=eval(args.ycuts),
        madd=[0],
        tface=args.tface,
    )
    of.save(args.name)
    print(f"Written grid to {args.name}")


if __name__ == "__main__":
    main()
