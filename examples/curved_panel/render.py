#!/usr/bin/env python3
"""Explain how kerf taper is computed from mould curvature.

Pedagogical intent (not “uniform FEA morph on an arc”):

1. **On the mould** — foam lands are rigid **rectangles** hinged about their
   centres. Kerfs are the **gaps between facing land edges**. Curvature opens
   or closes those gaps (signed arc: free face outer when opening).
2. **How taper is computed** — each wall rotates about the root; the slope
   that becomes the FEA law is
   ``slope = −sign(depth)·κ·pitch/2``,
   ``hw(z) = max(ε, hw₀ + slope·ζ)``.
3. **Flattened for FEA** — that law is applied as an interval-affine morph on
   the structured RVE (trapezoidal foam + tapered kerfs).

    uv run python examples/curved_panel/render.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyvista as pv

from b3_core.core.mesh import _hw_at, _physical_grooves, create_grooved_mesh

HERE = Path(__file__).parent
BASE = json.loads((HERE / "base.json").read_text())
STATES = [("closed", -0.012), ("flat", 0.0), ("opened", 0.012)]
PAIR_KX_OPEN = 0.012
PAIR_KX_CLOSED = -0.012
SLAB_HALF = 0.5
_Y0, _Y1 = 0.0, 1.0


# ---------------------------------------------------------------------------
# Flat FEA mesh (step 3 of the story)
# ---------------------------------------------------------------------------
def _mesh_flat_fea(kx: float) -> pv.UnstructuredGrid:
    """Flattened RVE: interval-affine morph implementing hw(z)."""
    mesh = create_grooved_mesh(
        thickness=BASE["thickness"],
        dx=BASE["dx"],
        dy=BASE["dy"],
        xcuts=BASE["xgr"],
        ycuts=BASE["ygr"],
        madd=tuple(BASE["madd"]),
        tface=0.0,
        kx=kx,
        ky=0.0,
    )
    grid = mesh.cast_to_unstructured_grid()
    for name in mesh.cell_data.keys():
        grid.cell_data[name] = mesh.cell_data[name]
    return grid


# ---------------------------------------------------------------------------
# Mould kinematics: rigid rectangular lands (step 1)
# ---------------------------------------------------------------------------
def _rect_point(
    x: float,
    z: float,
    xc: float,
    kappa: float,
    z_ref: float,
    x_mid: float,
) -> tuple[float, float]:
    """Rigid rectangle about land centre ``xc`` → signed cylinder."""
    if abs(kappa) < 1e-12:
        return float(x), float(z)
    R = 1.0 / kappa
    theta = kappa * (xc - x_mid)
    lx = x - xc
    r = R + (z - z_ref)
    ct, st = np.cos(theta), np.sin(theta)
    X = x_mid + r * st + lx * ct
    Z = z_ref - R + r * ct - lx * st
    return float(X), float(Z)


def _roll_points_to_cylinder(
    points: np.ndarray,
    kappa: float,
    *,
    x_mid: float,
    z_ref: float,
) -> np.ndarray:
    """Global polar roll (tests / FEA-roll utilities)."""
    pts = np.asarray(points, dtype=float)
    if abs(kappa) < 1e-12:
        return pts.copy()
    R = 1.0 / kappa
    out = pts.copy()
    x, z = out[:, 0], out[:, 2]
    theta = (x - x_mid) / R
    r = R + (z - z_ref)
    out[:, 0] = x_mid + r * np.sin(theta)
    out[:, 2] = z_ref - R + r * np.cos(theta)
    return out


def _extrude_quad(
    corners_xz: list[tuple[float, float]], *, resin: bool
) -> pv.UnstructuredGrid:
    (x0, z0), (x1, z1), (x2, z2), (x3, z3) = corners_xz
    area2 = (
        x0 * z1
        - x1 * z0
        + x1 * z2
        - x2 * z1
        + x2 * z3
        - x3 * z2
        + x3 * z0
        - x0 * z3
    )
    if area2 < 0:
        corners_xz = [corners_xz[0], corners_xz[3], corners_xz[2], corners_xz[1]]
    pts = []
    for y in (_Y0, _Y1):
        for x, z in corners_xz:
            pts.append([x, y, z])
    ug = pv.UnstructuredGrid(
        np.hstack([[8, 0, 1, 2, 3, 4, 5, 6, 7]]),
        np.array([pv.CellType.HEXAHEDRON]),
        np.asarray(pts, dtype=float),
    )
    ug.cell_data["resin"] = np.array([resin], dtype=bool)
    ug.cell_data["face"] = np.array([False])
    return ug


def _foam_and_kerf_layout(dx: float, pitch: float, width: float, offset: float):
    """Foam spans (L,R) and kerf centres (incl. domain-edge kerfs)."""
    hw = 0.5 * width
    c0s: list[float] = []
    for left in np.arange(offset - pitch - hw, dx + pitch, pitch):
        right = left + width
        if right <= 1e-9 or left >= dx - 1e-9:
            continue
        c0s.append(0.5 * (left + right))
    c0s_arr = np.asarray(c0s, dtype=float)
    foam_spans: list[tuple[float, float]] = []
    if len(c0s_arr) == 0:
        return [(0.0, dx)], c0s_arr
    for i, c0 in enumerate(c0s_arr):
        L = float(min(dx, max(0.0, c0 + hw)))
        R = (
            float(min(dx, max(0.0, c0s_arr[i + 1] - hw)))
            if i + 1 < len(c0s_arr)
            else dx
        )
        if R - L > 1e-6:
            foam_spans.append((L, R))
    first_R = float(min(dx, max(0.0, c0s_arr[0] - hw)))
    if first_R > 1e-6:
        foam_spans.insert(0, (0.0, first_R))
    return foam_spans, c0s_arr


def _kerf_land_centres(
    c0: float,
    hw0: float,
    pitch: float,
    foam_spans: list[tuple[float, float]],
) -> tuple[float, float]:
    """Hinge centres of the two lands facing kerf ``c0``.

    Interior faces resolve to the neighbouring foam-span centre. Domain-edge
    partials have no outer land in-domain: use the ideal lattice neighbour at
    ``c0 ± pitch/2`` so the opening angle stays ``κ·pitch`` (not the wrong
    ``κ·2·pitch`` from a ``c0 ± pitch`` phantom).
    """
    out: list[float] = []
    for side, ideal_wall in (("left", c0 - hw0), ("right", c0 + hw0)):
        found = None
        for a, b in foam_spans:
            if side == "left" and abs(b - ideal_wall) < 1e-4:
                found = 0.5 * (a + b)
                break
            if side == "right" and abs(a - ideal_wall) < 1e-4:
                found = 0.5 * (a + b)
                break
        if found is None:
            found = c0 - 0.5 * pitch if side == "left" else c0 + 0.5 * pitch
        out.append(float(found))
    return out[0], out[1]


def _map_kerf_gap_point(
    x: float,
    z: float,
    x_L_ideal: float,
    x_R_ideal: float,
    xc_L: float,
    xc_R: float,
    kappa: float,
    z_ref: float,
    x_mid: float,
) -> tuple[float, float]:
    """Map a flat-x point in a kerf gap into mould space.

    Ideal land faces use their land frames. Domain-clipped edges (partial edge
    kerfs) sit between those faces and are linear in the gap parameter so edge
    opening matches the mid-panel ``κ·pitch`` geometry.
    """
    p_L = _rect_point(x_L_ideal, z, xc_L, kappa, z_ref, x_mid)
    p_R = _rect_point(x_R_ideal, z, xc_R, kappa, z_ref, x_mid)
    span = x_R_ideal - x_L_ideal
    if abs(span) < 1e-12:
        return p_L
    t = (x - x_L_ideal) / span
    return (
        (1.0 - t) * p_L[0] + t * p_R[0],
        (1.0 - t) * p_L[1] + t * p_R[1],
    )


def _kerf_gap_widths(
    c0: float,
    *,
    hw0: float,
    pitch: float,
    dx: float,
    kappa: float,
    z_root: float,
    z_mouth: float,
    z_ref: float,
    x_mid: float,
    foam_spans: list[tuple[float, float]],
) -> tuple[float, float, float]:
    """Mouth/root chord widths and ideal wall angle (deg) for kerf ``c0``.

    Returns ``(w_root, w_mouth, ideal_angle_deg)``. Domain-clipped edge kerfs
    use the same ideal land frames as interior ones; widths scale with the
    in-domain gap fraction.
    """
    x_Li, x_Ri = c0 - hw0, c0 + hw0
    x_L, x_R = max(0.0, x_Li), min(dx, x_Ri)
    xc_L, xc_R = _kerf_land_centres(c0, hw0, pitch, foam_spans)

    def chord(z: float) -> float:
        p0 = np.array(
            _map_kerf_gap_point(x_L, z, x_Li, x_Ri, xc_L, xc_R, kappa, z_ref, x_mid)
        )
        p1 = np.array(
            _map_kerf_gap_point(x_R, z, x_Li, x_Ri, xc_L, xc_R, kappa, z_ref, x_mid)
        )
        return float(np.linalg.norm(p1 - p0))

    # Ideal full walls (even outside domain) give the hinge angle κ·pitch.
    wL0 = np.array(_rect_point(x_Li, z_root, xc_L, kappa, z_ref, x_mid))
    wL1 = np.array(_rect_point(x_Li, z_mouth, xc_L, kappa, z_ref, x_mid))
    wR0 = np.array(_rect_point(x_Ri, z_root, xc_R, kappa, z_ref, x_mid))
    wR1 = np.array(_rect_point(x_Ri, z_mouth, xc_R, kappa, z_ref, x_mid))
    vL, vR = wL1 - wL0, wR1 - wR0
    cos = float(np.dot(vL, vR) / (np.linalg.norm(vL) * np.linalg.norm(vR) + 1e-15))
    ang = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    return chord(z_root), chord(z_mouth), ang


def _mesh_curved_mould(kx: float) -> pv.UnstructuredGrid:
    """On-mould picture: rectangular foam lands; kerf = gap between land edges.

    This is the physical picture that *motivates* the FEA taper law
    ``slope = −sign(depth)·κ·pitch/2`` — not a roll of the morphed FEA mesh.
    """
    if abs(kx) < 1e-12:
        return _mesh_flat_fea(0.0)

    th = float(BASE["thickness"])
    dx = float(BASE["dx"])
    offset, pitch, depth, width = map(float, BASE["xgr"][0])
    hw0 = 0.5 * width
    foam_spans, c0s = _foam_and_kerf_layout(dx, pitch, width, offset)
    if len(c0s) == 0:
        return _mesh_flat_fea(kx)

    if depth < 0:
        z_lig0, z_lig1 = 0.0, th + depth  # ligament under top-mouth grooves
    else:
        z_lig0, z_lig1 = depth, th

    z_ref = 0.5 * th
    x_mid = 0.5 * dx
    parts: list[pv.UnstructuredGrid] = []

    def add_block(
        flat_xz: list[tuple[float, float]], xc: float, *, resin: bool
    ) -> None:
        (x0, _), (x1, _), (x2, _), (x3, _) = flat_xz
        if max(abs(x1 - x0), abs(x2 - x3)) < 1e-9:
            return
        corners = [_rect_point(x, z, xc, kx, z_ref, x_mid) for x, z in flat_xz]
        parts.append(_extrude_quad(corners, resin=resin))

    def add_gap(
        x_L: float,
        x_R: float,
        z0: float,
        z1: float,
        x_L_ideal: float,
        x_R_ideal: float,
        xc_L: float,
        xc_R: float,
        *,
        resin: bool,
    ) -> None:
        """Fill kerf gap (possibly domain-clipped) between ideal land faces."""
        if x_R - x_L < 1e-9:
            return
        corners = [
            _map_kerf_gap_point(x_L, z0, x_L_ideal, x_R_ideal, xc_L, xc_R, kx, z_ref, x_mid),
            _map_kerf_gap_point(x_R, z0, x_L_ideal, x_R_ideal, xc_L, xc_R, kx, z_ref, x_mid),
            _map_kerf_gap_point(x_R, z1, x_L_ideal, x_R_ideal, xc_L, xc_R, kx, z_ref, x_mid),
            _map_kerf_gap_point(x_L, z1, x_L_ideal, x_R_ideal, xc_L, xc_R, kx, z_ref, x_mid),
        ]
        parts.append(_extrude_quad(corners, resin=resin))

    # Rectangular foam lands (full thickness)
    for a, b in foam_spans:
        xc = 0.5 * (a + b)
        add_block([(a, 0.0), (b, 0.0), (b, th), (a, th)], xc, resin=False)

    # Kerf = gap between facing land edges (hinge opening = taper motivation).
    # Edge kerfs are domain-clipped but still use ideal wall frames so mid and
    # edge share the same κ·pitch opening.
    for c0 in c0s:
        c0 = float(c0)
        x_L_ideal = c0 - hw0
        x_R_ideal = c0 + hw0
        x_L = max(0.0, x_L_ideal)
        x_R = min(dx, x_R_ideal)
        if x_R - x_L < 1e-9:
            continue
        xc_L, xc_R = _kerf_land_centres(c0, hw0, pitch, foam_spans)

        # Groove band: resin fills the hinged gap
        if depth < 0:
            z0, z1 = th + depth, th
        else:
            z0, z1 = 0.0, depth
        add_gap(x_L, x_R, z0, z1, x_L_ideal, x_R_ideal, xc_L, xc_R, resin=True)

        # Foam ligament under/above the groove (hinge floor)
        if abs(z_lig1 - z_lig0) > 1e-6:
            add_gap(
                x_L, x_R, z_lig0, z_lig1, x_L_ideal, x_R_ideal, xc_L, xc_R, resin=False
            )

    if not parts:
        return _mesh_flat_fea(kx)
    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p)
    return out


def _mesh_curved_fea(kx: float) -> pv.UnstructuredGrid:
    """Alias: curved mould picture (rectangular foam + hinged kerf gaps)."""
    return _mesh_curved_mould(kx)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _mid_y_slab(mesh: pv.DataSet, half: float = SLAB_HALF) -> pv.DataSet:
    y0 = 0.5 * (mesh.bounds[2] + mesh.bounds[3])
    kept = mesh.clip(normal="y", origin=(0.0, y0 - half, 0.0), invert=False)
    return kept.clip(normal="-y", origin=(0.0, y0 + half, 0.0), invert=False)


def _set_side_camera(plotter: pv.Plotter, mesh: pv.DataSet) -> None:
    plotter.enable_parallel_projection()
    b = mesh.bounds
    cx = 0.5 * (b[0] + b[1])
    cy = 0.5 * (b[2] + b[3])
    cz = 0.5 * (b[4] + b[5])
    sx = max(b[1] - b[0], 1.0)
    sz = max(b[5] - b[4], 1.0)
    dist = max(sx, sz) * 2.2
    plotter.camera.position = (cx, cy - dist, cz)
    plotter.camera.focal_point = (cx, cy, cz)
    plotter.camera.up = (0.0, 0.0, 1.0)
    plotter.camera.parallel_scale = 0.55 * max(sx, sz)


def _taper_law(kx: float) -> dict:
    """Quantities that define how FEA taper is computed from κ."""
    th = float(BASE["thickness"])
    dx = float(BASE["dx"])
    offset, pitch, depth, width = map(float, BASE["xgr"][0])
    hw0 = 0.5 * width
    slope = -np.sign(depth) * kx * pitch / 2.0
    if depth < 0:
        z_root, z_mouth = th + depth, th
        zeta_mouth = abs(depth)
    else:
        z_root, z_mouth = depth, 0.0
        zeta_mouth = abs(depth)
    hw_root = _hw_at(hw0, depth, slope, z_root, th)
    hw_mouth = _hw_at(hw0, depth, slope, z_mouth, th)
    # Opening angle of linear walls in material xz
    dz = z_mouth - z_root
    vL = np.array([-(hw_mouth - hw_root), dz], dtype=float)
    vR = np.array([+(hw_mouth - hw_root), dz], dtype=float)
    nL = vL / (np.linalg.norm(vL) + 1e-15)
    nR = vR / (np.linalg.norm(vR) + 1e-15)
    ang = float(np.degrees(np.arccos(np.clip(np.dot(nL, nR), -1.0, 1.0))))
    ang = min(ang, 180.0 - ang)
    hinge_deg = abs(np.degrees(kx * pitch))
    return {
        "pitch": pitch,
        "hw0": hw0,
        "depth": depth,
        "slope": slope,
        "z_root": z_root,
        "z_mouth": z_mouth,
        "hw_root": hw_root,
        "hw_mouth": hw_mouth,
        "kerf_angle_deg": ang,
        "hinge_deg": hinge_deg,
        "formula": f"slope = −sign(d)·κ·p/2 = {slope:+.4f}",
    }


def _interior_c0() -> float:
    dx = float(BASE["dx"])
    offset, pitch, _, width = map(float, BASE["xgr"][0])
    _, c0s = _foam_and_kerf_layout(dx, pitch, width, offset)
    interior = [float(c) for c in c0s if 5.0 < c < dx - 5.0]
    return interior[len(interior) // 2] if interior else float(c0s[len(c0s) // 2])


def _add_taper_callout(
    plotter: pv.Plotter,
    kx: float,
    *,
    curved: bool,
    mesh: pv.DataSet,
) -> None:
    """Annotate the taper law and wall angle on one interior kerf."""
    law = _taper_law(kx)
    c0 = _interior_c0()
    th = float(BASE["thickness"])
    dx = float(BASE["dx"])
    z_ref, x_mid = 0.5 * th, 0.5 * dx
    y = float(mesh.points[:, 1].mean()) - 0.05
    hw0 = law["hw0"]
    pitch = law["pitch"]
    z0, z1 = law["z_root"], law["z_mouth"]
    offset, _, _, width = map(float, BASE["xgr"][0])
    foam_spans, _ = _foam_and_kerf_layout(dx, pitch, width, offset)
    land_L, land_R = max(0.0, c0 - hw0), min(dx, c0 + hw0)
    xc_L, xc_R = _kerf_land_centres(c0, hw0, pitch, foam_spans)

    def pt(x: float, z: float, xc: float) -> list[float]:
        if curved and abs(kx) > 1e-12:
            X, Z = _rect_point(x, z, xc, kx, z_ref, x_mid)
        else:
            X, Z = x, z
        return [X, y, Z]

    # Wall lines: curved = land faces (hinge); flat = FEA hw(z) walls
    if curved:
        left = np.array([pt(land_L, z0, xc_L), pt(land_L, z1, xc_L)])
        right = np.array([pt(land_R, z0, xc_R), pt(land_R, z1, xc_R)])
        ang = law["hinge_deg"]
        txt = (
            f"hinge opening  {ang:.1f} deg\n"
            f"|k|*pitch = {abs(kx * pitch):.4f}\n"
            f"-> slope = k*p/2"
        )
    else:
        left = np.array(
            [
                pt(c0 - law["hw_root"], z0, c0),
                pt(c0 - law["hw_mouth"], z1, c0),
            ]
        )
        right = np.array(
            [
                pt(c0 + law["hw_root"], z0, c0),
                pt(c0 + law["hw_mouth"], z1, c0),
            ]
        )
        ang = law["kerf_angle_deg"]
        txt = (
            f"FEA walls  {ang:.1f} deg\n"
            f"hw(z)=hw0+slope*zeta\n"
            f"slope={law['slope']:+.4f}"
        )

    for poly in (left, right):
        plotter.add_mesh(
            pv.lines_from_points(poly, close=False),
            color="#00e5ff",
            line_width=4,
            render_lines_as_tubes=True,
        )
    lab = np.array(pt(c0, 0.5 * (z0 + z1), c0 if not curved else 0.5 * (xc_L + xc_R)))
    plotter.add_point_labels(
        [lab],
        [txt],
        font_size=12,
        text_color="#0d0d0d",
        point_color="#00e5ff",
        point_size=6,
        always_visible=True,
        shape_opacity=0.92,
        shape_color="white",
        fill_shape=True,
    )


def _add_side_view(
    plotter: pv.Plotter,
    mesh: pv.DataSet,
    title: str,
    *,
    kx: float | None = None,
    curved: bool = False,
    show_taper_callout: bool = False,
) -> None:
    slab = _mid_y_slab(mesh)
    if slab.n_cells == 0:
        slab = mesh
    if "resin" in slab.array_names:
        resin = slab.threshold(0.5, scalars="resin")
        foam = slab.threshold(0.5, scalars="resin", invert=True)
    else:
        resin, foam = None, slab
    if foam is not None and foam.n_cells:
        plotter.add_mesh(
            foam,
            color="gainsboro",
            opacity=1.0,
            show_edges=True,
            edge_color="silver",
            line_width=0.5,
        )
    if resin is not None and resin.n_cells:
        plotter.add_mesh(
            resin,
            color="firebrick",
            opacity=1.0,
            show_edges=True,
            edge_color="darkred",
            line_width=0.6,
        )
    if show_taper_callout and kx is not None:
        _add_taper_callout(plotter, kx, curved=curved, mesh=mesh)
    plotter.add_text(title, font_size=11)
    plotter.background_color = "white"
    _set_side_camera(plotter, slab if slab.n_cells else mesh)


def _write_open_flat_closed(out: Path) -> None:
    grid = pv.Plotter(shape=(1, 3), off_screen=True, window_size=(1680, 520))
    for col, (name, kx) in enumerate(STATES):
        mesh = _mesh_flat_fea(kx)
        single = pv.Plotter(off_screen=True, window_size=(640, 480))
        _add_side_view(
            single,
            mesh,
            f"{name}  kx={kx:+.3f}  FEA hw(z)",
            kx=kx,
            curved=False,
            show_taper_callout=True,
        )
        single.screenshot(str(out / f"groove_{name}.png"))
        single.close()
        grid.subplot(0, col)
        _add_side_view(
            grid,
            mesh,
            f"{name}  kx={kx:+.3f}",
            kx=kx,
            curved=False,
            show_taper_callout=True,
        )
    grid.screenshot(str(out / "groove_strip.png"))
    grid.close()


def _write_curved_vs_flat(
    out: Path,
    kx: float,
    *,
    mode: str,
    write_singles: bool = False,
) -> None:
    curved = _mesh_curved_mould(kx)
    flat = _mesh_flat_fea(kx)
    kerf_word = "open" if mode == "open" else "closed"
    law = _taper_law(kx)

    if write_singles:
        for tag, mesh, title, is_c in (
            (
                "curved",
                curved,
                f"mould  kx={kx:+.3f}  rectangular foam, hinged gaps",
                True,
            ),
            (
                "flattened",
                flat,
                f"FEA  kx={kx:+.3f}  hw(z) morph  slope={law['slope']:+.4f}",
                False,
            ),
        ):
            pl = pv.Plotter(off_screen=True, window_size=(720, 480))
            _add_side_view(
                pl,
                mesh,
                title,
                kx=kx,
                curved=is_c,
                show_taper_callout=True,
            )
            pl.screenshot(str(out / f"groove_{tag}.png"))
            pl.close()

    board = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1500, 540))
    board.subplot(0, 0)
    action = "opens" if mode == "open" else "pinches"
    _add_side_view(
        board,
        curved,
        f"1 · on mould   kx={kx:+.3f}   rect. foam, hinge {action} kerfs",
        kx=kx,
        curved=True,
        show_taper_callout=True,
    )
    board.subplot(0, 1)
    _add_side_view(
        board,
        flat,
        f"2 · FEA model   slope=-sign(d)*k*p/2 -> hw(z)  ({kerf_word})",
        kx=kx,
        curved=False,
        show_taper_callout=True,
    )
    name = (
        "groove_curved_vs_flat.png"
        if mode == "open"
        else "groove_curved_vs_flat_closed.png"
    )
    board.screenshot(str(out / name))
    board.close()


def main() -> int:
    try:
        pv.start_xvfb()
    except Exception:
        pass
    out = HERE / "img"
    out.mkdir(parents=True, exist_ok=True)
    _write_open_flat_closed(out)
    _write_curved_vs_flat(out, PAIR_KX_OPEN, mode="open", write_singles=True)
    _write_curved_vs_flat(out, PAIR_KX_CLOSED, mode="closed", write_singles=False)
    for name in (
        "groove_closed",
        "groove_flat",
        "groove_opened",
        "groove_strip",
        "groove_curved",
        "groove_flattened",
        "groove_curved_vs_flat",
        "groove_curved_vs_flat_closed",
    ):
        print(f"wrote {out / f'{name}.png'}")
    # Print the law for the open case (docs / console)
    for kx, tag in ((PAIR_KX_OPEN, "open"), (PAIR_KX_CLOSED, "closed")):
        law = _taper_law(kx)
        print(
            f"{tag}: {law['formula']}  "
            f"hw_root={law['hw_root']:.3f} hw_mouth={law['hw_mouth']:.3f}  "
            f"hinge∠={law['hinge_deg']:.2f}° FEA∠={law['kerf_angle_deg']:.2f}°"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
