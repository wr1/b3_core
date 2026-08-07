"""Curved-panel render: mould hinge → FEA taper law.

Curved panel = rectangular foam + kerf gaps between land faces (motivates κ).
Flat panel = create_grooved_mesh morph with slope = −sign(d)·κ·pitch/2.

Edge kerfs are domain-clipped half-gaps that still open at the same κ·pitch
rate as interior ones (ideal land frames at c0 ± pitch/2, not c0 ± pitch).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from b3_core.core.mesh import _hw_at, _physical_grooves, create_grooved_mesh

ROOT = Path(__file__).resolve().parents[1]
RENDER_PATH = ROOT / "examples" / "curved_panel" / "render.py"
BASE_PATH = ROOT / "examples" / "curved_panel" / "base.json"


def _load_render():
    spec = importlib.util.spec_from_file_location("curved_panel_render", RENDER_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def render():
    return _load_render()


@pytest.fixture(scope="module")
def base():
    return json.loads(BASE_PATH.read_text())


@pytest.fixture(scope="module")
def layout(render, base):
    dx = float(base["dx"])
    offset, pitch, depth, width = map(float, base["xgr"][0])
    foam, c0s = render._foam_and_kerf_layout(dx, pitch, width, offset)
    return {
        "dx": dx,
        "th": float(base["thickness"]),
        "offset": offset,
        "pitch": pitch,
        "depth": depth,
        "width": width,
        "hw0": 0.5 * width,
        "foam": foam,
        "c0s": np.asarray(c0s, dtype=float),
        "z_ref": 0.5 * float(base["thickness"]),
        "x_mid": 0.5 * dx,
        "z_root": float(base["thickness"]) + depth if depth < 0 else 0.0,
        "z_mouth": float(base["thickness"]) if depth < 0 else depth,
    }


def test_taper_law_slope_from_kappa_pitch(render, base):
    """FEA slope is computed as −sign(depth)·κ·pitch/2."""
    pitch = float(base["xgr"][0][1])
    depth = float(base["xgr"][0][2])
    for kx in (0.012, -0.012, 0.0):
        law = render._taper_law(kx)
        expect = -np.sign(depth) * kx * pitch / 2.0
        assert law["slope"] == pytest.approx(expect)
        assert law["pitch"] == pytest.approx(pitch)


def test_hinge_angle_matches_fea_wall_angle_small(render, base):
    """For small open κ, hinge ∠ (κ·pitch) ≈ FEA wall angle (2 atan|slope|).

    Closed mouths can clamp at ε so FEA ∠ is slightly smaller — only open is exact.
    """
    law = render._taper_law(0.012)
    assert law["hinge_deg"] == pytest.approx(law["kerf_angle_deg"], abs=0.05)
    law_c = render._taper_law(-0.012)
    assert law_c["kerf_angle_deg"] > 0.0
    assert abs(law_c["hinge_deg"] - law_c["kerf_angle_deg"]) < 1.0


def test_rect_point_land_is_rectangle(render):
    """Each foam land after rigid map is a perfect rectangle."""
    x_mid, z_ref = 25.0, 15.0
    a, b, xc = 11.5, 18.5, 15.0
    for kappa in (0.012, -0.012):
        P = [
            np.array(render._rect_point(a, 0.0, xc, kappa, z_ref, x_mid)),
            np.array(render._rect_point(b, 0.0, xc, kappa, z_ref, x_mid)),
            np.array(render._rect_point(b, 30.0, xc, kappa, z_ref, x_mid)),
            np.array(render._rect_point(a, 30.0, xc, kappa, z_ref, x_mid)),
        ]
        w_bot = np.linalg.norm(P[1] - P[0])
        w_top = np.linalg.norm(P[2] - P[3])
        assert w_bot == pytest.approx(7.0, rel=1e-6)
        assert w_top == pytest.approx(w_bot, rel=1e-6)


def test_curved_mould_has_resin_and_foam(render):
    for kx in (0.012, -0.012):
        m = render._mesh_curved_mould(kx)
        res = np.asarray(m.cell_data["resin"], dtype=bool)
        assert res.any() and (~res).any()


def test_layout_includes_domain_edge_kerfs(layout):
    """Kerf lattice matches FEA physical grooves: centres at 0, pitch, …, dx."""
    c0s = layout["c0s"]
    pitch = layout["pitch"]
    dx = layout["dx"]
    assert c0s[0] == pytest.approx(0.0)
    assert c0s[-1] == pytest.approx(dx)
    assert np.allclose(np.diff(c0s), pitch)
    # Foam only between kerfs — no land outside the first/last edge kerf.
    assert layout["foam"][0][0] == pytest.approx(layout["hw0"])
    assert layout["foam"][-1][1] == pytest.approx(dx - layout["hw0"])


def test_kerf_land_centres_pitch_half_not_pitch(render, layout):
    """Edge phantom centres are c0 ± pitch/2 (regression: was c0 ± pitch)."""
    hw0, pitch = layout["hw0"], layout["pitch"]
    foam = layout["foam"]

    # Interior: resolve to real foam centres pitch apart.
    xc_L, xc_R = render._kerf_land_centres(20.0, hw0, pitch, foam)
    assert xc_L == pytest.approx(15.0)
    assert xc_R == pytest.approx(25.0)
    assert (xc_R - xc_L) == pytest.approx(pitch)

    # Domain-edge: ideal lattice neighbours — half-pitch from c0, not full pitch.
    xc_L0, xc_R0 = render._kerf_land_centres(0.0, hw0, pitch, foam)
    assert xc_L0 == pytest.approx(-0.5 * pitch)
    assert xc_R0 == pytest.approx(+0.5 * pitch)
    assert (xc_R0 - xc_L0) == pytest.approx(pitch)
    # Explicit regression guard against the old over-open phantom.
    assert xc_L0 != pytest.approx(-pitch)
    assert xc_R0 != pytest.approx(+pitch)

    xc_L1, xc_R1 = render._kerf_land_centres(layout["dx"], hw0, pitch, foam)
    assert xc_L1 == pytest.approx(layout["dx"] - 0.5 * pitch)
    assert xc_R1 == pytest.approx(layout["dx"] + 0.5 * pitch)
    assert (xc_R1 - xc_L1) == pytest.approx(pitch)


def test_ideal_wall_angle_same_at_edge_and_mid(render, layout):
    """Every kerf — including domain-edge partials — has ideal wall ∠ ≈ κ·pitch."""
    kx = 0.012
    pitch = layout["pitch"]
    expect_deg = np.degrees(abs(kx) * pitch)
    for c0 in layout["c0s"]:
        _, _, ang = render._kerf_gap_widths(
            float(c0),
            hw0=layout["hw0"],
            pitch=pitch,
            dx=layout["dx"],
            kappa=kx,
            z_root=layout["z_root"],
            z_mouth=layout["z_mouth"],
            z_ref=layout["z_ref"],
            x_mid=layout["x_mid"],
            foam_spans=layout["foam"],
        )
        assert ang == pytest.approx(expect_deg, abs=0.15), f"c0={c0}"


def test_edge_kerf_is_proportional_slice_of_mid(render, layout):
    """Half-width edge gap is a proportional slice of the full mid gap.

    Guards domain-cut mapping: wall at x=0 is gap-interpolated, not treated as
    a land face (which collapsed the root / over-opened the mouth).
    """
    for kx in (0.012, -0.012):
        mid = render._kerf_gap_widths(
            20.0,
            hw0=layout["hw0"],
            pitch=layout["pitch"],
            dx=layout["dx"],
            kappa=kx,
            z_root=layout["z_root"],
            z_mouth=layout["z_mouth"],
            z_ref=layout["z_ref"],
            x_mid=layout["x_mid"],
            foam_spans=layout["foam"],
        )
        edge = render._kerf_gap_widths(
            0.0,
            hw0=layout["hw0"],
            pitch=layout["pitch"],
            dx=layout["dx"],
            kappa=kx,
            z_root=layout["z_root"],
            z_mouth=layout["z_mouth"],
            z_ref=layout["z_ref"],
            x_mid=layout["x_mid"],
            foam_spans=layout["foam"],
        )
        w_mid_r, w_mid_m, ang_mid = mid
        w_edge_r, w_edge_m, ang_edge = edge
        assert ang_edge == pytest.approx(ang_mid, abs=0.05)
        assert w_edge_r == pytest.approx(0.5 * w_mid_r, rel=0.05)
        assert w_edge_m == pytest.approx(0.5 * w_mid_m, rel=0.05)
        # Open: mouth > root; closed: mouth < root (both via same geometry).
        if kx > 0:
            assert w_edge_m > w_edge_r
            assert w_mid_m > w_mid_r
        else:
            assert w_edge_m < w_edge_r
            assert w_mid_m < w_mid_r

        # Right domain edge mirrors left.
        edge_r = render._kerf_gap_widths(
            layout["dx"],
            hw0=layout["hw0"],
            pitch=layout["pitch"],
            dx=layout["dx"],
            kappa=kx,
            z_root=layout["z_root"],
            z_mouth=layout["z_mouth"],
            z_ref=layout["z_ref"],
            x_mid=layout["x_mid"],
            foam_spans=layout["foam"],
        )
        assert edge_r[0] == pytest.approx(w_edge_r, rel=0.05)
        assert edge_r[1] == pytest.approx(w_edge_m, rel=0.05)


def test_edge_kerf_wall_continuous_with_foam_land(render, layout):
    """Inner wall of edge kerf coincides with the first foam land face."""
    kx = 0.012
    hw0, pitch = layout["hw0"], layout["pitch"]
    foam = layout["foam"]
    z_ref, x_mid = layout["z_ref"], layout["x_mid"]
    # Left edge kerf c0=0: right ideal wall at +hw0 faces foam span starting at hw0.
    a0, b0 = foam[0]
    assert a0 == pytest.approx(hw0)
    xc_foam = 0.5 * (a0 + b0)
    xc_L, xc_R = render._kerf_land_centres(0.0, hw0, pitch, foam)
    assert xc_R == pytest.approx(xc_foam)

    for z in (layout["z_root"], layout["z_ref"], layout["z_mouth"]):
        kerf_wall = np.array(
            render._map_kerf_gap_point(hw0, z, -hw0, hw0, xc_L, xc_R, kx, z_ref, x_mid)
        )
        land_face = np.array(render._rect_point(a0, z, xc_foam, kx, z_ref, x_mid))
        assert kerf_wall == pytest.approx(land_face, abs=1e-9)

    # Right edge kerf c0=dx: left wall faces last foam land.
    a1, b1 = foam[-1]
    assert b1 == pytest.approx(layout["dx"] - hw0)
    xc_foam_r = 0.5 * (a1 + b1)
    xc_L, xc_R = render._kerf_land_centres(layout["dx"], hw0, pitch, foam)
    assert xc_L == pytest.approx(xc_foam_r)
    for z in (layout["z_root"], layout["z_mouth"]):
        kerf_wall = np.array(
            render._map_kerf_gap_point(
                layout["dx"] - hw0,
                z,
                layout["dx"] - hw0,
                layout["dx"] + hw0,
                xc_L,
                xc_R,
                kx,
                z_ref,
                x_mid,
            )
        )
        land_face = np.array(render._rect_point(b1, z, xc_foam_r, kx, z_ref, x_mid))
        assert kerf_wall == pytest.approx(land_face, abs=1e-9)


def test_mould_mesh_edge_resin_cells_present(render, layout):
    """Built mould mesh has resin cells at both domain-edge kerfs and mid."""
    for kx in (0.012, -0.012):
        m = render._mesh_curved_mould(kx)
        res = np.asarray(m.cell_data["resin"], dtype=bool)
        centers = m.cell_centers().points
        resin_c = centers[res]
        # At least one resin cell near each domain end and one interior.
        x = resin_c[:, 0]
        assert np.any(x < 5.0), "missing left edge kerf resin"
        assert np.any(x > 45.0), "missing right edge kerf resin"
        assert np.any((x > 15.0) & (x < 35.0)), "missing interior kerf resin"
        # Edge + interior kerfs: base lattice has 6 centres → 6 resin bands.
        assert int(res.sum()) == len(layout["c0s"])


def test_mould_mesh_edge_not_overopen_vs_mid(render, layout):
    """Mesh-level: edge resin band mouth chord ~ half mid (not over-open).

    After rigid land mapping the two mouth corners need not share a single Z,
    so the chord is the segment joining the two highest-Z unique XZ corners.
    """
    kx = 0.012
    m = render._mesh_curved_mould(kx)
    res = np.asarray(m.cell_data["resin"], dtype=bool)

    def mouth_chord(cell_id: int) -> float:
        ids = m.get_cell(cell_id).point_ids
        pts = m.points[ids]
        # One y-layer only; 4 unique XZ corners of the extruded quad.
        y0 = pts[:, 1].min()
        face = pts[np.abs(pts[:, 1] - y0) < 1e-9]
        xz = np.unique(np.round(face[:, [0, 2]], decimals=6), axis=0)
        assert len(xz) == 4, f"expected 4 XZ corners, got {len(xz)}"
        # Mouth edge = the two corners with largest Z.
        top2 = xz[np.argsort(xz[:, 1])[-2:]]
        return float(np.linalg.norm(top2[0] - top2[1]))

    centers = m.cell_centers().points
    resin_ids = np.where(res)[0]
    edge_chords = []
    mid_chords = []
    for i in resin_ids:
        cx = centers[i, 0]
        w = mouth_chord(int(i))
        if cx < 8.0 or cx > 42.0:
            edge_chords.append(w)
        elif 15.0 < cx < 35.0:
            mid_chords.append(w)
    assert edge_chords and mid_chords
    # Edge is half-width → mouth chord ~ half mid; must not exceed mid (old bug).
    assert max(edge_chords) < max(mid_chords)
    assert max(edge_chords) == pytest.approx(0.5 * max(mid_chords), rel=0.15)


def test_flat_fea_implements_hw_at(base):
    th = float(base["thickness"])
    dx = float(base["dx"])
    depth = float(base["xgr"][0][2])
    hw0 = 0.5 * float(base["xgr"][0][3])
    kx = 0.012
    mesh = create_grooved_mesh(
        thickness=th,
        dx=dx,
        dy=float(base["dy"]),
        xcuts=base["xgr"],
        ycuts=[],
        madd=[0.0],
        tface=0.0,
        kx=kx,
        ky=0.0,
    )
    grooves = _physical_grooves(base["xgr"], dx, kx)
    c0, _, d, slope = [g for g in grooves if abs(g[0] - 20) < 1][0]
    z_mouth = th if depth < 0 else 0.0
    hw_m = _hw_at(hw0, d, slope, z_mouth, th)
    band = np.abs(mesh.points[:, 2] - z_mouth) < 1e-6
    xs = mesh.points[band, 0]
    assert np.min(np.abs(xs - (c0 + hw_m))) < 0.1
