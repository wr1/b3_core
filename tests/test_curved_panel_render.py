"""Curved-panel render kinematics: FEA mesh rolled onto a cylinder.

Locks the behaviour that fixed needle-sharp curved kerfs: the curved view is
the *same* interval-morphed FEA mesh as the flat panel, only point coordinates
are rolled onto an arc. Resin tags, cell count, and material-space kerf
half-widths must match; root tips stay blunt like the flat FEA.
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


def _resin_halfwidths_flat(mesh, c0: float, pitch: float, zs) -> list[float]:
    """Half-width of resin *geometry* about ``c0`` at each z (material x)."""
    res_ids = np.where(np.asarray(mesh.cell_data["resin"]).astype(bool))[0]
    ug = mesh.extract_cells(res_ids)
    pts = ug.points
    half_pitch = 0.5 * pitch
    hws = []
    for z in zs:
        band = np.abs(pts[:, 2] - float(z)) < 0.2
        p = pts[band & (np.abs(pts[:, 0] - c0) < half_pitch)]
        if len(p) < 2:
            hws.append(0.0)
        else:
            hws.append(0.5 * float(p[:, 0].max() - p[:, 0].min()))
    return hws


def test_roll_zero_kappa_is_identity(render):
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(30, 3))
    out = render._roll_points_to_cylinder(pts, 0.0, x_mid=25.0, z_ref=15.0)
    assert np.array_equal(out, pts)


def test_roll_curves_and_preserves_count(render):
    x = np.linspace(0.0, 50.0, 21)
    pts = np.column_stack([x, np.zeros_like(x), np.full_like(x, 15.0)])
    bent = render._roll_points_to_cylinder(pts, 0.012, x_mid=25.0, z_ref=15.0)
    assert bent.shape == pts.shape
    assert not np.allclose(bent, pts)
    # mid-plane arc: ends pull inward in x, mid stays put
    assert bent[10, 0] == pytest.approx(25.0, abs=1e-9)
    assert bent[0, 0] > pts[0, 0]  # left end moves toward centre on smile arc
    assert bent[-1, 0] < pts[-1, 0]


def test_curved_and_flat_share_topology_and_resin_tags(render):
    """Curved is a pure point roll of the FEA mesh — no re-meshing / re-tagging."""
    for kx in (0.012, -0.012, 0.0):
        flat = render._mesh_flat_fea(kx)
        curved = render._mesh_curved_fea(kx)
        assert curved.n_cells == flat.n_cells
        assert curved.n_points == flat.n_points
        assert np.array_equal(
            np.asarray(curved.cell_data["resin"]),
            np.asarray(flat.cell_data["resin"]),
        )
        if abs(kx) < 1e-12:
            assert np.allclose(curved.points, flat.points)
        else:
            assert not np.allclose(curved.points, flat.points)


def test_open_fea_root_stays_blunt_like_analytical(base):
    """Open morph: root half-width stays ~hw0 (meshadd ≥ ideal), not needle-sharp."""
    th = float(base["thickness"])
    dx = float(base["dx"])
    offset, pitch, depth, width = map(float, base["xgr"][0])
    kx = 0.012
    hw0 = 0.5 * width
    mesh = create_grooved_mesh(
        thickness=th,
        dx=dx,
        dy=float(base["dy"]),
        xcuts=base["xgr"],
        ycuts=base["ygr"],
        madd=tuple(base["madd"]),
        tface=0.0,
        kx=kx,
        ky=0.0,
    )
    grooves = _physical_grooves(base["xgr"], dx, kx)
    interior = [g for g in grooves if 1.0 < g[0] < dx - 1.0]
    assert interior
    c0, _, d, slope = interior[0]
    z_root = th + depth  # depth < 0
    z_mouth = th
    hw_root_an = _hw_at(hw0, d, slope, z_root, th)
    hw_mouth_an = _hw_at(hw0, d, slope, z_mouth, th)
    assert hw_mouth_an > hw_root_an
    # Analytical root is nominal hw0; must not be the ε clamp
    assert hw_root_an == pytest.approx(hw0, abs=1e-9)

    hws = _resin_halfwidths_flat(mesh, c0, pitch, [z_root, z_mouth])
    hw_root, hw_mouth = hws
    assert hw_root > 0.8 * hw0  # blunt tip (mesh geometry ≥ ideal)
    assert hw_mouth > hw_root


def test_closed_fea_mouth_pinches_but_stays_finite(base):
    """Closed morph: mouth narrower than root but not zero (ε / mesh floor)."""
    th = float(base["thickness"])
    dx = float(base["dx"])
    offset, pitch, depth, width = map(float, base["xgr"][0])
    kx = -0.012
    hw0 = 0.5 * width
    mesh = create_grooved_mesh(
        thickness=th,
        dx=dx,
        dy=float(base["dy"]),
        xcuts=base["xgr"],
        ycuts=base["ygr"],
        madd=tuple(base["madd"]),
        tface=0.0,
        kx=kx,
        ky=0.0,
    )
    grooves = _physical_grooves(base["xgr"], dx, kx)
    interior = [g for g in grooves if 1.0 < g[0] < dx - 1.0]
    c0, _, d, slope = interior[0]
    z_root = th + depth
    z_mouth = th
    hws = _resin_halfwidths_flat(mesh, c0, pitch, [z_root, z_mouth])
    hw_root, hw_mouth = hws
    assert hw_mouth < hw_root
    assert hw_mouth > 0.15  # not a needle; FEA envelope stays finite
    assert hw_root > 0.8 * hw0


def test_curved_roll_preserves_material_x_differences(render, base):
    """Rolling maps Δx → Δθ; inverse recovers material x, so tip widths match FEA."""
    kx = 0.012
    th = float(base["thickness"])
    dx = float(base["dx"])
    x_mid = 0.5 * dx
    z_ref = 0.5 * th
    R = 1.0 / abs(kx)

    flat = render._mesh_flat_fea(kx)
    curved = render._mesh_curved_fea(kx)

    # Invert roll: θ = atan2(X - x_mid relative to centre in polar about hinge)
    # Built so θ = κ(x - x_mid), r = R + (z - z_ref)
    # X = x_mid + r sin θ, Z = z_ref - R + r cos θ
    # → θ = atan2(X - x_mid, Z - (z_ref - R)), r = hypot(...), x = x_mid + θ/κ
    X = curved.points[:, 0]
    Z = curved.points[:, 2]
    dx_ = X - x_mid
    dz_ = Z - (z_ref - R)
    theta = np.arctan2(dx_, dz_)
    x_back = x_mid + theta / kx
    z_back = np.hypot(dx_, dz_) - R + z_ref
    assert np.allclose(x_back, flat.points[:, 0], atol=1e-5)
    assert np.allclose(z_back, flat.points[:, 2], atol=1e-5)
    assert np.allclose(curved.points[:, 1], flat.points[:, 1], atol=1e-12)


def test_render_helpers_use_base_case_geometry(render, base):
    """Example BASE drives both panels; open and closed meshes are non-empty."""
    assert float(base["thickness"]) == 30.0
    for kx in (render.PAIR_KX_OPEN, render.PAIR_KX_CLOSED):
        m = render._mesh_flat_fea(kx)
        assert m.n_cells > 0
        assert "resin" in m.cell_data
        assert np.any(np.asarray(m.cell_data["resin"]).astype(bool))
        c = render._mesh_curved_fea(kx)
        assert c.n_cells == m.n_cells
