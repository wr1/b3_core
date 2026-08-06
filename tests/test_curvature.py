"""Curvature-driven kerf open/close via wall-aligned mesh morph.

A mold curvature opens grooves mouthing on the convex face and closes those on
the concave face. The taper hinges at the groove root corner; wall *faces* move
with ``hw(z)`` (interval-affine morph). ``kappa == 0`` must reproduce the flat
mesh exactly. Resin tags stay rectangular; volumes change with the morph.
"""

import numpy as np
import pytest

from b3_core.core.analysis import geom_analysis
from b3_core.core.cprop import CpropInput
from b3_core.core.mesh import (
    _hw_at,
    _physical_grooves,
    create_grooved_mesh,
)

MADD = [-0.4, -0.2, 0, 0.2, 0.4]


def _resin_centres(mesh):
    centres = mesh.cell_centers().points
    return centres[mesh.cell_data["resin"].astype(bool)]


def _resin_vf(**kw):
    return geom_analysis(create_grooved_mesh(**kw))["resin_vf"]


def test_zero_curvature_is_identical_to_flat():
    """kx=ky=0 must give bit-for-bit the original (untapered) resin marking."""
    common = dict(
        thickness=30.0,
        dx=50.0,
        dy=50.0,
        xcuts=[[10, 10, 8, 3]],
        ycuts=[[-2, 25, 17, 2], [2, 25, -17, 2]],
        madd=MADD,
        tface=0.0,
    )
    flat = create_grooved_mesh(**common)
    explicit_zero = create_grooved_mesh(**common, kx=0.0, ky=0.0)
    assert np.array_equal(
        flat.cell_data["resin"], explicit_zero.cell_data["resin"]
    )
    assert np.allclose(flat.points, explicit_zero.points)


def test_opening_raises_closing_lowers_resin_vf():
    """A top-mouth groove (depth<0) opens under +kx and closes under -kx."""
    common = dict(
        thickness=30.0, dx=50.0, dy=50.0, xcuts=[[10, 10, -8, 3]], ycuts=[], madd=MADD, tface=0.0
    )
    opened = _resin_vf(**common, kx=0.008)
    flat = _resin_vf(**common, kx=0.0)
    closed = _resin_vf(**common, kx=-0.008)
    assert opened > flat > closed


def test_full_closure_pinches_groove_shut():
    """Beyond kappa_close the mouth is at min width; resin volume drops vs flat."""
    w, p, d = 3.0, 10.0, 8.0
    kappa_close = w / (p * d)  # +kx closes a bottom-mouth groove
    common = dict(
        thickness=30.0, dx=50.0, dy=50.0, xcuts=[[10, p, d, w]], ycuts=[], madd=MADD, tface=0.0
    )
    flat = _resin_vf(**common, kx=0.0)
    pinched = _resin_vf(**common, kx=2.0 * kappa_close)
    assert pinched < 0.70 * flat
    assert pinched > 0.0


def test_wall_nodes_track_analytical_taper():
    """After morph, wall grid lines lie on c0 ± hw(z) at each mesh z-plane."""
    th, c0, pitch, depth, width = 30.0, 25.0, 60.0, -10.0, 3.0
    kx = 0.01
    hw0 = 0.5 * width
    mesh = create_grooved_mesh(
        thickness=th, dx=50.0, dy=50.0,
        xcuts=[[c0, pitch, depth, width]], ycuts=[],
        madd=[0.0], tface=0.0, kx=kx,
    )
    grooves = _physical_grooves([[c0, pitch, depth, width]], 50.0, kx)
    assert len(grooves) == 1
    _, _, d, slope = grooves[0]
    z_root = th + depth
    z_planes = np.unique(np.round(mesh.points[:, 2], 8))
    z_planes = z_planes[(z_planes >= z_root - 1e-6) & (z_planes <= th + 1e-6)]
    assert len(z_planes) >= 3
    for z in z_planes:
        hw = _hw_at(hw0, depth, slope, float(z), th)
        band = np.abs(mesh.points[:, 2] - z) < 1e-7
        xs = mesh.points[band, 0]
        for sign in (-1.0, 1.0):
            x_wall = c0 + sign * hw
            assert np.min(np.abs(xs - x_wall)) < 0.05, (z, x_wall, xs)


def test_mouth_wider_than_root_on_wall_line():
    """Opened top-mouth groove: wall half-width grows from root to mouth."""
    th, c0, depth, width, kx = 30.0, 25.0, -10.0, 3.0, 0.01
    hw0 = 0.5 * width
    slope = -np.sign(depth) * kx * 60.0 / 2.0  # pitch 60 single instance
    z_root, z_mouth = th + depth, th
    hw_root = _hw_at(hw0, depth, slope, z_root, th)
    hw_mouth = _hw_at(hw0, depth, slope, z_mouth, th)
    assert hw_mouth > hw_root

    mesh = create_grooved_mesh(
        thickness=th, dx=50.0, dy=50.0,
        xcuts=[[c0, 60.0, depth, width]], ycuts=[],
        madd=MADD, tface=0.0, kx=kx,
    )
    # Resin cell-centre span at mouth vs root (should track morphed walls)
    pts = _resin_centres(mesh)
    pts = pts[(pts[:, 0] > 15.0) & (pts[:, 0] < 35.0)]
    near_mouth = pts[pts[:, 2] > 28.0]
    near_root = pts[(pts[:, 2] > 20.0) & (pts[:, 2] < 22.0)]
    assert len(near_mouth) and len(near_root)
    spread_mouth = near_mouth[:, 0].max() - near_mouth[:, 0].min()
    spread_root = near_root[:, 0].max() - near_root[:, 0].min()
    assert spread_mouth > spread_root


def test_taper_has_multiple_z_stations_not_prismatic():
    """Opened kerfs resolve hw(z) over several z layers; width grows with z."""
    mesh = create_grooved_mesh(
        thickness=30.0, dx=50.0, dy=50.0,
        xcuts=[[10, 10, -27, 3]], ycuts=[],
        madd=MADD, tface=0.0, kx=0.012,
    )
    pts = _resin_centres(mesh)
    pts = pts[(pts[:, 0] > 6.0) & (pts[:, 0] < 14.0)]
    assert len(pts)
    z_levels = np.unique(np.round(pts[:, 2], 3))
    assert len(z_levels) >= 5

    def width_at(z, tol=0.4):
        band = pts[np.abs(pts[:, 2] - z) < tol]
        if len(band) == 0:
            return None
        return float(band[:, 0].max() - band[:, 0].min())

    z_root, z_mid, z_mouth = z_levels[0], z_levels[len(z_levels) // 2], z_levels[-1]
    w_root, w_mid, w_mouth = width_at(z_root), width_at(z_mid), width_at(z_mouth)
    assert w_root is not None and w_mid is not None and w_mouth is not None
    assert w_mouth > w_mid > w_root


def test_positive_cell_volumes_after_morph():
    """Morphed open and near-closed meshes keep positive cell volumes."""
    common = dict(
        thickness=30.0, dx=50.0, dy=50.0,
        xcuts=[[10, 10, -27, 3]], ycuts=[], madd=MADD, tface=0.0,
    )
    for kx in (0.012, -0.012, 0.0):
        mesh = create_grooved_mesh(**common, kx=kx)
        vol = np.abs(mesh.compute_cell_sizes().cell_data["Volume"])
        assert np.all(vol > 0.0), kx
        assert float(vol.min()) > 1e-9


def test_two_sided_opposite_ky_swaps_band_volumes():
    """Opposite-face y-grooves: +ky vs -ky exchange lower/upper resin volume share.

    Tags stay rectangular (same cells); morph changes volumes so one face's
    kerf grows while the other pinches.
    """
    common = dict(
        thickness=30.0, dx=50.0, dy=50.0, xcuts=[],
        ycuts=[[-2, 25, 17, 2], [2, 25, -17, 2]], madd=MADD, tface=0.0,
    )

    def band_share(ky: float) -> tuple[float, float]:
        m = create_grooved_mesh(**common, ky=ky)
        c = m.cell_centers().points
        vol = np.abs(m.compute_cell_sizes().cell_data["Volume"])
        res = m.cell_data["resin"].astype(bool)
        lo = res & (c[:, 2] < 15.0)
        hi = res & (c[:, 2] >= 15.0)
        v_lo, v_hi = float(vol[lo].sum()), float(vol[hi].sum())
        tot = v_lo + v_hi + 1e-30
        return v_lo / tot, v_hi / tot

    lo_p, hi_p = band_share(0.004)
    lo_m, hi_m = band_share(-0.004)
    # +ky and -ky should reverse which half of the thickness holds more resin volume
    assert (lo_p - hi_p) * (lo_m - hi_m) < 0


def test_cprop_input_accepts_and_validates_curvature():
    base = dict(
        dx=50.0, dy=50.0, thickness=30.0, xgr=[[10, 10, 8, 3]], ygr=[],
        core={"E": 4e9, "nu": 0.3, "rho": 100.0},
        resin={"E": 4e9, "nu": 0.3, "rho": 1100.0},
    )
    cfg = CpropInput(**base, curvature={"kx": 0.004, "ky": 0.0})
    assert cfg.curvature == {"kx": 0.004, "ky": 0.0}

    with pytest.raises(ValueError):
        CpropInput(**base, curvature={"kz": 1.0})
    with pytest.raises(ValueError):
        CpropInput(**base, curvature={"kx": "nope"})
