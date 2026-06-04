"""Curvature-driven groove taper for curved-panel RVEs.

A mold curvature opens grooves mouthing on the convex face and closes those on
the concave face. The taper hinges at the groove root corner; ``kx`` acts on the
x-groove family, ``ky`` on the y-groove family. ``kappa == 0`` must reproduce the
flat mesh exactly.
"""

import numpy as np
import pytest

from b3_core.core.analysis import geom_analysis
from b3_core.core.cprop import CpropInput
from b3_core.core.mesh import create_grooved_mesh

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
    """Beyond kappa_close = w/(p*depth) the family contributes almost no resin."""
    w, p, d = 3.0, 10.0, 8.0
    kappa_close = w / (p * d)  # closing curvature for a bottom-mouth groove (+kx closes)
    common = dict(
        thickness=30.0, dx=50.0, dy=50.0, xcuts=[[10, p, d, w]], ycuts=[], madd=MADD, tface=0.0
    )
    flat = _resin_vf(**common, kx=0.0)
    pinched = _resin_vf(**common, kx=2.0 * kappa_close)
    assert pinched < 0.15 * flat


def test_mouth_flares_wider_than_root_when_opened():
    """For an opened groove the marked resin is wider near the mouth than the root."""
    mesh = create_grooved_mesh(
        thickness=30.0, dx=50.0, dy=50.0,
        xcuts=[[25, 60, -10, 3]], ycuts=[],  # single top-mouth groove (pitch>dx => one slot)
        madd=MADD, tface=0.0, kx=0.01,
    )
    pts = _resin_centres(mesh)
    near_mouth = pts[pts[:, 2] > 28.0]   # mouth at z = thickness = 30
    near_root = pts[pts[:, 2] < 22.0]    # root at z = 30 - 10 = 20
    assert len(near_mouth) and len(near_root)
    spread_mouth = near_mouth[:, 0].max() - near_mouth[:, 0].min()
    spread_root = near_root[:, 0].max() - near_root[:, 0].min()
    assert spread_mouth > spread_root


def test_two_sided_grades_asymmetrically():
    """Opposite-face families: +ky and -ky shift the resin centroid opposite ways."""
    common = dict(
        thickness=30.0, dx=50.0, dy=50.0, xcuts=[],
        ycuts=[[-2, 25, 17, 2], [2, 25, -17, 2]], madd=MADD, tface=0.0,
    )
    z_flat = _resin_centres(create_grooved_mesh(**common)).mean(axis=0)[2]
    z_plus = _resin_centres(create_grooved_mesh(**common, ky=0.004)).mean(axis=0)[2]
    z_minus = _resin_centres(create_grooved_mesh(**common, ky=-0.004)).mean(axis=0)[2]
    # one curvature pushes the resin centroid up, the other down
    assert (z_plus - z_flat) * (z_minus - z_flat) < 0


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
