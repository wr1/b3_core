"""Resin halo + kerf open/close composition.

Locks that curvature taper and the stochastic / geometric halo share one
``hw(z)`` law: ScoreField grades off the morphed walls, geometric ``halo``
tags sit outside those walls after morph, and open/close still ranks resin
volume (and effective resin with halo) correctly.
"""

from __future__ import annotations

import numpy as np

from b3_core.core.analysis import geom_analysis
from b3_core.core.mesh import _hw_at, create_grooved_mesh
from b3_core.core.scoring import ScoreField, effective_resin_vf
from b3_core.io import aniso

# Compact grid-scored case (matches test_halo scale) — FEA stays fast.
FOAM = {
    "E1": 32e6,
    "E2": 32e6,
    "E3": 70e6,
    "G12": 19e6,
    "G13": 19e6,
    "G23": 19e6,
    "nu12": 0.3,
    "nu13": 0.3,
    "nu23": 0.3,
    "rho": 60,
}
RESIN = {"E": 3e9, "nu": 0.3, "rho": 1100}
XG, YG = [[0, 30, 18.0, 1.0]], [[0, 30, 18.0, 1.0]]
TH, DX, DY = 20.0, 30.0, 30.0
MADD = (-0.15, 0.0, 0.15)
CS = 0.3
S_HALO = 0.5
KX_OPEN, KX_CLOSED = 0.01, -0.01


def _mesh(kx: float, *, s_halo: float = S_HALO):
    return create_grooved_mesh(
        thickness=TH,
        dx=DX,
        dy=DY,
        xcuts=XG,
        ycuts=YG,
        madd=MADD,
        tface=0.0,
        kx=kx,
        ky=0.0,
        s_halo=s_halo,
    )


def _inp(kx: float, cell_size=CS) -> dict:
    return {
        "dx": DX,
        "dy": DY,
        "thickness": TH,
        "xgr": XG,
        "ygr": YG,
        "core": {**FOAM, "cell_size": cell_size},
        "resin": RESIN,
        "curvature": {"kx": kx, "ky": 0.0},
    }


def _Ez(mesh, *, halo: bool, kx: float) -> float:
    if halo:
        sf = ScoreField(_inp(kx))
        C = aniso.runnumpy(
            mesh, RESIN, {**FOAM, "cell_size": CS}, score_field=sf
        ).stiffness
    else:
        C = aniso.runnumpy(mesh, RESIN, FOAM).stiffness
    return aniso._properties_from_stiffness(C)[0]["Ezz"]


def test_scorefield_reads_curvature_slopes():
    """ScoreField x-grooves carry ±κ taper slopes (same law as the mesh morph)."""
    sf_open = ScoreField(_inp(-0.01))  # depth>0: −κ opens the mouth
    sf_closed = ScoreField(_inp(+0.01))
    # axis 0 only — y-family has ky=0 so zero slopes
    slopes_open = sorted(
        {g[3] for g in sf_open.grooves if g[0] == 0 and abs(g[3]) > 1e-15}
    )
    slopes_closed = sorted(
        {g[3] for g in sf_closed.grooves if g[0] == 0 and abs(g[3]) > 1e-15}
    )
    assert slopes_open and slopes_closed
    assert slopes_open == sorted(-s for s in slopes_closed)
    assert all(s > 0 for s in slopes_open)  # opens mouth (depth>0)


def test_open_close_ranks_resin_vf_with_and_without_halo_mesh():
    """Morph open/close still ranks neat resin_vf when halo refinement is on."""
    # depth>0 (mouth at z=0): −kx opens, +kx closes.
    opened = geom_analysis(_mesh(-0.01, s_halo=S_HALO))["resin_vf"]
    flat = geom_analysis(_mesh(0.0, s_halo=S_HALO))["resin_vf"]
    closed = geom_analysis(_mesh(+0.01, s_halo=S_HALO))["resin_vf"]
    assert opened > flat > closed


def test_effective_resin_vf_halo_composes_with_open_close():
    """ScoreField halo_vf > 0 and effective_resin_vf ranks with open/close."""
    rows = {}
    for tag, kx in (("open", -0.01), ("flat", 0.0), ("closed", +0.01)):
        m = _mesh(kx, s_halo=S_HALO)
        g = geom_analysis(m)
        sf = ScoreField(_inp(kx))
        eff, hv = effective_resin_vf(m, sf, g["resin_vf"])
        assert hv > 0.0, tag
        assert eff > g["resin_vf"], tag
        rows[tag] = (g["resin_vf"], eff, hv)
    assert rows["open"][0] > rows["flat"][0] > rows["closed"][0]
    assert rows["open"][1] > rows["flat"][1] > rows["closed"][1]


def test_geometric_halo_tracks_tapered_walls_after_morph():
    """Post-morph halo tags sit just outside analytical hw(z) on the opened kerf."""
    # Uniaxial x-grooves only so y-halo does not pollute the x-wall gap check.
    kx = -0.01  # depth>0 → opens at mouth z=0
    m = create_grooved_mesh(
        thickness=TH,
        dx=DX,
        dy=DY,
        xcuts=XG,
        ycuts=[],
        madd=MADD,
        tface=0.0,
        kx=kx,
        ky=0.0,
        s_halo=S_HALO,
    )
    c = m.cell_centers().points
    halo = np.asarray(m.cell_data["halo"], dtype=bool)
    resin = np.asarray(m.cell_data["resin"], dtype=bool)
    assert halo.any()
    assert not np.any(halo & resin)

    pitch, depth, width = 30.0, 18.0, 1.0
    hw0 = 0.5 * width
    slope = -np.sign(depth) * kx * pitch / 2.0
    # Use an actual mesh z-plane near the open mouth
    z_levels = np.unique(np.round(c[halo, 2], 3))
    z = float(z_levels[z_levels < 4.0][0])
    hw = _hw_at(hw0, depth, slope, z, TH)
    assert hw > hw0

    band = np.abs(c[:, 2] - z) < 0.4
    h = c[band & halo]
    assert len(h) > 0
    # Edge kerfs centred near 0 and 30; gap to nearest ideal centre
    c0s = np.array([0.0, 30.0])
    gaps = []
    for p in h:
        c0 = c0s[np.argmin(np.abs(c0s - p[0]))]
        gaps.append(abs(p[0] - c0) - hw)
    gap = np.asarray(gaps, dtype=float)
    assert float(np.median(gap)) >= -0.08
    assert float(np.median(gap)) < S_HALO + 0.2
    assert float(np.percentile(gap, 90)) < S_HALO + 0.45


def test_scorefield_wall_matches_morph_hw_at_mouth():
    """Distance grows with stand-off from the open mouth wall at ScoreField hw(z)."""
    kx = -0.01
    sf = ScoreField(_inp(kx))
    # First x-groove instance (edge partial under meshadd=[0] lattice)
    axis, c0, hw0, slope, depth = next(g for g in sf.grooves if g[0] == 0)
    z = 0.5
    hw = _hw_at(hw0, depth, slope, z, TH)
    assert hw > hw0  # opened at mouth
    pts = np.array(
        [
            [c0 + hw + 0.02, 15.0, z],
            [c0 + hw + CS * 0.5, 15.0, z],
            [c0 + hw + CS * 1.5, 15.0, z],
        ]
    )
    d = sf.distance_to_saw_cut(pts)
    assert d[0] < d[1] < d[2]
    assert d[0] < 0.1


def test_halo_boosts_stiffness_for_open_flat_closed():
    """Halo raises Ezz for open, flat, and closed morphs."""
    for kx in (-0.01, 0.0, +0.01):
        m = _mesh(kx, s_halo=S_HALO)
        e_h = _Ez(m, halo=True, kx=kx)
        e_s = _Ez(m, halo=False, kx=kx)
        assert e_h > e_s, kx


def test_open_stiffer_than_closed_with_halo():
    """With halo, opened morph stays stiffer than closed (composition ranks)."""
    m_open = _mesh(-0.01, s_halo=S_HALO)
    m_closed = _mesh(+0.01, s_halo=S_HALO)
    e_open = _Ez(m_open, halo=True, kx=-0.01)
    e_closed = _Ez(m_closed, halo=True, kx=+0.01)
    assert e_open > e_closed
