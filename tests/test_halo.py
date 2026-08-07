import numpy as np
import pytest

from b3_core.core.mesh import create_grooved_mesh
from b3_core.core.scoring import ScoreField
from b3_core.io import aniso

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


def _inp(cell_size):
    return {
        "dx": 30,
        "dy": 30,
        "thickness": 20,
        "xgr": XG,
        "ygr": YG,
        "core": {**FOAM, "cell_size": cell_size},
    }


def _mesh(s_halo):
    return create_grooved_mesh(
        thickness=20,
        dx=30,
        dy=30,
        xcuts=XG,
        ycuts=YG,
        madd=(-0.15, 0, 0.15),
        tface=0.0,
        s_halo=s_halo,
    )


# One mesh, band resolved to 0.5 mm, reused so field comparisons are like-for-like
# (avoids confounding the halo effect with mesh-convergence between meshes).
_M = _mesh(0.5)


def _Ez(cell_size=None, strategy="exact"):
    if cell_size is None:
        C = aniso.runnumpy(_M, RESIN, FOAM).stiffness  # no field
    else:
        inp = _inp(cell_size)
        C = aniso.runnumpy(
            _M,
            RESIN,
            inp["core"],
            score_field=ScoreField(inp),
            scoring={"sampling": {"strategy": strategy}},
        ).stiffness
    return aniso._properties_from_stiffness(C)[0]["Ezz"]


# -- correctness (all on the same mesh) ------------------------------------
def test_no_field_matches_score_field_none():
    a = aniso.runnumpy(_M, RESIN, FOAM).stiffness
    b = aniso.runnumpy(_M, RESIN, FOAM, score_field=None).stiffness
    assert np.allclose(a, b)


def test_field_stiffer_than_sharp():
    assert _Ez(0.3) > _Ez(None)


def test_stiffness_monotone_in_cell_size():
    assert _Ez(0.2) < _Ez(0.3) < _Ez(0.4)  # bigger cells -> wider halo -> stiffer


def test_exact_and_local_cloud_both_work():
    sharp = _Ez(None)
    e, lc = _Ez(0.3, "exact"), _Ez(0.3, "local_cloud")
    assert e > sharp and lc > sharp
    assert lc == pytest.approx(e, rel=0.1)


def test_distribution_cell_size_stiffer():
    assert _Ez({"mean": 0.25, "std": 0.08, "dist": "lognormal"}) > _Ez(None)


# -- pipeline ---------------------------------------------------------------
def test_pipeline_routes_and_reports_vf():
    from b3_core.viz.model import CoreModel

    m = CoreModel.from_json("examples/diab_gs30_scored.json")
    g = m.geom
    assert "effective_resin_vf" in g and "halo_vf" in g
    assert g["effective_resin_vf"] > g["resin_vf"]  # halo adds resin
    assert type(m.details).__name__ == "AnisoResult"  # numpy backend
