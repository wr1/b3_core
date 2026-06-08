import json

import numpy as np
import pytest

from b3_core.core.mesh import create_grooved_mesh
from b3_core.io import aniso

GS30 = json.loads(open("examples/diab_gs30.json").read())
XG, YG = [[0, 30, 18.0, 1.0]], [[0, 30, 18.0, 1.0]]


def _mesh(s_halo):
    return create_grooved_mesh(
        thickness=20, dx=30, dy=30, xcuts=XG, ycuts=YG,
        madd=(-0.15, 0, 0.15), tface=0.0, s_halo=s_halo,
    )


def _Ez(s_halo):
    m = _mesh(s_halo)
    C = aniso.runnumpy(m, GS30["resin"], GS30["core"],
                       scoring={"solid_density": 1400}).stiffness
    return aniso._properties_from_stiffness(C)[0]["Ezz"]


def test_halo_off_is_unchanged():
    m0, m1 = _mesh(0.0), _mesh(0.0)
    assert m0.n_cells == m1.n_cells
    assert m0.cell_data["halo"].sum() == 0
    # identical to the sharp-kerf result
    C_off = aniso.runnumpy(m0, GS30["resin"], GS30["core"]).stiffness
    C_explicit = aniso.runnumpy(_mesh(0.0), GS30["resin"], GS30["core"],
                                scoring=None).stiffness
    assert np.allclose(C_off, C_explicit)


def test_halo_marking_graded_and_geometry_preserved():
    off, on = _mesh(0.0), _mesh(0.6)
    # resin geometry untouched by the halo sub-lines
    from b3_core.core.analysis import geom_analysis
    assert geom_analysis(off)["resin_vf"] == pytest.approx(
        geom_analysis(on)["resin_vf"], rel=1e-9
    )
    hf = np.asarray(on.cell_data["halo_fraction"])
    assert (hf > 0).any()
    assert hf.max() <= 1.0 and hf.min() == 0.0
    g = geom_analysis(on)
    assert g["halo_vf"] > 0 and 0 < g["halo_resin_equiv"] < g["halo_vf"]


def test_stiffness_monotone_in_cell_size():
    # bigger cell_size -> wider halo -> more effective resin -> stiffer
    e0, e3, e6 = _Ez(0.0), _Ez(0.3), _Ez(0.6)
    assert e0 < e3 < e6


def test_halo_bounded_by_sharp_and_full_resin():
    # the halo result sits between the sharp kerf and an all-resin halo
    sharp = _Ez(0.0)
    m = _mesh(0.6)
    graded = aniso._properties_from_stiffness(
        aniso.runnumpy(m, GS30["resin"], GS30["core"],
                       scoring={"solid_density": 1400}).stiffness
    )[0]["Ezz"]
    # force phi=1 everywhere in the band by a near-solid foam (porosity~1)
    full = aniso._properties_from_stiffness(
        aniso.runnumpy(m, GS30["resin"], GS30["core"],
                       scoring={"solid_density": 1e9}).stiffness
    )[0]["Ezz"]
    assert sharp < graded <= full + 1.0


def test_porosity():
    assert aniso.foam_porosity({"rho": 60}, {"solid_density": 1400}) == pytest.approx(
        1 - 60 / 1400
    )
    assert aniso.foam_porosity({"rho": 60}, None) == pytest.approx(1 - 60 / 1400)


def test_pipeline_routes_halo_to_numpy_and_reports_vf():
    from b3_core.viz.model import CoreModel

    m = CoreModel.from_json("examples/diab_gs30_scored.json")
    g = m.geom
    assert "halo_vf" in g and "effective_resin_vf" in g
    assert g["effective_resin_vf"] > g["resin_vf"]   # halo adds resin
    assert type(m.details).__name__ == "AnisoResult"  # routed to numpy
