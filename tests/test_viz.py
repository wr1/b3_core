import matplotlib
import numpy as np
import pytest

from b3_core.io import mfem_backend
from b3_core.viz import (
    CoreModel,
    CoreScene,
    GroovedCoreView,
    slices,
    tensor,
    tensorplot,
)

matplotlib.use("Agg")

MESH_CASE = "examples/with_grooves.json"  # small mesh, no MFEM needed for geometry
needs_mfem = pytest.mark.skipif(
    not mfem_backend.is_mfem_available(), reason="MFEM not installed"
)


def iso_C(E: float, nu: float) -> np.ndarray:
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    C = np.full((6, 6), 0.0)
    for i in range(3):
        for j in range(3):
            C[i, j] = lam
        C[i, i] += 2 * mu
    for k in (3, 4, 5):
        C[k, k] = mu
    return C


# -- tensor maths (pure, no FEA) -------------------------------------------
def test_isotropic_modulus_is_direction_independent():
    C = iso_C(4e9, 0.3)
    dirs = np.random.default_rng(0).normal(size=(40, 3))
    assert np.allclose(tensor.youngs_modulus(C, dirs), 4e9, rtol=1e-6)


def test_orthotropic_axis_modulus_matches_engineering_constants():
    C = iso_C(4e9, 0.3)
    C[2, 2] *= 2.0
    ec = tensor.engineering_constants(C)
    e_z = float(tensor.youngs_modulus(C, np.array([0.0, 0.0, 1.0])))
    assert e_z == pytest.approx(ec["E_z"], rel=1e-9)
    assert e_z > ec["E_x"]  # stiffened along z


def test_modulus_surface_and_polar():
    C = iso_C(4e9, 0.3)
    C[2, 2] *= 2.0
    surf = tensor.modulus_surface(C, resolution=30)
    assert surf.n_points > 100
    assert "value_GPa" in surf.point_data
    theta, E = tensor.polar_modulus(C, "xz", n=181)
    assert theta.shape == (181,) and E.shape == (181,)
    assert E.max() > E.min()  # anisotropic in the xz plane


def test_tensor_figures():
    C = iso_C(4e9, 0.3)
    C[2, 2] *= 2.0
    ax = tensorplot.plot_stiffness_heatmap(C)
    assert ax is not None
    axes = tensorplot.plot_modulus_polar(C)
    assert len(axes) == 3


# -- model + 2D geometry (mesh only, no FEA) -------------------------------
def test_coremodel_mesh_is_cached():
    m = CoreModel.from_json(MESH_CASE)
    assert m.mesh is m.mesh
    assert m.material_codes.shape[0] == m.mesh.n_cells
    assert set(np.unique(m.material_codes)).issubset({0, 1, 2})


def test_orthogonal_cuts_figure():
    m = CoreModel.from_json(MESH_CASE)
    fig, aspect = slices.plot_orthogonal_cuts(m)
    assert aspect > 0
    assert fig.axes


def test_scene_geometry_screenshot(tmp_path):
    m = CoreModel.from_json(MESH_CASE)
    out = (
        CoreScene(m).add_phases(edges=True).add_mesh_edges().add_axes()
        .isometric().screenshot(tmp_path / "geom.png")
    )
    assert out.stat().st_size > 5000


# -- full homogenisation views (need MFEM) ---------------------------------
@needs_mfem
def test_model_stiffness_and_modulus_scene(tmp_path):
    m = CoreModel.from_json(MESH_CASE)
    C = m.stiffness
    assert C.shape == (6, 6)
    assert np.allclose(C, C.T, atol=1e-3)
    out = CoreScene(m).add_modulus_surface().isometric().screenshot(tmp_path / "mod.png")
    assert out.stat().st_size > 5000


@needs_mfem
def test_view_gallery(tmp_path):
    out = GroovedCoreView.from_json(MESH_CASE).gallery(tmp_path / "board.png")
    assert out.is_file() and out.stat().st_size > 10_000
