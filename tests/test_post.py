import numpy as np
import pytest
import pyvista as pv
import vtk

from b3_core.post.planar import postprocess_planar
from b3_core.post.skins import postprocess


@pytest.fixture
def mock_vtu(tmp_path):
    points = np.array(
        [
            [0, 0, 0],
            [0.05, 0, 0],
            [0.05, 0.05, 0],
            [0, 0.05, 0],
            [0, 0, 0.03],
            [0.05, 0, 0.03],
            [0.05, 0.05, 0.03],
            [0, 0.05, 0.03],
        ],
        dtype=float,
    )
    cells = np.array([8, 0, 1, 2, 3, 4, 5, 6, 7])
    cell_types = np.array([vtk.VTK_HEXAHEDRON])
    grid = pv.UnstructuredGrid(cells, cell_types, points)
    disp = np.zeros((8, 3))
    disp[:, 1] = points[:, 0] * 0.01  # Non-zero y-displacement varying with x
    grid.point_data["DISP_1.000"] = disp
    force = np.zeros((8, 3))
    ymin_indices = [0, 1, 4, 5]  # y=0 points
    force[ymin_indices, 0] = 1.0
    grid.point_data["FORC_1.000"] = force
    grid.cell_data["material"] = np.array([1])
    grid.point_data["mises_stress"] = np.ones(8)
    grid.point_data["mises_strain"] = np.ones(8) * 0.01
    vtu_file = tmp_path / "test_xy.vtu"
    grid.save(vtu_file)
    dat_file = tmp_path / "test_xy.dat"
    with open(dat_file, "w") as f:
        f.write("0.0 0.0 0.0\n0.0 0.0 0.0\n0.0 0.0 0.0\n")
    return str(vtu_file), str(dat_file)


def test_postprocess_skins(mock_vtu):
    vtu_file, dat_file = mock_vtu
    result = postprocess([pv.read(vtu_file)], [dat_file], thickness=30.0)
    assert "Gxy" in result
    assert isinstance(result["Gxy"], float)


def test_postprocess_planar(mock_vtu):
    vtu_file, _ = mock_vtu
    result = postprocess_planar([vtu_file])
    assert "Gxy" in result
    assert isinstance(result["Gxy"], float)
