import numpy as np
import pytest
import pyvista as pv
import vtk

from b3_core.io.vts2ccx import vtstoccx


@pytest.fixture
def simple_mesh():
    points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=float,
    )
    cells = np.array([8, 0, 1, 2, 3, 4, 5, 6, 7])
    cell_types = np.array([vtk.VTK_HEXAHEDRON])
    grid = pv.UnstructuredGrid(cells, cell_types, points)
    grid.cell_data["resin"] = np.array([0])
    grid.cell_data["face"] = np.array([0])
    return grid


def test_vtstoccx(simple_mesh, tmp_path):
    output = tmp_path / "test.inp"
    resin = {"E": 4e9, "nu": 0.3}
    core = {"E": 4e9, "nu": 0.3}
    files = vtstoccx(simple_mesh, str(output), resin, core)
    assert len(files) == 6
    for f in files:
        assert f.endswith(".inp")
        content = open(f).read()
        assert "*material,name=face" in content
        assert "12000000000,0.3" in content
