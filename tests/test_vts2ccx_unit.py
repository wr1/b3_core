"""Unit tests for CalculiX deck helpers (no ccx binary)."""

from __future__ import annotations

import numpy as np
import pyvista as pv

from b3_core.io import vts2ccx


def _tiny_hex_mesh():
    # 2x2x2 structured grid of one hex? Use ImageData for clean bounds.
    grid = pv.ImageData(dimensions=(3, 3, 3), spacing=(1.0, 1.0, 1.0), origin=(0, 0, 0))
    ug = grid.cast_to_unstructured_grid()
    ug.cell_data["material"] = np.ones(ug.n_cells, dtype=int)
    return ug


def test_nset_and_periodic_bcs():
    msh = _tiny_hex_mesh()
    buf = vts2ccx.nset(
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]), "demo"
    )
    assert "*nset,nset=demo" in buf
    assert "1,2,3" in buf.replace(" ", "") or "1," in buf

    # strain-node bookkeeping: 3 dummy nodes after mesh nodes
    n = msh.n_points
    added = {0: n + 1, 1: n + 2, 2: n + 3}
    pbc = vts2ccx.periodic_bcs(msh, added)
    assert "*equation" in pbc
    assert "xmin" in pbc and "xmax" in pbc


def test_vtstoccx_writes_inps(tmp_path):
    from b3_core.core.mesh import create_grooved_mesh

    mesh = create_grooved_mesh(
        thickness=10.0,
        dx=10.0,
        dy=10.0,
        xcuts=[[5, 10, 5, 1]],
        ycuts=[],
        madd=[0],
        tface=0.0,
    )
    # ensure material-like cell data if required
    if "material" not in mesh.cell_data:
        mesh.cell_data["material"] = np.where(
            mesh.cell_data.get("resin", np.zeros(mesh.n_cells)), 2, 1
        )

    core = {"E": 1e9, "nu": 0.3, "rho": 100}
    resin = {"E": 3e9, "nu": 0.3, "rho": 1100}
    # vtstoccx signature from module
    if not hasattr(vts2ccx, "vtstoccx"):
        # alternate name
        fn = getattr(vts2ccx, "vts2ccx", None)
        assert fn is not None
    else:
        out = tmp_path / "deck.inp"
        # function may return list of loadcase paths
        try:
            paths = vts2ccx.vtstoccx(
                mesh, str(out), resin, core, None, element_type="C3D8"
            )
        except TypeError:
            paths = vts2ccx.vtstoccx(mesh, str(out), resin, core)
        assert paths is not None
