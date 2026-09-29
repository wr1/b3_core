"""CalculiX backend. Returns a full 6×6 from volume-averaged stress."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np

from b3_core.solvers.calculix.stress import (
    case_tag,
    macro_strain_from_displacement,
    stiffness_from_responses,
    volume_average_from_dat,
)
from b3_core.solvers.elasticity import (
    constituent_dict,
    face_material_dict,
    properties_from_stiffness,
)
from b3_core.solvers.protocol import Capabilities, SolveRequest, SolveResult


class CalculixBackend:
    name = "ccx"
    capabilities = Capabilities(
        orthotropic=False,
        halo=False,
        face_layer=True,
        displacements=False,
        element_types=frozenset({"C3D8", "C3D20"}),
    )

    def is_available(self) -> bool:
        return shutil.which("ccx") is not None

    def solve(self, req: SolveRequest) -> SolveResult:
        from frd2vtu import frd2vtu

        from b3_core.solvers.calculix.runner import runccx
        from b3_core.solvers.calculix.writer import vtstoccx

        cleanup = None
        if req.workdir is None:
            cleanup = tempfile.TemporaryDirectory(prefix="b3ccx_")
            work = Path(cleanup.name)
        else:
            work = Path(req.workdir)
            work.mkdir(parents=True, exist_ok=True)
        try:
            core = constituent_dict(req.core)
            resin = constituent_dict(req.resin)
            face = face_material_dict(req.face)
            inpfiles = vtstoccx(
                req.mesh,
                str(work / "rve.inp"),
                resin,
                core,
                face,
                element_type=req.element_type,
            )
            frds = runccx(list(inpfiles))
            if not frds or any(item is None for item in frds):
                raise RuntimeError(
                    "CalculiX did not produce an .frd for every load case"
                )
            frd2vtu(frds)
            strains = []
            stresses = []
            volumes = _metres(req.mesh)
            for frd in frds:
                tag = case_tag(frd)
                vtu_path = Path(frd).with_suffix(".vtu")
                dat_path = Path(frd).with_suffix(".dat")
                import pyvista as pv

                grid = pv.read(vtu_path)
                disp_key = next(
                    (k for k in grid.point_data.keys() if str(k).startswith("DISP")),
                    None,
                )
                if disp_key is None:
                    raise KeyError(f"{vtu_path} has no DISP array")
                strains.append(
                    macro_strain_from_displacement(grid, grid.point_data[disp_key])
                )
                stresses.append(volume_average_from_dat(dat_path.read_text(), volumes))
                del tag
            stiffness = stiffness_from_responses(strains, stresses)
            properties, compliance = properties_from_stiffness(stiffness)
            return SolveResult(
                stiffness=np.asarray(stiffness, dtype=float),
                properties=properties,
                compliance=np.asarray(compliance, dtype=float),
            )
        finally:
            if cleanup is not None:
                cleanup.cleanup()


def _metres(mesh) -> np.ndarray:
    """Cell volumes in m³, same order as the element ids written by ``vtstoccx``."""
    grid = mesh.scale((1e-3, 1e-3, 1e-3), inplace=False)
    if hasattr(grid, "cast_to_unstructured_grid"):
        grid = grid.cast_to_unstructured_grid()
    sized = grid.compute_cell_sizes(length=False, area=False, volume=True)
    return np.asarray(sized.cell_data["Volume"], dtype=float)
