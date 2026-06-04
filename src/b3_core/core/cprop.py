#!/usr/bin/env python3

import json
import hashlib
import os
from .mesh import create_grooved_mesh
from .analysis import geom_analysis
from ..io.vts2ccx import vtstoccx
from ..io.runccx import runccx
from ..io.fenicsx import runfenicsx, validate_against_ccx
from ..io.mfem_backend import runmfem
from frd2vtu import frd2vtu
import pyvista as pv
from ..post.skins import postprocess
from ..result import CoreResult
from pydantic import BaseModel, Field, validator
from rich.console import Console


class Material(BaseModel):
    E: float = Field(..., gt=0)
    nu: float = Field(..., ge=0, lt=0.5)
    rho: float = Field(..., gt=0)


class CpropInput(BaseModel):
    dx: float = Field(..., gt=0)
    dy: float = Field(..., gt=0)
    thickness: float = Field(..., gt=0)
    xgr: list[list[float]]
    ygr: list[list[float]]
    core: Material
    resin: Material
    madd: list[float] = [0]
    face: dict = {}
    curvature: dict = {}
    element_type: str = "C3D8"
    backend: str = "ccx"
    validate_with_ccx: bool = False

    @validator("element_type")
    def validate_element_type(cls, v):
        if v not in ("C3D8", "C3D20"):
            raise ValueError(f"element_type must be 'C3D8' or 'C3D20', got {v!r}")
        return v

    @validator("backend")
    def validate_backend(cls, v):
        if v not in ("ccx", "fenicsx", "mfem"):
            raise ValueError(
                f"backend must be 'ccx', 'fenicsx', or 'mfem', got {v!r}"
            )
        return v

    @validator("xgr", "ygr")
    def validate_grooves(cls, v):
        for groove in v:
            if len(groove) != 4:
                raise ValueError(
                    "Each groove must have 4 values: offset, spacing, depth, width"
                )
        return v

    @validator("curvature")
    def validate_curvature(cls, v):
        allowed = {"kx", "ky"}
        extra = set(v) - allowed
        if extra:
            raise ValueError(
                f"curvature only accepts {sorted(allowed)} (1/length), got extra {sorted(extra)}"
            )
        for key, val in v.items():
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise ValueError(f"curvature {key!r} must be a number, got {val!r}")
        return v


def _run_ccx_backend(mesh, name, dct, status=None):
    if status is not None:
        status.update("Generating CCX input files")
    inpfiles = vtstoccx(
        mesh,
        f"{name}.inp",
        dct["resin"],
        dct["core"],
        dct.get("face"),
        element_type=dct.get("element_type", "C3D8"),
    )

    if status is not None:
        status.update("Running CCX simulations")
    outfiles = runccx(inpfiles)

    if status is not None:
        status.update("Converting FRD to VTU")
    frd2vtu(outfiles)

    vtus = [pv.read(f.replace(".frd", ".vtu")) for f in outfiles]
    datfiles = [f.replace(".frd", ".dat") for f in outfiles]

    if status is not None:
        status.update("Postprocessing CCX results")
    return postprocess(vtus, datfiles, dct["thickness"])


def _run_fenicsx_backend(mesh, dct, status=None):
    if status is not None:
        status.update("Running FEniCSx simulations")
    return runfenicsx(mesh, dct["resin"], dct["core"], dct.get("face"))


def _run_mfem_backend(mesh, dct, status=None):
    if status is not None:
        status.update("Running MFEM simulations")
    return runmfem(mesh, dct["resin"], dct["core"], dct.get("face"))


def cprop(json_data):
    """Run FEA analysis on a JSON configuration."""
    console = Console()
    if isinstance(json_data, str):
        dct = json.load(open(json_data, "r"))
        dirname = os.path.dirname(json_data)
    else:
        dct = json_data
        dirname = "."

    # Validate input
    validated = CpropInput(**dct)
    dct = validated.dict()

    dct["hash"] = hashlib.md5(str(dct).encode()).hexdigest()

    name = f"{dirname}/run{dct['hash']}"
    oname = f"{name}.json"
    log_file = f"{name}.log"

    if os.path.isfile(oname):
        raise FileExistsError(f"Output file {oname} already exists")

    with open(log_file, "w") as log:
        with console.status("[bold green]Running FEA analysis...") as status:
            status.update("Creating mesh")
            mesh = create_grooved_mesh(
                dct["thickness"],
                dct["dx"],
                dct["dy"],
                dct["xgr"],
                dct["ygr"],
                madd=dct["madd"],
                tface=dct.get("face", {}).get("thickness", 0.0),
                kx=dct.get("curvature", {}).get("kx", 0.0),
                ky=dct.get("curvature", {}).get("ky", 0.0),
            )

            status.update("Performing geometric analysis")
            geom_output = geom_analysis(mesh)

            geom_output["rho_infused"] = (
                dct["core"]["rho"] * (1.0 - geom_output["resin_vf"])
                + dct["resin"]["rho"] * geom_output["resin_vf"]
            )

            backend = dct["backend"]
            if backend == "ccx":
                output = _run_ccx_backend(mesh, name, dct, status=status)
                if dct["validate_with_ccx"]:
                    fenicsx_output = _run_fenicsx_backend(mesh, dct, status=status)
                    output["fenicsx_validation"] = validate_against_ccx(
                        output, fenicsx_output, label="fenicsx"
                    )
            else:
                if backend == "fenicsx":
                    output = _run_fenicsx_backend(mesh, dct, status=status)
                else:
                    output = _run_mfem_backend(mesh, dct, status=status)
                if dct["validate_with_ccx"]:
                    ccx_output = _run_ccx_backend(mesh, name, dct, status=status)
                    output["ccx_validation"] = validate_against_ccx(
                        ccx_output, output, label=backend
                    )

            output.update(geom_output)
            output.update(dct)

            json.dump(output, open(oname, "w"), indent=4)
            console.print(f"[bold green]Written to {oname}[/bold green]")
    return output


def homogenize(json_data, *, name: str | None = None) -> CoreResult:
    """Run the full pipeline and return a b3_mat-backed `CoreResult`.

    Thin wrapper over `cprop` for callers that want the homogenized result as
    a `b3_mat.OrthotropicMaterial` (ready for the b3 section / beam pipeline)
    rather than the raw output dict that `cprop` writes to JSON. Requires the
    orthotropic engineering constants the ccx backend produces.
    """
    output = cprop(json_data)
    return CoreResult.from_cprop_output(output, name=name)
