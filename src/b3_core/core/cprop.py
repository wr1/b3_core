#!/usr/bin/env python3

import json
import hashlib
import logging
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
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

_ORTHO_FIELDS = ("E1", "E2", "E3", "G12", "G13", "G23", "nu12", "nu13", "nu23")


class Material(BaseModel):
    """Isotropic (E, nu) or orthotropic (E1..nu23) elastic material.

    Orthotropic axes are 1=x, 2=y, 3=z. Orthotropic materials require the
    `numpy` backend (the ccx/mfem integrators are isotropic-only).
    """

    rho: float = Field(..., gt=0)
    # isotropic
    E: float | None = Field(None, gt=0)
    nu: float | None = Field(None, ge=0, lt=0.5)
    # orthotropic (engineering constants)
    E1: float | None = Field(None, gt=0)
    E2: float | None = Field(None, gt=0)
    E3: float | None = Field(None, gt=0)
    G12: float | None = Field(None, gt=0)
    G13: float | None = Field(None, gt=0)
    G23: float | None = Field(None, gt=0)
    nu12: float | None = None
    nu13: float | None = None
    nu23: float | None = None

    @property
    def is_orthotropic(self) -> bool:
        return self.E1 is not None

    @model_validator(mode="after")
    def _check_complete(self):
        iso = self.E is not None and self.nu is not None
        ortho = all(getattr(self, f) is not None for f in _ORTHO_FIELDS)
        if not (iso or ortho):
            raise ValueError(
                "Material must be isotropic (E, nu) or orthotropic "
                f"({', '.join(_ORTHO_FIELDS)})"
            )
        return self


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

    @field_validator("element_type")
    @classmethod
    def validate_element_type(cls, v):
        if v not in ("C3D8", "C3D20"):
            raise ValueError(f"element_type must be 'C3D8' or 'C3D20', got {v!r}")
        return v

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v):
        if v not in ("ccx", "fenicsx", "mfem", "numpy"):
            raise ValueError(
                f"backend must be 'ccx', 'fenicsx', 'mfem' or 'numpy', got {v!r}"
            )
        return v

    @field_validator("xgr", "ygr")
    @classmethod
    def validate_grooves(cls, v):
        for groove in v:
            if len(groove) != 4:
                raise ValueError(
                    "Each groove must have 4 values: offset, spacing, depth, width"
                )
        return v

    @field_validator("curvature")
    @classmethod
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


def _run_ccx_backend(mesh, name, dct):
    logger.info("generating CCX input files")
    inpfiles = vtstoccx(
        mesh,
        f"{name}.inp",
        dct["resin"],
        dct["core"],
        dct.get("face"),
        element_type=dct.get("element_type", "C3D8"),
    )

    logger.info("running CCX simulations")
    outfiles = runccx(inpfiles)

    logger.info("converting FRD to VTU")
    frd2vtu(outfiles)

    vtus = [pv.read(f.replace(".frd", ".vtu")) for f in outfiles]
    datfiles = [f.replace(".frd", ".dat") for f in outfiles]

    logger.info("postprocessing CCX results")
    return postprocess(vtus, datfiles, dct["thickness"])


def _run_fenicsx_backend(mesh, dct):
    logger.info("running FEniCSx simulations")
    return runfenicsx(mesh, dct["resin"], dct["core"], dct.get("face"))


def _run_mfem_backend(mesh, dct):
    logger.info("running MFEM simulations")
    return runmfem(mesh, dct["resin"], dct["core"], dct.get("face"))


def _run_numpy_backend(mesh, dct):
    from ..io.aniso import runnumpy

    logger.info("running numpy anisotropic homogenisation")
    return runnumpy(mesh, dct["resin"], dct["core"], dct.get("face")).properties


def _is_orthotropic(dct):
    return any((dct.get(p) or {}).get("E1") is not None for p in ("core", "resin"))


def cprop(json_data):
    """Run FEA analysis on a JSON configuration."""
    if isinstance(json_data, str):
        dct = json.load(open(json_data, "r"))
        dirname = os.path.dirname(json_data)
    else:
        dct = json_data
        dirname = "."

    # Validate input
    validated = CpropInput(**dct)
    dct = validated.model_dump()

    dct["hash"] = hashlib.md5(str(dct).encode()).hexdigest()

    name = f"{dirname}/run{dct['hash']}"
    oname = f"{name}.json"

    if os.path.isfile(oname):
        raise FileExistsError(f"Output file {oname} already exists")

    logger.info("creating mesh")
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

    logger.info("performing geometric analysis")
    geom_output = geom_analysis(mesh)

    geom_output["rho_infused"] = (
        dct["core"]["rho"] * (1.0 - geom_output["resin_vf"])
        + dct["resin"]["rho"] * geom_output["resin_vf"]
    )

    backend = dct["backend"]
    # Orthotropic constituents need the anisotropic numpy backend (ccx/mfem
    # integrators are isotropic-only), so route there regardless of the request.
    if _is_orthotropic(dct):
        backend = "numpy"

    if backend == "numpy":
        output = _run_numpy_backend(mesh, dct)
    elif backend == "ccx":
        output = _run_ccx_backend(mesh, name, dct)
        if dct["validate_with_ccx"]:
            fenicsx_output = _run_fenicsx_backend(mesh, dct)
            output["fenicsx_validation"] = validate_against_ccx(
                output, fenicsx_output, label="fenicsx"
            )
    else:
        if backend == "fenicsx":
            output = _run_fenicsx_backend(mesh, dct)
        else:
            output = _run_mfem_backend(mesh, dct)
        if dct["validate_with_ccx"]:
            ccx_output = _run_ccx_backend(mesh, name, dct)
            output["ccx_validation"] = validate_against_ccx(
                ccx_output, output, label=backend
            )

    output.update(geom_output)
    output.update(dct)

    json.dump(output, open(oname, "w"), indent=4)
    logger.info("written results to %s", oname)
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
