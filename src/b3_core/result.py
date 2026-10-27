"""Output of a homogenization run.

Wraps the engineering constants extracted from the six periodic-BC load
cases as a `b3_mat.OrthotropicMaterial`, so the result drops straight into
the b3 material ecosystem.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from b3_mat.materials import OrthotropicMaterial
from pydantic import BaseModel, ConfigDict, Field

from b3_core.provenance import Provenance


class CoreResult(BaseModel):
    """Homogenized properties of a grooved, infused core."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    material: OrthotropicMaterial
    resin_volume_fraction: float = Field(..., ge=0, le=1)
    surface_area_factor: float = Field(
        ..., description="Core surface area including groove walls / ungrooved area"
    )
    engineering_constants: dict[str, float] = Field(
        ..., description="Legacy Exx/Eyy/Ezz constants (0.3). Prefer properties."
    )
    properties: dict[str, float] = Field(
        default_factory=dict,
        description="Canonical Ex/Ey/Ez constants",
    )
    stiffness: Any = None
    provenance: Provenance | None = None
    kerfs: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Kerf half-widths after curvature is applied as hw(z). "
            "Each row has axis, mouth, pitch_mm, hw_root_mm, hw_mouth_mm."
        ),
    )

    @classmethod
    def from_engineering_constants(
        cls,
        eng: dict[str, float],
        *,
        rho: float,
        resin_volume_fraction: float,
        surface_area_factor: float,
        name: str | None = None,
    ) -> CoreResult:
        """Build a CoreResult from the six-loadcase engineering-constant dict.

        `eng` must contain Exx, Eyy, Ezz, Gxy, Gxz, Gyz, nuxy, nuxz, nuyz.
        """
        material = OrthotropicMaterial(
            name=name,
            Ex=eng["Exx"],
            Ey=eng["Eyy"],
            Ez=eng["Ezz"],
            Gxy=eng["Gxy"],
            Gxz=eng["Gxz"],
            Gyz=eng["Gyz"],
            nuxy=eng["nuxy"],
            nuxz=eng["nuxz"],
            nuyz=eng["nuyz"],
            rho=rho,
        )
        canonical = {
            "Ex": eng["Exx"],
            "Ey": eng["Eyy"],
            "Ez": eng["Ezz"],
            "Gxy": eng["Gxy"],
            "Gxz": eng["Gxz"],
            "Gyz": eng["Gyz"],
            "nuxy": eng["nuxy"],
            "nuxz": eng["nuxz"],
            "nuyz": eng["nuyz"],
        }
        return cls(
            material=material,
            resin_volume_fraction=resin_volume_fraction,
            surface_area_factor=surface_area_factor,
            engineering_constants=eng,
            properties=canonical,
        )

    @classmethod
    def from_cprop_output(cls, output: dict, *, name: str | None = None) -> CoreResult:
        """Build a CoreResult from the dict returned by `cprop`.

        Pulls the engineering constants plus `rho_infused`, `resin_vf` and
        `area_increase` that the pipeline writes into its output dict.
        """
        keys = ("Exx", "Eyy", "Ezz", "Gxy", "Gxz", "Gyz", "nuxy", "nuxz", "nuyz")
        missing = [k for k in keys if k not in output]
        if missing:
            msg = (
                f"run output is missing engineering constants {missing}; "
                "expected keys from solvers.elasticity.PROPERTY_KEYS"
            )
            raise KeyError(msg)
        return cls.from_engineering_constants(
            {k: output[k] for k in keys},
            rho=output["rho_infused"],
            resin_volume_fraction=output["resin_vf"],
            surface_area_factor=output["area_increase"],
            name=name,
        )

    @classmethod
    def from_record(cls, record: RunRecord, *, name: str | None = None) -> CoreResult:
        """Build a CoreResult from a namespaced :class:`RunRecord`."""
        props = dict(record.result.properties)
        eng = {
            "Exx": float(props.get("Ex", props.get("Exx"))),
            "Eyy": float(props.get("Ey", props.get("Eyy"))),
            "Ezz": float(props.get("Ez", props.get("Ezz"))),
            "Gxy": float(props["Gxy"]),
            "Gxz": float(props["Gxz"]),
            "Gyz": float(props["Gyz"]),
            "nuxy": float(props["nuxy"]),
            "nuxz": float(props["nuxz"]),
            "nuyz": float(props["nuyz"]),
        }
        geom = record.geometry
        built = cls.from_engineering_constants(
            eng,
            rho=float(geom["rho_infused"]),
            resin_volume_fraction=float(geom["resin_vf"]),
            surface_area_factor=float(geom["area_increase"]),
            name=name,
        )
        canonical = {
            "Ex": eng["Exx"],
            "Ey": eng["Eyy"],
            "Ez": eng["Ezz"],
            "Gxy": eng["Gxy"],
            "Gxz": eng["Gxz"],
            "Gyz": eng["Gyz"],
            "nuxy": eng["nuxy"],
            "nuxz": eng["nuxz"],
            "nuyz": eng["nuyz"],
        }
        for key in ("nuyx", "nuzx", "nuzy"):
            if key in props:
                canonical[key] = float(props[key])
        stiffness = np.asarray(record.result.stiffness, dtype=float)
        return built.model_copy(
            update={
                "properties": canonical,
                "stiffness": stiffness,
                "kerfs": [dict(row) for row in geom.get("kerfs") or []],
                "provenance": record.provenance,
            }
        )

    def ccx_ortho(
        self,
        *,
        name: str | None = None,
        temperature: float | None = 293.0,
    ) -> str:
        """CalculiX ``*elastic,type=ortho`` card from the nine constants.

        Reconstructs C from ``material`` (Voigt xx,yy,zz,yz,xz,xy) and emits
        D1111…D2323. Includes ``*density`` from ``material.rho``.
        """
        from b3_core.export.ccx import ccx_ortho_card
        from b3_core.solvers.elasticity import orthotropic_C

        m = self.material
        C = orthotropic_C(m.Ex, m.Ey, m.Ez, m.Gxy, m.Gxz, m.Gyz, m.nuxy, m.nuxz, m.nuyz)
        from b3_core.provenance import comment_line

        mat_name = name or m.name or "core_hom"
        return ccx_ortho_card(
            C,
            name=mat_name,
            rho=m.rho,
            temperature=temperature,
            provenance_comment=comment_line(self.provenance),
        )


class ResultBlock(BaseModel):
    backend: str
    stiffness: list[list[float]]
    properties: dict[str, float]
    extras: dict[str, float] = Field(default_factory=dict)


class RunRecord(BaseModel):
    """Namespaced homogenisation record stored by caches and ``homogenize_to_disk``."""

    model_config = ConfigDict(populate_by_name=True)

    schema_name: str = Field("b3_core.run/2", alias="schema")
    case_hash: str
    b3_core_version: str
    solver_stamp: str = ""
    provenance: Provenance | None = Field(default=None, alias="b3_core")
    input: dict[str, Any]
    geometry: dict[str, Any]
    result: ResultBlock
    validation: dict[str, Any] | None = None
    diagnostics: dict[str, Any] = Field(
        default_factory=lambda: {"checks": [], "ties": []}
    )

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("by_alias", True)
        return super().model_dump(**kwargs)

    def to_agent_json(
        self,
        *,
        units: str = "Pa",
        cache_hit: bool = False,
        elapsed_s: float | None = None,
        written: str | None = None,
    ) -> dict[str, Any]:
        """Machine-readable payload. Moduli are scaled only when ``units='GPa'``."""
        if units not in ("Pa", "GPa"):
            raise ValueError("units must be 'Pa' or 'GPa'")
        scale = 1e-9 if units == "GPa" else 1.0
        moduli = {"Ex", "Ey", "Ez", "Gxy", "Gxz", "Gyz", "Exx", "Eyy", "Ezz"}
        properties: dict[str, Any] = {}
        for key, value in self.result.properties.items():
            number = float(value)
            if key in moduli:
                properties[key] = {"value": number * scale, "unit": units}
            else:
                properties[key] = {"value": number, "unit": "-"}
        geometry = dict(self.geometry)
        if "rho_infused" in geometry:
            geometry["rho_infused"] = {
                "value": float(geometry["rho_infused"]),
                "unit": "kg/m^3",
            }
        stiffness = [
            [float(entry) * scale for entry in row] for row in self.result.stiffness
        ]
        provenance = None
        if self.provenance is not None:
            provenance = self.provenance.model_dump(mode="json")
        return {
            "case_hash": self.case_hash,
            "backend": self.result.backend,
            "cache_hit": bool(cache_hit),
            "elapsed_s": None if elapsed_s is None else float(elapsed_s),
            "written": written,
            "provenance": provenance,
            "properties": properties,
            "stiffness": stiffness,
            "stiffness_unit": units,
            "geometry": geometry,
            "diagnostics": {
                "checks": list(self.diagnostics.get("checks", [])),
                "ties": list(self.diagnostics.get("ties", [])),
            },
        }

    def flat(self) -> dict[str, Any]:
        """Legacy flat dict: input fields, geometry, and ``Exx``/``Eyy``/``Ezz``."""
        props = self.result.properties
        eng = {
            "Exx": float(props.get("Ex", props.get("Exx"))),
            "Eyy": float(props.get("Ey", props.get("Eyy"))),
            "Ezz": float(props.get("Ez", props.get("Ezz"))),
            "Gxy": float(props["Gxy"]),
            "Gxz": float(props["Gxz"]),
            "Gyz": float(props["Gyz"]),
            "nuxy": float(props["nuxy"]),
            "nuxz": float(props["nuxz"]),
            "nuyz": float(props["nuyz"]),
        }
        for key in ("nuyx", "nuzx", "nuzy"):
            if key in props:
                eng[key] = float(props[key])
        out: dict[str, Any] = {}
        out.update(self.input)
        out.update(self.geometry)
        out.update(eng)
        out["backend"] = self.result.backend
        out["stiffness"] = self.result.stiffness
        out["case_hash"] = self.case_hash
        if self.validation is not None:
            out["ccx_validation"] = self.validation
        if self.provenance is not None:
            out["b3_core"] = self.provenance.model_dump(mode="json")
        return out
