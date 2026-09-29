"""Typed RVE case: materials, grooves, curvature, scoring.

``z = 0`` is the mould face. A groove with ``mouth="top"`` opens at
``z = thickness``; ``mouth="bottom"`` opens at ``z = 0``. Legacy 4-lists
``[offset, pitch, depth, width]`` still load: a negative depth means
``mouth="top"``.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*$")


class CellSizeDist(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mean: float = Field(gt=0)
    std: float = Field(ge=0)
    dist: Literal["lognormal", "normal"] = "lognormal"


class Material(BaseModel):
    """Isotropic (``E``, ``nu``) or orthotropic (``Ex``…``nuyz``) constituent.

    Orthotropic axes are x, y, z. Legacy names ``E1``…``nu23`` are accepted
    on input and serialised as the canonical names.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    rho: float = Field(..., gt=0)
    E: float | None = Field(None, gt=0)
    nu: float | None = Field(None, ge=0, lt=0.5)
    Ex: float | None = Field(None, gt=0, validation_alias=AliasChoices("Ex", "E1"))
    Ey: float | None = Field(None, gt=0, validation_alias=AliasChoices("Ey", "E2"))
    Ez: float | None = Field(None, gt=0, validation_alias=AliasChoices("Ez", "E3"))
    Gxy: float | None = Field(None, gt=0, validation_alias=AliasChoices("Gxy", "G12"))
    Gxz: float | None = Field(None, gt=0, validation_alias=AliasChoices("Gxz", "G13"))
    Gyz: float | None = Field(None, gt=0, validation_alias=AliasChoices("Gyz", "G23"))
    nuxy: float | None = Field(None, validation_alias=AliasChoices("nuxy", "nu12"))
    nuxz: float | None = Field(None, validation_alias=AliasChoices("nuxz", "nu13"))
    nuyz: float | None = Field(None, validation_alias=AliasChoices("nuyz", "nu23"))
    cell_size: float | CellSizeDist | None = None

    @field_validator("cell_size", mode="before")
    @classmethod
    def _cell_size(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return CellSizeDist.model_validate(value)
        return value

    @property
    def is_orthotropic(self) -> bool:
        return self.Ex is not None

    @property
    def E1(self) -> float | None:
        return self.Ex

    @property
    def E2(self) -> float | None:
        return self.Ey

    @property
    def E3(self) -> float | None:
        return self.Ez

    @model_validator(mode="after")
    def _complete(self) -> Material:
        iso = self.E is not None and self.nu is not None
        ortho = all(
            getattr(self, name) is not None
            for name in ("Ex", "Ey", "Ez", "Gxy", "Gxz", "Gyz", "nuxy", "nuxz", "nuyz")
        )
        if not (iso or ortho):
            raise ValueError(
                "Material must be isotropic (E, nu) or orthotropic "
                "(Ex, Ey, Ez, Gxy, Gxz, Gyz, nuxy, nuxz, nuyz)"
            )
        return self


class Face(BaseModel):
    """Optional face sheet. ``thickness`` is in mm. ``z = 0`` is the mould face."""

    model_config = ConfigDict(extra="allow")

    thickness: float = Field(0.0, ge=0)
    material: Material | None = None


class Curvature(BaseModel):
    """Mould curvature in 1/mm. Positive ``kx`` opens a top-mouth x-groove."""

    model_config = ConfigDict(extra="forbid")

    kx: float = 0.0
    ky: float = 0.0

    def __getitem__(self, key: str) -> float:
        if key not in ("kx", "ky"):
            raise KeyError(key)
        return float(getattr(self, key))


class SurfaceHalo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    cell_size: float | CellSizeDist | None = None
    scale: float | None = Field(None, gt=0)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    @field_validator("cell_size", mode="before")
    @classmethod
    def _cell_size(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return CellSizeDist.model_validate(value)
        return value


class SamplingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["exact", "local_cloud"] = "local_cloud"
    resolution: int = Field(3, ge=1)
    idw_power: float = Field(2.0, gt=0)


class Scoring(BaseModel):
    model_config = ConfigDict(extra="forbid")

    damage_cells: float = Field(1.0, gt=0)
    surfaces: dict[str, SurfaceHalo] = Field(default_factory=dict)
    sampling: SamplingSpec = Field(default_factory=SamplingSpec)

    @field_validator("surfaces")
    @classmethod
    def _surface_names(cls, value: dict[str, SurfaceHalo]) -> dict[str, SurfaceHalo]:
        unknown = set(value) - {"saw_cut", "face"}
        if unknown:
            raise ValueError(
                f"scoring.surfaces only accepts saw_cut and face, got {sorted(unknown)}"
            )
        return value

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


class Groove(BaseModel):
    """One saw cut. ``depth`` is always positive; ``mouth`` says which face opens.

    ``bottom`` opens at the mould face ``z = 0``. ``top`` opens at
    ``z = thickness``. Index ``[2]`` is the legacy signed depth
    (negative when ``mouth`` is ``top``) so older call sites keep working.
    """

    model_config = ConfigDict(extra="forbid")

    offset: float
    pitch: float = Field(gt=0)
    depth: float = Field(gt=0)
    width: float = Field(gt=0)
    mouth: Literal["bottom", "top"] = "bottom"

    @model_validator(mode="before")
    @classmethod
    def _from_list(cls, value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            if len(value) != 4:
                raise ValueError(
                    "Each groove must have 4 values: offset, spacing, depth, width"
                )
            offset, pitch, depth, width = value
            depth_f = float(depth)
            if depth_f < 0:
                return {
                    "offset": offset,
                    "pitch": pitch,
                    "depth": -depth_f,
                    "width": width,
                    "mouth": "top",
                }
            return {
                "offset": offset,
                "pitch": pitch,
                "depth": depth_f,
                "width": width,
                "mouth": "bottom",
            }
        return value

    @property
    def signed_depth(self) -> float:
        """Legacy signed depth: negative when the mouth is at ``z = thickness``."""
        if self.mouth == "top":
            return -self.depth
        return self.depth

    def as_cut(self) -> list[float]:
        """``[offset, pitch, signed_depth, width]`` for the mesh kernel."""
        return [
            float(self.offset),
            float(self.pitch),
            float(self.signed_depth),
            float(self.width),
        ]

    def __getitem__(self, index: int) -> float:
        return self.as_cut()[index]

    def __len__(self) -> int:
        return 4


class CaseInput(BaseModel):
    """One homogenisation case. ``backend="auto"`` is resolved at solve time."""

    model_config = ConfigDict(extra="forbid")

    dx: float = Field(..., gt=0)
    dy: float = Field(..., gt=0)
    thickness: float = Field(..., gt=0)
    xgr: list[Groove] = Field(default_factory=list)
    ygr: list[Groove] = Field(default_factory=list)
    core: Material
    resin: Material
    madd: list[float] = Field(default_factory=lambda: [0.0])
    face: Face | None = None
    curvature: Curvature = Field(default_factory=Curvature)
    scoring: Scoring | None = None
    element_type: str = "C3D8"
    backend: str = "auto"
    validate_with_ccx: bool = False

    @field_validator("element_type")
    @classmethod
    def _element_type(cls, value: str) -> str:
        if value not in ("C3D8", "C3D20"):
            raise ValueError(f"element_type must be 'C3D8' or 'C3D20', got {value!r}")
        return value

    @field_validator("backend")
    @classmethod
    def _backend_name(cls, value: str) -> str:
        if not value or _IDENT.fullmatch(value) is None:
            raise ValueError(f"backend must be a non-empty identifier, got {value!r}")
        return value

    @field_validator("face", "scoring", mode="before")
    @classmethod
    def _empty_mapping(cls, value: Any) -> Any:
        if value == {} or value is None:
            return None
        return value

    @field_validator("curvature", mode="before")
    @classmethod
    def _curvature(cls, value: Any) -> Any:
        if value is None or value == {}:
            return {}
        return value

    @model_validator(mode="after")
    def _depth_fits(self) -> CaseInput:
        for groove in (*self.xgr, *self.ygr):
            if groove.depth > self.thickness + 1e-9:
                raise ValueError(
                    f"groove depth {groove.depth} exceeds thickness {self.thickness}"
                )
        return self

    @property
    def is_orthotropic(self) -> bool:
        return self.core.is_orthotropic or self.resin.is_orthotropic

    def mesh_kwargs(self) -> dict[str, Any]:
        """Arguments for :func:`b3_core.core.mesh.create_grooved_mesh`."""
        face_thickness = 0.0 if self.face is None else float(self.face.thickness)
        curvature = self.curvature
        if isinstance(curvature, dict):
            kx = float(curvature.get("kx", 0.0))
            ky = float(curvature.get("ky", 0.0))
        else:
            kx = float(curvature.kx)
            ky = float(curvature.ky)
        return {
            "thickness": float(self.thickness),
            "dx": float(self.dx),
            "dy": float(self.dy),
            "xcuts": [groove.as_cut() for groove in self.xgr],
            "ycuts": [groove.as_cut() for groove in self.ygr],
            "madd": tuple(self.madd),
            "tface": face_thickness,
            "kx": kx,
            "ky": ky,
        }

    def score_dict(self) -> dict[str, Any]:
        """Dict shape :class:`~b3_core.core.scoring.ScoreField` already accepts."""
        payload = self.model_dump(mode="python")
        payload["xgr"] = [groove.as_cut() for groove in self.xgr]
        payload["ygr"] = [groove.as_cut() for groove in self.ygr]
        curvature = self.curvature
        if isinstance(curvature, dict):
            payload["curvature"] = {
                "kx": float(curvature.get("kx", 0.0)),
                "ky": float(curvature.get("ky", 0.0)),
            }
        else:
            payload["curvature"] = {
                "kx": float(curvature.kx),
                "ky": float(curvature.ky),
            }
        payload["face"] = {} if self.face is None else self.face.model_dump()
        scoring = self.scoring
        if scoring is None:
            payload["scoring"] = {}
        elif hasattr(scoring, "model_dump"):
            payload["scoring"] = scoring.model_dump()
        else:
            payload["scoring"] = dict(scoring)
        for key in ("core", "resin"):
            block = payload[key]
            if isinstance(block.get("cell_size"), CellSizeDist):
                block["cell_size"] = block["cell_size"].model_dump()
            elif hasattr(block.get("cell_size"), "model_dump"):
                block["cell_size"] = block["cell_size"].model_dump()
        return payload


# Permanent alias. ``cprop`` the function goes away in 1.0; this name stays.
CpropInput = CaseInput

__all__ = [
    "CaseInput",
    "CellSizeDist",
    "CpropInput",
    "Curvature",
    "Face",
    "Groove",
    "Material",
    "SamplingSpec",
    "Scoring",
    "SurfaceHalo",
]
