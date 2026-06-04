"""Output of a homogenization run.

Wraps the engineering constants extracted from the six periodic-BC load
cases as a `b3_mat.OrthotropicMaterial`, so the result drops straight into
the b3 material ecosystem.
"""

from __future__ import annotations

from b3_mat.materials import OrthotropicMaterial
from pydantic import BaseModel, ConfigDict, Field


class CoreResult(BaseModel):
    """Homogenized properties of a grooved, infused core."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    material: OrthotropicMaterial
    resin_volume_fraction: float = Field(..., ge=0, le=1)
    surface_area_factor: float = Field(
        ..., description="Core surface area including groove walls / ungrooved area"
    )
    engineering_constants: dict[str, float] = Field(
        ..., description="Raw per-loadcase engineering constants from FEA"
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
        return cls(
            material=material,
            resin_volume_fraction=resin_volume_fraction,
            surface_area_factor=surface_area_factor,
            engineering_constants=eng,
        )

    @classmethod
    def from_cprop_output(
        cls, output: dict, *, name: str | None = None
    ) -> CoreResult:
        """Build a CoreResult from the dict returned by `cprop`.

        Pulls the engineering constants plus `rho_infused`, `resin_vf` and
        `area_increase` that the pipeline writes into its output dict.
        """
        keys = ("Exx", "Eyy", "Ezz", "Gxy", "Gxz", "Gyz", "nuxy", "nuxz", "nuyz")
        missing = [k for k in keys if k not in output]
        if missing:
            msg = (
                f"cprop output is missing engineering constants {missing}; "
                "the b3_mat wrapper currently supports the orthotropic ccx backend."
            )
            raise KeyError(msg)
        return cls.from_engineering_constants(
            {k: output[k] for k in keys},
            rho=output["rho_infused"],
            resin_volume_fraction=output["resin_vf"],
            surface_area_factor=output["area_increase"],
            name=name,
        )
