"""CoreModel — one lazily-evaluated handle on a grooved-core case.

Builds the RVE mesh, geometric metrics and (on demand) the MFEM homogenisation
once, and caches them, so every renderer in :mod:`b3_core.viz` shares the same
mesh and stiffness instead of each re-solving. This is the single place the
mesh-build + MFEM-solve pipeline lives for visualization.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from b3_core.core.analysis import geom_analysis
from b3_core.core.cprop import CpropInput
from b3_core.core.mesh import create_grooved_mesh
from b3_core.io import mfem_backend
from b3_core.viz import geometry, tensor

logger = logging.getLogger(__name__)


class CoreModel:
    """A grooved-core case with lazy, cached mesh / geometry / homogenisation."""

    def __init__(self, inp: dict, *, name: str | None = None, config_path: str = ""):
        self.inp = CpropInput(**inp).model_dump()
        self.name = name or "core"
        self.config_path = config_path
        self._mesh = None
        self._mat = None
        self._geom = None
        self._details = None

    @classmethod
    def from_json(cls, path: str | Path) -> "CoreModel":
        path = Path(path)
        return cls(json.loads(path.read_text()), name=path.stem, config_path=path.name)

    @classmethod
    def from_dict(cls, inp: dict, **kw) -> "CoreModel":
        return cls(inp, **kw)

    # -- geometry -----------------------------------------------------------
    @property
    def mesh(self):
        if self._mesh is None:
            from b3_core.core.cprop import halo_reach

            i = self.inp
            self._mesh = create_grooved_mesh(
                thickness=i["thickness"], dx=i["dx"], dy=i["dy"],
                xcuts=i["xgr"], ycuts=i["ygr"], madd=tuple(i["madd"]),
                tface=(i.get("face") or {}).get("thickness", 0.0),
                kx=(i.get("curvature") or {}).get("kx", 0.0),
                ky=(i.get("curvature") or {}).get("ky", 0.0),
                s_halo=halo_reach(i),
            )
        return self._mesh

    @property
    def material_codes(self) -> np.ndarray:
        if self._mat is None:
            self._mat = geometry.cell_material(self.mesh)
        return self._mat

    @property
    def axis_vectors(self):
        return geometry.axis_vectors(self.mesh)

    @property
    def geom(self) -> dict:
        if self._geom is None:
            from b3_core.core.cprop import _score_field
            from b3_core.core.scoring import effective_resin_vf

            g = geom_analysis(self.mesh)
            field = _score_field(self.inp)
            eff, halo_vf = effective_resin_vf(self.mesh, field, g["resin_vf"])
            if field is not None:
                g["effective_resin_vf"] = eff
                g["halo_vf"] = halo_vf
            g["rho_infused"] = (
                self.inp["core"]["rho"] * (1.0 - eff)
                + self.inp["resin"]["rho"] * eff
            )
            self._geom = g
        return self._geom

    # -- homogenisation -----------------------------------------------------
    def _needs_numpy(self) -> bool:
        from b3_core.core.cprop import _needs_numpy

        return self.inp.get("backend") == "numpy" or _needs_numpy(self.inp)

    @property
    def details(self):
        if self._details is None:
            if self._needs_numpy():
                from b3_core.core.cprop import _score_field
                from b3_core.io import aniso

                logger.info("running numpy anisotropic backend for %s", self.name)
                self._details = aniso.runnumpy(
                    self.mesh, self.inp["resin"], self.inp["core"],
                    self.inp.get("face"), score_field=_score_field(self.inp),
                    scoring=self.inp.get("scoring"), return_details=True,
                )
            else:
                logger.info("running MFEM backend for %s", self.name)
                self._details = mfem_backend.runmfem(
                    self.mesh, self.inp["resin"], self.inp["core"],
                    self.inp.get("face"), return_details=True,
                )
        return self._details

    @property
    def stiffness(self) -> np.ndarray:
        """Effective 6x6 stiffness C_eff (Pa, order xx,yy,zz,yz,xz,xy)."""
        return np.asarray(self.details.stiffness, dtype=float)

    @property
    def compliance(self) -> np.ndarray:
        return np.asarray(self.details.compliance, dtype=float)

    @property
    def engineering_constants(self) -> dict[str, float]:
        """Orthotropic constants (E_x, E_y, E_z, G_*, nu_*) from the tensor."""
        return tensor.engineering_constants(self.stiffness)

    def displacements(self, case: str) -> np.ndarray:
        """Total periodic displacement u = E.x + w for one load case (metres)."""
        return np.asarray(self.details.displacements[case])
