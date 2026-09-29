"""CoreModel — a view over one prepared case and its homogenisation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from b3_core.loaders import from_dict, from_path, normalize_case
from b3_core.models import CaseInput
from b3_core.pipeline import BackendCapabilityError
from b3_core.viz import geometry, tensor

logger = logging.getLogger(__name__)


class CoreModel:
    """Grooved-core case with lazy mesh, geometry, and stiffness."""

    def __init__(
        self,
        case: Any,
        *,
        name: str | None = None,
        config_path: str = "",
        backend: str | None = None,
        cache: Any = None,
    ) -> None:
        model, _workdir = normalize_case(case)
        self.case: CaseInput = model
        self.inp = model.score_dict()
        self.name = name or "core"
        self.config_path = config_path
        self.backend = backend
        self.cache = cache
        self.prepared = None
        self.material_codes_cache = None
        self.geometry_cache = None
        self.solved = None

    @classmethod
    def from_path(cls, path: str | Path, **kwargs: Any) -> CoreModel:
        path = Path(path)
        textile = from_path(path)
        return cls(
            textile,
            name=kwargs.pop("name", path.stem),
            config_path=path.name,
            **kwargs,
        )

    @classmethod
    def from_json(cls, path: str | Path, **kwargs: Any) -> CoreModel:
        """Alias of :meth:`from_path`. JSON and YAML both load."""
        return cls.from_path(path, **kwargs)

    @classmethod
    def from_dict(cls, data: dict, **kwargs: Any) -> CoreModel:
        return cls(from_dict(data), **kwargs)

    def ensure_prepared(self):
        if self.prepared is None:
            from b3_core.pipeline import prepare

            self.prepared = prepare(self.case)
        return self.prepared

    @property
    def mesh(self):
        return self.ensure_prepared().mesh

    @property
    def material_codes(self) -> np.ndarray:
        if self.material_codes_cache is None:
            self.material_codes_cache = geometry.cell_material(self.mesh)
        return self.material_codes_cache

    @property
    def axis_vectors(self):
        return geometry.axis_vectors(self.mesh)

    @property
    def geom(self) -> dict:
        if self.geometry_cache is None:
            self.geometry_cache = self.ensure_prepared().geometry.as_dict()
        return self.geometry_cache

    @property
    def details(self):
        if self.solved is None:
            from b3_core.api import run_case
            from b3_core.pipeline import solve_result_from_record

            record = run_case(self.case, backend=self.backend, cache=self.cache)
            self.solved = solve_result_from_record(record)
            self.resolved_backend = record.result.backend
            self.displacement_fields = None
            logger.info("solved %s with %s", self.name, self.resolved_backend)
        return self.solved

    @property
    def stiffness(self) -> np.ndarray:
        """Effective 6×6 stiffness (Pa, order xx, yy, zz, yz, xz, xy)."""
        return np.asarray(self.details.stiffness, dtype=float)

    def ccx_ortho(
        self,
        *,
        name: str | None = None,
        temperature: float | None = 293.0,
    ) -> str:
        """CalculiX ``*elastic,type=ortho`` card from the effective stiffness."""
        from b3_core.export.ccx import ccx_ortho_card

        return ccx_ortho_card(
            self.stiffness,
            name=name or self.name,
            rho=float(self.geom["rho_infused"]),
            temperature=temperature,
        )

    @property
    def compliance(self) -> np.ndarray:
        return np.asarray(self.details.compliance, dtype=float)

    @property
    def engineering_constants(self) -> dict[str, float]:
        """Orthotropic constants from the stiffness tensor."""
        return tensor.engineering_constants(self.stiffness)

    def displacements(self, case: str) -> np.ndarray:
        """Total periodic displacement for one load case (metres)."""
        from b3_core.pipeline import backend_capabilities, run_pipeline

        name = getattr(self, "resolved_backend", None)
        if name is None:
            _ = self.details
            name = self.resolved_backend
        caps = backend_capabilities(name)
        fields = getattr(self, "displacement_fields", None)
        if fields is None:
            if not caps.displacements:
                raise BackendCapabilityError(
                    f"backend {name!r} cannot return displacements"
                )
            _prep, solved, resolved = run_pipeline(
                self.case, backend=name, details=True
            )
            self.displacement_fields = solved.displacements
            self.resolved_backend = resolved
            fields = solved.displacements
        field = None if fields is None else fields.get(case)
        if not caps.displacements or field is None:
            raise BackendCapabilityError(
                f"backend {name!r} cannot return displacements"
            )
        return np.asarray(field)
