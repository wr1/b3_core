"""Canonical case hash.

The package version is not part of the key. ``CACHE_SCHEMA`` and the
per-backend solver stamp are. Bump the stamp when numerics change.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from b3_core.models import CaseInput
from b3_core.solvers.stamps import stamp_for

CACHE_SCHEMA = 2


def _strip_material_provenance(node: object) -> None:
    """Drop ``source`` and ``reference``. They are not part of the cache key."""
    if isinstance(node, dict):
        if "rho" in node:
            node.pop("source", None)
            node.pop("reference", None)
        for value in node.values():
            _strip_material_provenance(value)
    elif isinstance(node, list):
        for item in node:
            _strip_material_provenance(item)


def canonical_json(case: CaseInput) -> str:
    """Stable JSON of a validated case. Key order and ``_note`` keys are gone.

    ``Material.source`` and ``Material.reference`` are omitted so a
    calibration label does not split the cache.
    """
    payload = case.model_dump(mode="json")
    _strip_material_provenance(payload)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class CacheKey:
    """Identity of one cached solve."""

    schema: int
    solver_stamp: str
    backend: str
    case_json: str

    def body(self) -> str:
        return (
            f"b3_core-case/{self.schema}/{self.solver_stamp}/"
            f"{self.backend}/{self.case_json}"
        )

    def hexdigest(self) -> str:
        return hashlib.sha256(self.body().encode()).hexdigest()


def cache_key(
    case: CaseInput,
    *,
    backend: str,
    solver_stamp: str | None = None,
) -> CacheKey:
    """Key for ``backend``. Pass ``solver_stamp`` only to compare an old stamp."""
    return CacheKey(
        schema=CACHE_SCHEMA,
        solver_stamp=stamp_for(backend) if solver_stamp is None else solver_stamp,
        backend=backend,
        case_json=canonical_json(case),
    )


def case_hash(
    case: CaseInput,
    *,
    backend: str,
    solver_stamp: str | None = None,
) -> str:
    """SHA-256 of schema, solver stamp, resolved backend, and canonical JSON."""
    return cache_key(case, backend=backend, solver_stamp=solver_stamp).hexdigest()
