"""Attribute a curvature change to the mesh, the resin field, or the wall morph."""

from __future__ import annotations

from typing import Any

import numpy as np

from b3_core.loaders import normalize_case
from b3_core.models import Curvature
from b3_core.pipeline import prepare

_KEYS = ("Ex", "Ey", "Ez", "Gxy", "Gxz", "Gyz")


def _props(record: Any) -> dict[str, float]:
    if isinstance(record, dict):
        source = record
    else:
        source = dict(getattr(record, "properties", {}) or {})
        nested = getattr(record, "result", None)
        if nested is not None:
            source.update(getattr(nested, "properties", {}) or {})
    return {
        key: float(source[key])
        for key in _KEYS
        if key in source and isinstance(source[key], (int, float))
    }


def _default_solve(mesh: Any, field: Any, case: Any) -> dict[str, float]:
    from b3_core.solvers.numpy_fe.backend import runnumpy

    result = runnumpy(
        mesh,
        case.resin,
        case.core,
        case.face,
        score_field=field,
        scoring=case.scoring,
    )
    return _props(result)


def _displacement_mm(flat_mesh: Any, curved_mesh: Any) -> float:
    flat = np.asarray(flat_mesh.points, dtype=float)
    curved = np.asarray(curved_mesh.points, dtype=float)
    peak = 0.0
    for z_value in np.unique(np.round(flat[:, 2], 9)):
        left = np.abs(flat[:, 2] - z_value) < 1e-9
        right = np.abs(curved[:, 2] - z_value) < 1e-9
        if not left.any() or not right.any() or int(left.sum()) != int(right.sum()):
            continue
        peak = max(peak, float(np.max(np.abs(curved[right] - flat[left]))))
    return peak


def _delta_pct(base: dict[str, float], other: dict[str, float]) -> dict[str, float]:
    out = {}
    for key, value in base.items():
        if key not in other or value == 0.0:
            continue
        out[key] = (float(other[key]) - float(value)) / abs(float(value)) * 100.0
    return out


def bisect(
    case: Any,
    *,
    kx: float,
    ky: float = 0.0,
    backend: str = "numpy",
    solve: Any = None,
) -> dict[str, Any]:
    """a flat/flat, b flat mesh + curved field, c curved mesh + flat field, c2 morph off.

    ``solve(mesh, score_field, case)`` returns a property dict. The default is
    the numpy backend. ``max_node_displacement_mm`` compares shared z stations.
    """
    case_in, _workdir = normalize_case(case)
    flat = case_in.model_copy(update={"curvature": Curvature(kx=0.0, ky=0.0)})
    curved = case_in.model_copy(
        update={"curvature": Curvature(kx=float(kx), ky=float(ky))}
    )
    prepared_flat = prepare(flat)
    prepared_curved = prepare(curved)
    prepared_off = prepare(curved, wall_morph=False)
    runner = solve or _default_solve
    arms = {
        "a_flat_mesh_flat_field": runner(
            prepared_flat.mesh, prepared_flat.score_field, prepared_flat.case
        ),
        "b_flat_mesh_curved_field": runner(
            prepared_flat.mesh, prepared_curved.score_field, prepared_flat.case
        ),
        "c_curved_mesh_flat_field": runner(
            prepared_curved.mesh, prepared_flat.score_field, prepared_curved.case
        ),
        "c2_curved_zmesh_morph_off": runner(
            prepared_off.mesh, prepared_flat.score_field, prepared_off.case
        ),
    }
    properties = {name: _props(value) for name, value in arms.items()}
    reference = properties["a_flat_mesh_flat_field"]
    return {
        "kx": float(kx),
        "ky": float(ky),
        "backend": backend,
        **properties,
        "delta_pct": {
            name: _delta_pct(reference, values)
            for name, values in properties.items()
            if name != "a_flat_mesh_flat_field"
        },
        "max_node_displacement_mm": _displacement_mm(
            prepared_flat.mesh, prepared_curved.mesh
        ),
    }
