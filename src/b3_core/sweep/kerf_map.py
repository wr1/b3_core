"""3×3 kerf state map. Geometry at every point, solves at the four corners."""

from __future__ import annotations

import time
from typing import Any

from b3_core.loaders import normalize_case
from b3_core.models import Curvature
from b3_core.pipeline import prepare
from b3_core.units import curvature_tick, radius_m


def _state(row: dict[str, Any]) -> str:
    mouth = float(row["hw_mouth_mm"])
    root = float(row["hw_root_mm"])
    if abs(mouth - root) <= 1e-6:
        return "flat"
    if mouth > root:
        return "open"
    return "closed"


def _properties(record: Any) -> dict[str, float]:
    result = getattr(record, "result", None)
    source = dict(getattr(result, "properties", {}) or {})
    if isinstance(record, dict):
        source.update(record)
    out = {}
    for key in ("Ex", "Ey", "Ez", "Gxy", "Gxz", "Gyz"):
        if key in source and isinstance(source[key], (int, float)):
            out[key] = float(source[key])
    return out


def kerf_map(
    case: Any,
    magnitude: float,
    *,
    solve: Any = None,
    backend: str = "fenicsx",
    cache: Any = None,
) -> dict[str, Any]:
    """Sign grid at ±magnitude and 0. Corners are solved; the rest are geometry."""
    started = time.perf_counter()
    case_in, _workdir = normalize_case(case)
    level = abs(float(magnitude))
    samples = (-level, 0.0, level)
    corners = {(-level, -level), (-level, level), (level, -level), (level, level)}
    flat_prep = prepare(
        case_in.model_copy(update={"curvature": Curvature(kx=0.0, ky=0.0)})
    )
    grid = []
    for ky in samples:
        for kx in samples:
            point = case_in.model_copy(
                update={"curvature": Curvature(kx=float(kx), ky=float(ky))}
            )
            prepared = prepare(point)
            kerfs = []
            for row in prepared.geometry.kerfs:
                item = dict(row)
                item["state"] = _state(item)
                kerfs.append(item)
            cell: dict[str, Any] = {
                "kx": float(kx),
                "ky": float(ky),
                "kx_tick": curvature_tick(kx),
                "ky_tick": curvature_tick(ky),
                "kerfs": kerfs,
                "resin_vf": float(prepared.geometry.resin_vf),
                "rho_infused": float(prepared.geometry.rho_infused),
                "solved": False,
            }
            if (kx, ky) in corners and level > 0.0:
                runner = solve
                if runner is None:
                    from b3_core.api import run_case

                    runner = run_case
                record = runner(point, backend=backend, cache=cache)
                cell.update(_properties(record))
                cell["solved"] = True
            grid.append(cell)
    radius = radius_m(level)
    return {
        "magnitude_1_per_mm": level,
        "radius_m": radius,
        "convention": [
            {
                "axis": row["axis"],
                "mouth": row["mouth"],
                "opens_for": row["opens_for"],
            }
            for row in flat_prep.geometry.kerfs
        ],
        "flat": {
            "resin_vf": float(flat_prep.geometry.resin_vf),
            "rho_infused": float(flat_prep.geometry.rho_infused),
        },
        "grid": grid,
        "wall_seconds": time.perf_counter() - started,
    }
