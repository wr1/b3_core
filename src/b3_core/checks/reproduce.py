"""Re-solve stored points. A stamp change is drift, not a pass."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from b3_core.models import CaseInput, Curvature
from b3_core.solvers.stamps import stamp_for

_KEYS = ("Ex", "Ey", "Ez", "Gxy", "Gxz", "Gyz")


def _load(values: Any) -> list[dict[str, Any]]:
    if isinstance(values, str | Path):
        payload = json.loads(Path(values).read_text(encoding="utf-8"))
    else:
        payload = values
    if isinstance(payload, dict):
        payload = payload.get("points") or payload.get("rows") or []
    return list(payload)


def _rel_pct(stored: dict[str, Any], fresh: dict[str, Any]) -> float:
    left = []
    right = []
    for key in _KEYS:
        if key in stored and key in fresh and stored[key] is not None:
            left.append(float(stored[key]))
            right.append(float(fresh[key]))
    if not left:
        return 0.0
    a = np.asarray(left)
    b = np.asarray(right)
    scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1.0)
    return float(np.max(np.abs(a - b)) / scale * 100.0)


def _props(record: Any) -> dict[str, float]:
    if isinstance(record, dict):
        source = record
    else:
        source = dict(getattr(getattr(record, "result", None), "properties", {}) or {})
    return {key: float(source[key]) for key in _KEYS if key in source}


def check_reproduce(
    values: Any,
    *,
    cache: Any = None,
    solve: Any = None,
    rtol_pct: float = 1e-6,
) -> dict[str, Any]:
    """Pass when the fresh solve matches under ``rtol_pct`` at the same stamp."""
    points = _load(values)
    rows = []
    ok = True
    for point in points:
        case = point.get("case")
        if isinstance(case, str | Path):
            case_in = CaseInput.model_validate(
                json.loads(Path(case).read_text(encoding="utf-8"))
            )
        elif isinstance(case, dict):
            case_in = CaseInput.model_validate(case)
        else:
            case_in = case
        kx = float(point.get("kx") or 0.0)
        ky = float(point.get("ky") or 0.0)
        case_in = case_in.model_copy(update={"curvature": Curvature(kx=kx, ky=ky)})
        backend = str(point.get("backend") or "fenicsx")
        stored_stamp = point.get("solver_stamp")
        current = stamp_for(backend)
        drift = stored_stamp is not None and stored_stamp != current
        runner = solve
        if runner is None:
            from b3_core.api import run_case

            runner = run_case
        fresh = _props(runner(case_in, backend=backend, cache=cache))
        rel = _rel_pct(point.get("properties") or {}, fresh)
        passed = (not drift) and rel <= float(rtol_pct)
        ok = ok and passed
        rows.append(
            {
                "kx": kx,
                "ky": ky,
                "backend": backend,
                "solver_stamp": current,
                "stored_stamp": stored_stamp,
                "drift": drift,
                "rel_pct": rel,
                "ok": passed,
            }
        )
    return {"ok": ok, "rtol_pct": float(rtol_pct), "rows": rows}
