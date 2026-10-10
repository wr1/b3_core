"""Two madd refinements. A one-station madd cannot be refined."""

from __future__ import annotations

from typing import Any

import numpy as np

from b3_core.checks.backends import refine_madd
from b3_core.loaders import normalize_case

_KEYS = ("Ex", "Ey", "Ez", "Gxy", "Gxz", "Gyz")


def _props(record: Any) -> dict[str, float]:
    if isinstance(record, dict):
        source = record
    else:
        source = dict(getattr(getattr(record, "result", None), "properties", {}) or {})
    return {
        key: float(source[key])
        for key in _KEYS
        if key in source and source[key] is not None
    }


def check_convergence(
    case: Any,
    *,
    levels: tuple[int, int] = (0, 1),
    backend: str = "numpy",
    solve: Any = None,
    rtol_pct: float = 1.0,
) -> dict[str, Any]:
    """Relative change between two madd refinements of the same case."""
    case_in, _workdir = normalize_case(case)
    base = [float(value) for value in case_in.madd]
    refined = [refine_madd(base, level) for level in levels]
    if refined[0] == refined[1]:
        return {
            "ok": False,
            "value": "madd has fewer than two stations, so refine does not change it",
            "threshold": rtol_pct,
            "levels": [
                {"level": int(level), "madd": values}
                for level, values in zip(levels, refined, strict=True)
            ],
        }
    runner = solve
    if runner is None:
        from b3_core.api import run_case

        runner = run_case
    rows = []
    for level, madd in zip(levels, refined, strict=True):
        point = case_in.model_copy(update={"madd": madd})
        rows.append(
            {
                "level": int(level),
                "n_madd": len(madd),
                "properties": _props(runner(point, backend=backend, cache=None)),
            }
        )
    left = rows[0]["properties"]
    right = rows[-1]["properties"]
    keys = [key for key in _KEYS if key in left and key in right]
    if not keys:
        rel = None
        ok = False
    else:
        a = np.array([left[key] for key in keys])
        b = np.array([right[key] for key in keys])
        scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1.0)
        rel = float(np.max(np.abs(a - b)) / scale * 100.0)
        ok = rel <= float(rtol_pct)
    return {
        "ok": ok,
        "rel_pct": rel,
        "threshold_pct": float(rtol_pct),
        "rows": rows,
    }
