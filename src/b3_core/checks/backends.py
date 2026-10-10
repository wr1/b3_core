"""Solve one case on several backends and compare the stiffness.

Flat points fail when a pair differs by more than ``rtol`` (default 0.01 %).
Curved points are reported and flagged when they exceed that same bar. They
do not fail the command: the growth is the Q1 envelope, not a hard gate.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from b3_core.loaders import normalize_case

PROP_KEYS = ("Ex", "Ey", "Ez", "Gxy", "Gxz", "Gyz", "nuxy", "nuxz", "nuyz")
FLAT_ANCHOR_RTOL = 1e-4  # 0.01 %


def parse_points(text: str) -> list[tuple[float, float]]:
    """``"0,0 5e-5,0 1e-3,1e-3"`` → curvature pairs in 1/mm."""
    rows: list[tuple[float, float]] = []
    for token in text.split():
        if "," not in token:
            raise ValueError(f"point {token!r} must be kx,ky (for example 5e-5,0)")
        kx_text, ky_text = token.split(",", 1)
        rows.append((float(kx_text), float(ky_text)))
    if not rows:
        raise ValueError("points is empty")
    return rows


def parse_axis(text: str) -> list[float]:
    """One value, or ``start:stop:count`` inclusive."""
    text = text.strip()
    if not text:
        return [0.0]
    if ":" not in text:
        return [float(text)]
    start_text, stop_text, count_text = text.split(":")
    count = int(count_text)
    if count < 1:
        raise ValueError(f"axis count must be >= 1, got {count}")
    return [
        float(value)
        for value in np.linspace(float(start_text), float(stop_text), count)
    ]


def refine_madd(madd: list[float], level: int) -> list[float]:
    """Insert midpoints ``level`` times. One value stays one value."""
    values = sorted({float(item) for item in madd}) or [0.0]
    for _ in range(max(0, int(level))):
        if len(values) < 2:
            break
        nxt = [values[0]]
        for left, right in zip(values, values[1:]):
            nxt.append(0.5 * (left + right))
            nxt.append(right)
        values = nxt
    return values


def _max_rel(left: Any, right: Any) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    scale = max(float(np.abs(a).max()), float(np.abs(b).max()), 1.0)
    return float(np.abs(a - b).max() / scale)


def _case_with_madd(case: Any, madd: list[float]) -> Any:
    from b3_core.cases import CoreCase

    case_in, _workdir = normalize_case(case)
    updated = case_in.model_copy(update={"madd": [float(value) for value in madd]})
    if isinstance(case, CoreCase):
        return case.__class__(input=updated, workdir=case.workdir)
    return updated


def _policy_reason(case_in: Any, backend: str, kx: float, ky: float) -> str | None:
    """Limits of a backend. Install state is separate, so tests can inject a solve."""
    from b3_core.core.scoring import halo_reach

    curved = kx != 0.0 or ky != 0.0
    if backend == "ccx":
        if curved:
            return "ccx supports node-pair periodicity only"
        if case_in.is_orthotropic:
            return "ccx is isotropic two-phase"
        if halo_reach(case_in) > 0.0:
            return "ccx does not grade a resin halo"
    return None


def _install_reason(backend: str) -> str | None:
    from b3_core.solvers import UnknownBackendError, get_backend

    try:
        available = get_backend(backend).is_available()
    except UnknownBackendError as exc:
        return str(exc)
    if not available:
        return f"{backend} is not installed"
    return None


def _block_from_record(record: Any, *, pair: bool) -> dict[str, Any]:
    props = record.result.properties
    provenance = record.provenance
    return {
        "skipped": None,
        "allow_pair_periodicity": pair,
        "stamp": record.solver_stamp,
        "properties": {key: float(props[key]) for key in PROP_KEYS if key in props},
        "diagnostics": {
            "checks": list(record.diagnostics.get("checks", [])),
            "ties": list(record.diagnostics.get("ties", [])),
        },
        "provenance": (
            provenance.model_dump(mode="json") if provenance is not None else None
        ),
    }


def _skipped(reason: str) -> dict[str, Any]:
    return {
        "skipped": reason,
        "allow_pair_periodicity": False,
        "stamp": None,
        "properties": None,
        "diagnostics": None,
        "provenance": None,
    }


def _default_solve(
    case: Any, backend: str, kx: float, ky: float, pair: bool, cache: Any
):
    from b3_core.api import run_case

    return run_case(
        case,
        backend=backend,
        kx=kx,
        ky=ky,
        allow_pair_periodicity=True if pair else None,
        cache=cache,
    )


def check_backends(
    case: Any,
    *,
    backends: list[str],
    points: list[tuple[float, float]],
    refine: int = 0,
    rtol: float = FLAT_ANCHOR_RTOL,
    cache: Any = None,
    solve: Callable[..., Any] | None = None,
    check_installed: bool = True,
) -> dict[str, Any]:
    """One row per (point, refinement). ``ok`` is the flat-anchor gate."""
    started = time.perf_counter()
    solver = solve or _default_solve
    case_in, _workdir = normalize_case(case)
    base_madd = [float(value) for value in case_in.madd]
    levels = [refine_madd(base_madd, level) for level in range(int(refine) + 1)]
    rows: list[dict[str, Any]] = []
    worst = 0.0
    ok = True
    for kx, ky in points:
        curved = kx != 0.0 or ky != 0.0
        for level, madd in enumerate(levels):
            prepared = _case_with_madd(case, madd)
            prepared_in, _wd = normalize_case(prepared)
            blocks: dict[str, Any] = {}
            solved: dict[str, Any] = {}
            for name in backends:
                reason = _policy_reason(prepared_in, name, kx, ky)
                if reason is None and check_installed:
                    reason = _install_reason(name)
                if reason is not None:
                    blocks[name] = _skipped(reason)
                    continue
                pair = name == "mfem" and curved
                try:
                    record = solver(prepared, name, kx, ky, pair, cache)
                except Exception as exc:
                    blocks[name] = _skipped(f"{type(exc).__name__}: {exc}")
                    if not curved:
                        ok = False
                    continue
                blocks[name] = _block_from_record(record, pair=pair)
                solved[name] = record
            pairs: dict[str, Any] = {}
            names = list(solved)
            for i, left_name in enumerate(names):
                for right_name in names[i + 1 :]:
                    rel = _max_rel(
                        solved[left_name].result.stiffness,
                        solved[right_name].result.stiffness,
                    )
                    key = "_vs_".join(sorted((left_name, right_name)))
                    pairs[key] = {"max_rel": rel, "max_rel_pct": rel * 100.0}
                    worst = max(worst, rel * 100.0)
                    if rel > rtol and not curved:
                        ok = False
            flags: list[str] = []
            if not solved:
                flags.append("no_backends")
                if not curved:
                    ok = False
            if not curved and any(item["max_rel"] > rtol for item in pairs.values()):
                flags.append("flat_anchor")
            if curved and any(item["max_rel"] > rtol for item in pairs.values()):
                flags.append("curved_growth")
            rows.append(
                {
                    "kx": float(kx),
                    "ky": float(ky),
                    "refine": level,
                    "madd": madd,
                    "backends": blocks,
                    "pairs": pairs,
                    "flags": flags,
                }
            )
    return {
        "ok": ok,
        "rtol": float(rtol),
        "worst_rel_pct": worst,
        "wall_s": time.perf_counter() - started,
        "rows": rows,
    }


def check_cross(case: Any, **kwargs: Any) -> dict[str, Any]:
    """Flat anchor only. Ignores any curvature grid the caller had in mind."""
    kwargs = dict(kwargs)
    kwargs["points"] = [(0.0, 0.0)]
    return check_backends(case, **kwargs)


def format_table(report: dict[str, Any]) -> str:
    """One line per row for a terminal, plus the flat-anchor verdict."""
    lines = []
    for row in report["rows"]:
        pair_text = " ".join(
            f"{name}={item['max_rel_pct']:.4g}%" for name, item in row["pairs"].items()
        )
        skipped = [
            f"{name} skipped ({block['skipped']})"
            for name, block in row["backends"].items()
            if block.get("skipped")
        ]
        flag = ",".join(row["flags"]) or "ok"
        lines.append(
            f"kx={row['kx']:g} ky={row['ky']:g} refine={row['refine']} "
            f"{pair_text} [{flag}]"
        )
        lines.extend(f"  {item}" for item in skipped)
    verdict = "PASS" if report["ok"] else "FAIL"
    lines.append(
        f"{verdict} flat anchor rtol={report['rtol']:g} "
        f"worst_rel_pct={report['worst_rel_pct']:.4g} "
        f"wall_s={report['wall_s']:.2f}"
    )
    return "\n".join(lines) + "\n"
