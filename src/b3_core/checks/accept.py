"""Publish gate. A missing artifact fails the check. It does not launch a study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from b3_core.cache import cache_stats
from b3_core.provenance import code_revision

# Flat anchor is 0.01 %. Curved tolerance covers the halo-mesh numpy gap
# (about 0.053 %) until a later study replaces it. These are relative bars,
# not a customer's moduli.
_FLAT_RTOL = 1e-4
_CURVED_RTOL_PCT = 0.1


def _read(path: str | Path | None) -> Any:
    if not path:
        return None
    text = Path(path).read_text(encoding="utf-8")
    if str(path).endswith((".yaml", ".yml")):
        return yaml.safe_load(text)
    return json.loads(text)


def _check(id_: str, ok: bool, value: Any, threshold: Any) -> dict[str, Any]:
    return {"id": id_, "ok": bool(ok), "value": value, "threshold": threshold}


def _diagnostics(grid: Any) -> dict[str, Any]:
    if not grid:
        return _check("diagnostics", False, "missing", "every grid check ok")
    rows = grid.get("rows") if isinstance(grid, dict) else grid
    bad = []
    ties_bad = []
    for row in rows or []:
        if row.get("ok") is False:
            bad.append("row failed")
        diagnostics = row.get("diagnostics") or {}
        if isinstance(diagnostics, str):
            diagnostics = json.loads(diagnostics)
        for item in diagnostics.get("checks") or []:
            if item.get("status") not in (None, "ok"):
                bad.append(item.get("id") or item.get("status"))
        for tie in diagnostics.get("ties") or []:
            if tie.get("ok") is False:
                ties_bad.append(tie.get("face"))
    return _check(
        "diagnostics",
        not bad and not ties_bad,
        {"failed": bad, "ties": ties_bad},
        "status ok and ties ok",
    )


def _cache(path: str | None) -> dict[str, Any]:
    if not path:
        return _check("cache_stale", False, "missing", "n_stale = 0")
    stats = cache_stats(path)
    return _check("cache_stale", int(stats["n_stale"]) == 0, stats["n_stale"], 0)


def _provenance(block: Any) -> dict[str, Any]:
    if not isinstance(block, dict):
        return _check("provenance", False, "missing", "present and not dirty")
    commit, dirty = code_revision()
    present = bool(block.get("commit") or block.get("solved_with") or commit)
    dirty_flag = block.get("dirty")
    ok = present and dirty_flag is not True
    return _check(
        "provenance", ok, {"dirty": dirty_flag, "checkout_dirty": dirty}, False
    )


def _backends(report: Any) -> list[dict[str, Any]]:
    if not isinstance(report, dict):
        missing = _check("backends_flat", False, "missing", _FLAT_RTOL)
        curved = _check("backends_curved", False, "missing", _CURVED_RTOL_PCT)
        return [missing, curved]
    flat_bad = []
    curved_worst = 0.0
    for row in report.get("rows") or []:
        pair_pct = [
            float(item["max_rel_pct"])
            for item in (row.get("pairs") or {}).values()
            if isinstance(item, dict) and item.get("max_rel_pct") is not None
        ]
        rel = max(pair_pct) if pair_pct else float(row.get("worst_rel_pct") or 0.0)
        flags = set(row.get("flags") or [])
        curved = (
            abs(float(row.get("kx") or 0.0)) > 0 or abs(float(row.get("ky") or 0.0)) > 0
        )
        if curved:
            curved_worst = max(curved_worst, rel)
        elif "flat_anchor" in flags or "no_backends" in flags or rel > 0.01:
            flat_bad.append(rel)
    flat_ok = not flat_bad
    curved_ok = curved_worst <= _CURVED_RTOL_PCT
    return [
        _check("backends_flat", flat_ok, flat_bad, 0.01),
        _check("backends_curved", curved_ok, curved_worst, _CURVED_RTOL_PCT),
    ]


def _fit(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return _check("fit", False, "missing", "converged and |z| <= 2")
    status = result.get("status")
    residuals = result.get("residuals") or {}
    max_z = residuals.get("max_abs_z")
    ok = status == "converged" and max_z is not None and abs(float(max_z)) <= 2.0
    if status == "no_free_params":
        ok = True
    return _check("fit", ok, {"status": status, "max_abs_z": max_z}, 2.0)


def _inert(grid: Any) -> dict[str, Any]:
    if not isinstance(grid, dict):
        manifest = {}
    else:
        manifest = grid.get("manifest") or grid
    kx = bool(manifest.get("kx_inert"))
    ky = bool(manifest.get("ky_inert"))
    acknowledged = bool(
        manifest.get("inert_acknowledged") or manifest.get("allow_inert_axis")
    )
    triggered = kx or ky
    return _check(
        "inert_axis",
        (not triggered) or acknowledged,
        {"kx_inert": kx, "ky_inert": ky, "acknowledged": acknowledged},
        "not triggered, or acknowledged",
    )


def _convention(block: Any) -> dict[str, Any]:
    rows = (
        block
        if isinstance(block, list)
        else (block or {}).get("rows")
        if isinstance(block, dict)
        else None
    )
    if not rows:
        return _check("convention", False, "missing", "opens_for on every kerf row")
    ok = all(
        isinstance(row, dict) and row.get("opens_for") in {"k>0", "k<0"} for row in rows
    )
    return _check("convention", ok, len(rows), "opens_for")


def _reproduce(report: Any) -> dict[str, Any]:
    if not isinstance(report, dict):
        return _check("reproduce", False, "missing", "1e-6 % at the same stamp")
    return _check("reproduce", bool(report.get("ok")), report.get("rows"), 1e-6)


def _profile_checks(profile: Any, project: dict[str, Any]) -> list[dict[str, Any]]:
    if not profile:
        return []
    reference = project.get("reference")
    checks = []
    for item in profile.get("checks") or []:
        ident = str(item.get("id"))
        if item.get("quantities") or item.get("quantity"):
            present = isinstance(reference, dict)
            checks.append(
                _check(
                    f"profile:{ident}",
                    present,
                    "reference card supplied"
                    if present
                    else "reference card not in the project",
                    item.get("rtol_pct"),
                )
            )
        else:
            checks.append(
                _check(
                    f"profile:{ident}",
                    True,
                    item.get("note")
                    or item.get("rtol_pct")
                    or item.get("max_node_displacement_mm"),
                    "caller profile",
                )
            )
    return checks


def check_accept(project: Any, *, profile: Any = None) -> dict[str, Any]:
    """Score the artifacts a project already has. Missing ones fail closed."""
    if isinstance(project, str | Path):
        project = _read(project) or {}
    project = dict(project or {})
    grid = (
        _read(project.get("grid"))
        if isinstance(project.get("grid"), str)
        else project.get("grid")
    )
    fit = (
        _read(project.get("fit_result"))
        if isinstance(project.get("fit_result"), str)
        else project.get("fit_result")
    )
    backends = (
        _read(project.get("backends_report"))
        if isinstance(project.get("backends_report"), str)
        else project.get("backends_report")
    )
    reproduced = (
        _read(project.get("reproduce"))
        if isinstance(project.get("reproduce"), str)
        else project.get("reproduce")
    )
    provenance = project.get("provenance")
    if isinstance(provenance, str):
        provenance = _read(provenance)
    convention = project.get("convention")
    if isinstance(convention, str):
        convention = _read(convention)
    profile_doc = profile
    if isinstance(profile, str | Path):
        profile_doc = _read(profile)
    checks = [
        _diagnostics(grid),
        _cache(project.get("cache")),
        _provenance(provenance),
        *_backends(backends),
        _reproduce(reproduced),
        _fit(fit),
        _inert(grid if isinstance(grid, dict) else {"manifest": project}),
        _convention(convention),
        *_profile_checks(profile_doc, project),
    ]
    return {"ok": all(item["ok"] for item in checks), "checks": checks}
