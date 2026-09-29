"""Compare two engineering-constant dicts (ccx reference vs another solver)."""

from __future__ import annotations

import numpy as np

from b3_core.solvers.elasticity import LEGACY_PROPERTY_KEYS, PROPERTY_KEYS


def validate_against(
    reference: dict,
    candidate: dict,
    *,
    label: str = "candidate",
    rtol: float = 0.05,
    atol: float = 0.0,
) -> dict:
    """Relative check on every property key present in both dicts."""
    comparison = {}
    passed = True
    keys = list(dict.fromkeys([*PROPERTY_KEYS, *LEGACY_PROPERTY_KEYS]))
    for key in keys:
        if key not in reference or key not in candidate:
            continue
        ref = float(reference[key])
        other = float(candidate[key])
        abs_error = abs(other - ref)
        rel_error = abs_error / max(abs(ref), atol, np.finfo(float).eps)
        ok = bool(abs_error <= atol or rel_error <= rtol)
        comparison[key] = {
            "reference": ref,
            label: other,
            "abs_error": float(abs_error),
            "rel_error": float(rel_error),
            "ok": ok,
        }
        passed = passed and ok
    return {"passed": passed, "rtol": rtol, "atol": atol, "properties": comparison}


def validate_against_ccx(
    ccx_output, other_output, *, label="fenicsx", rtol=0.05, atol=0.0
):
    """0.3 name kept for callers that still say ``ccx`` for the reference side."""
    out = validate_against(ccx_output, other_output, label=label, rtol=rtol, atol=atol)
    # Historic shape stored the reference value under "ccx".
    for row in out["properties"].values():
        row["ccx"] = row["reference"]
    return out
