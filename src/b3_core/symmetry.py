"""x/y interchange symmetry of a case: detected, never assumed.

Exx = Eyy, Gxz = Gyz and nuxz = nuyz hold only when the whole input is
invariant under swapping x and y: same pitch, same groove rows, and every
material (core, resin, face) with Ex = Ey, Gxz = Gyz and nuxz = nuyz. Nothing
in the package may set one of these pairs from the other; this module only
*detects* the symmetry so it can be used as a check (flat solve) or as an
explicitly requested optimisation (mirrored sweep, averaged fit targets).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: (x quantity, y quantity) pairs that a symmetric material must match.
MATERIAL_PAIRS = (("Ex", "Ey"), ("Gxz", "Gyz"), ("nuxz", "nuyz"))
#: (x property, y property) pairs of the homogenised result.
RESULT_PAIRS = (("Exx", "Eyy"), ("Gxz", "Gyz"), ("nuxz", "nuyz"))


def _get(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _dump(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj


def _rows(rows: Any) -> list[list[float]]:
    out = []
    for r in rows or []:
        cut = r.as_cut() if hasattr(r, "as_cut") else [r[i] for i in range(4)]
        out.append([float(v) for v in cut])
    return sorted(out)


def _close(a: float, b: float, rtol: float) -> bool:
    return abs(a - b) <= rtol * max(abs(a), abs(b), 1e-300)


def material_asymmetry(mat: Any, rtol: float = 1e-12) -> list[str]:
    """Pairs in which ``mat`` differs between x and y (empty: symmetric).

    An isotropic material (``E``, ``nu``) is symmetric. An orthotropic one is
    symmetric when Ex = Ey, Gxz = Gyz and nuxz = nuyz.
    """
    mat = _dump(mat)
    if mat is None or _get(mat, "Ex") is None:
        return []
    bad = []
    for a, b in MATERIAL_PAIRS:
        va, vb = _get(mat, a), _get(mat, b)
        if (va is None) != (vb is None) or (
            va is not None and not _close(float(va), float(vb), rtol)
        ):
            bad.append(f"{a}≠{b}")
    return bad


def xy_asymmetry(case: Any, rtol: float = 1e-12) -> list[str]:
    """Every reason the case is not invariant under x ↔ y (empty: symmetric).

    Covers geometry (dx/dy, groove rows) and materials (core, resin, face
    material). Scoring/halo settings are per surface (saw cut, face), not per
    axis, so they cannot break the symmetry; a per-axis key is reported if one
    ever appears.
    """
    reasons: list[str] = []
    if not _close(float(_get(case, "dx")), float(_get(case, "dy")), rtol):
        reasons.append("dx≠dy")
    if _rows(_get(case, "xgr")) != _rows(_get(case, "ygr")):
        reasons.append("x and y groove rows differ")
    for part in ("core", "resin"):
        reasons += [f"{part} {r}" for r in material_asymmetry(_get(case, part), rtol)]
    face = _dump(_get(case, "face"))
    if face:
        reasons += [
            f"face {r}" for r in material_asymmetry(_get(face, "material"), rtol)
        ]
    scoring = _dump(_get(case, "scoring"))
    if isinstance(scoring, Mapping):
        per_axis = [k for k in scoring if k.endswith(("_x", "_y"))]
        if per_axis:
            reasons.append(f"per-axis scoring settings {sorted(per_axis)}")
    return reasons


def is_xy_symmetric(case: Any, rtol: float = 1e-12) -> bool:
    return not xy_asymmetry(case, rtol)


def symmetry_check(
    case: Any, props: Mapping[str, float], tol_pct: float = 1e-6
) -> dict[str, Any]:
    """Diagnostic for a flat solve: do the x/y result pairs agree as expected?

    For a symmetric input the pairs must match (``ok`` False flags a solver or
    mesh problem). For an asymmetric input they are free to differ and nothing
    is checked; the differences are only reported.
    """
    symmetric = is_xy_symmetric(case)
    diff = {
        f"{a}/{b}": 100
        * abs(float(props[a]) - float(props[b]))
        / max(abs(float(props[a])), abs(float(props[b])), 1e-300)
        for a, b in RESULT_PAIRS
        if a in props and b in props
    }
    worst = max(diff.values(), default=0.0)
    if symmetric:
        ok = worst <= tol_pct
        msg = (
            f"x/y pairs differ by ≤ {worst:.2g} %, consistent with symmetric input"
            if ok
            else f"x/y pairs differ by {worst:.3g} % although the input is "
            "symmetric: check the solve"
        )
    else:
        ok = True
        msg = (
            "input is not x/y symmetric ("
            + "; ".join(xy_asymmetry(case))
            + "); pairs differ: "
            + ", ".join(f"{k} {v:.3g} %" for k, v in diff.items())
        )
    return {"symmetric_input": symmetric, "ok": ok, "diff_pct": diff, "message": msg}
