"""Post-solve diagnostics. The published stiffness stays the symmetrised matrix.

Each check is ``{id, status, value}`` with ``status`` one of ``ok``, ``warn``,
``fail``. Ties are a separate list, one row per high face. Thresholds are the
solver tolerances already used in this package (direct-factor roundoff, the
periodic face tolerance, and ``MIN_HW``). They are not a KD200 acceptance band.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from b3_core.solvers.elasticity import LOAD_CASES, face_material_dict, material_C
from b3_core.solvers.periodic import face_tol, z_face_ties

# ‖C−Cᵀ‖/‖C‖ before symmetrisation. Direct factors sit under 1e-8.
_ASYM_OK = 1e-6
_ASYM_WARN = 1e-3
# Relative Loewner violation of the two-phase Voigt/Reuss bounds.
_BOUND_OK = 1e-3
_BOUND_WARN = 2e-2
# max |w(x+L)−w(x)| in metres. A periodic fluctuation cancels below 1e-6 m.
_GAP_OK = 1e-6
_GAP_WARN = 1e-4
_WEIGHT_CUTOFF = 1e-8

_MACRO = {
    "xx": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    "yy": np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
    "zz": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    "yz": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0]]),
    "xz": np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
    "xy": np.array([[0.0, 0.5, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]),
}


def raw_asymmetry(raw: np.ndarray) -> dict[str, Any]:
    """‖C−Cᵀ‖/‖C‖ on the matrix from before ``0.5(C+Cᵀ)``."""
    matrix = np.asarray(raw, dtype=float)
    scale = float(np.linalg.norm(matrix))
    if scale == 0.0:
        value = 0.0
    else:
        value = float(np.linalg.norm(matrix - matrix.T) / scale)
    if value <= _ASYM_OK:
        status = "ok"
    elif value <= _ASYM_WARN:
        status = "warn"
    else:
        status = "fail"
    return {"id": "raw_asymmetry", "status": status, "value": value}


def positive_definite(stiffness: np.ndarray) -> dict[str, Any]:
    """Smallest eigenvalue of the published (symmetrised) stiffness."""
    matrix = np.asarray(stiffness, dtype=float)
    matrix = 0.5 * (matrix + matrix.T)
    value = float(np.linalg.eigvalsh(matrix).min())
    status = "ok" if value > 0.0 else "fail"
    return {"id": "positive_definite", "status": status, "value": value}


def voigt_reuss(
    stiffness: np.ndarray,
    core: Any,
    resin: Any,
    phi: float,
    face: Any = None,
) -> dict[str, Any]:
    """Loewner gap of ``C`` against the two-phase Voigt and Reuss matrices.

    A face sheet is a third constituent, so the two-phase bound does not apply
    and the value is null. ``phi`` is ``effective_resin_vf``.
    """
    if face_material_dict(face) is not None:
        return {"id": "voigt_reuss", "status": "ok", "value": None}
    fraction = float(phi)
    if not 0.0 <= fraction <= 1.0:
        return {"id": "voigt_reuss", "status": "fail", "value": None}
    core_c = material_C(core)
    resin_c = material_C(resin)
    voigt = (1.0 - fraction) * core_c + fraction * resin_c
    reuss = np.linalg.inv(
        (1.0 - fraction) * np.linalg.inv(core_c) + fraction * np.linalg.inv(resin_c)
    )
    matrix = np.asarray(stiffness, dtype=float)
    matrix = 0.5 * (matrix + matrix.T)
    scale = max(float(np.linalg.norm(voigt)), 1.0)
    upper = float(np.linalg.eigvalsh(voigt - matrix).min())
    lower = float(np.linalg.eigvalsh(matrix - reuss).min())
    value = max(0.0, -upper / scale, -lower / scale)
    if value <= _BOUND_OK:
        status = "ok"
    elif value <= _BOUND_WARN:
        status = "warn"
    else:
        status = "fail"
    return {"id": "voigt_reuss", "status": status, "value": float(value)}


def unmatched_nodes(ties: list[dict[str, Any]]) -> dict[str, Any]:
    """High-face nodes that are not slaves. Counted once per face."""
    missing = 0
    for row in ties:
        missing += max(int(row["top_nodes"]) - int(row["slaves"]), 0)
    status = "ok" if missing == 0 else "fail"
    return {"id": "unmatched_nodes", "status": status, "value": int(missing)}


def kerf_pinch(case: Any) -> dict[str, Any]:
    """Smallest unclamped kerf half-width. The mesh floors widths at ``MIN_HW``."""
    from b3_core.core.mesh import MIN_HW, hw_unclamped

    thickness = float(case.thickness)
    widths: list[float] = []
    for grooves, kappa in (
        (case.xgr, float(case.curvature.kx)),
        (case.ygr, float(case.curvature.ky)),
    ):
        for groove in grooves:
            depth = float(groove.signed_depth)
            hw0 = 0.5 * float(groove.width)
            slope = -float(np.sign(depth)) * kappa * float(groove.pitch) / 2.0
            z_mouth = 0.0 if groove.mouth == "bottom" else thickness
            z_root = depth if depth > 0.0 else thickness + depth
            for height in (z_mouth, z_root):
                widths.append(hw_unclamped(hw0, depth, slope, height, thickness))
    if not widths:
        return {"id": "kerf_pinch", "status": "ok", "value": None}
    worst = float(min(widths))
    if worst < 0.0:
        status = "fail"
    elif worst < MIN_HW:
        status = "warn"
    else:
        status = "ok"
    return {"id": "kerf_pinch", "status": status, "value": worst}


def _key(coords: np.ndarray, tol: float) -> tuple[int, ...]:
    scale = max(float(tol), 1e-15)
    rounded = np.round(np.asarray(coords, dtype=float) / scale)
    return tuple(int(value) for value in rounded)


def _unique_pairs(points: np.ndarray, axis: int) -> tuple[int, int, int, int]:
    """``(top_nodes, slaves, multi_node_rows, unmatched)`` for one axis."""
    points = np.asarray(points, dtype=float)
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    tol = face_tol(points)
    high = np.flatnonzero(np.abs(points[:, axis] - hi[axis]) < tol)
    low = np.flatnonzero(np.abs(points[:, axis] - lo[axis]) < tol)
    transverse = [index for index in range(3) if index != axis]
    buckets: dict[tuple[int, ...], list[int]] = {}
    for node in low:
        buckets.setdefault(_key(points[node, transverse], tol), []).append(int(node))
    slaves = 0
    multi = 0
    unmatched = 0
    for node in high:
        hits = buckets.get(_key(points[node, transverse], tol), [])
        if len(hits) == 1:
            slaves += 1
        elif len(hits) > 1:
            slaves += 1
            multi += 1
        else:
            unmatched += 1
    return int(len(high)), slaves, multi, unmatched


def _pair_row(points: np.ndarray, axis: int, name: str) -> dict[str, Any]:
    top, slaves, multi, unmatched = _unique_pairs(points, axis)
    identity = top > 0 and slaves == top and multi == 0 and unmatched == 0
    return {
        "face": name,
        "top_nodes": top,
        "slaves": slaves,
        "identity": identity,
        "multi_node_rows": multi,
        "ok": identity,
    }


def _z_row(points: np.ndarray) -> dict[str, Any]:
    try:
        ties = z_face_ties(points)
    except ValueError:
        return {
            "face": "z",
            "top_nodes": 0,
            "slaves": 0,
            "identity": False,
            "multi_node_rows": 0,
            "ok": False,
        }
    nonzero = np.count_nonzero(np.abs(ties.weights) > _WEIGHT_CUTOFF, axis=1)
    multi = int(np.count_nonzero(nonzero > 1))
    count = int(len(ties.slave))
    sums_ok = bool(np.allclose(ties.weights.sum(axis=1), 1.0, atol=1e-6))
    return {
        "face": "z",
        "top_nodes": count,
        "slaves": count,
        "identity": bool(ties.identity),
        "multi_node_rows": multi,
        "ok": count > 0 and sums_ok,
    }


def tie_report(points: np.ndarray) -> list[dict[str, Any]]:
    """Coordinate ties for x, y, and the bilinear z face. Numpy and MFEM use these."""
    points = np.asarray(points, dtype=float)
    return [
        _pair_row(points, 0, "x"),
        _pair_row(points, 1, "y"),
        _z_row(points),
    ]


def _fluctuation(points: np.ndarray, displacement: np.ndarray, case: str) -> np.ndarray:
    origin = points.min(axis=0)
    shifted = points - origin
    return np.asarray(displacement, dtype=float) - shifted @ _MACRO[case]


def _pair_gap(points: np.ndarray, fluctuation: np.ndarray, axis: int) -> float:
    points = np.asarray(points, dtype=float)
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    tol = face_tol(points)
    high = np.flatnonzero(np.abs(points[:, axis] - hi[axis]) < tol)
    low = np.flatnonzero(np.abs(points[:, axis] - lo[axis]) < tol)
    transverse = [index for index in range(3) if index != axis]
    buckets: dict[tuple[int, ...], list[int]] = {}
    for node in low:
        buckets.setdefault(_key(points[node, transverse], tol), []).append(int(node))
    worst = 0.0
    for node in high:
        hits = buckets.get(_key(points[node, transverse], tol), [])
        if len(hits) != 1:
            continue
        gap = float(np.max(np.abs(fluctuation[int(node)] - fluctuation[hits[0]])))
        worst = max(worst, gap)
    return worst


def periodic_mismatch(
    points: np.ndarray | None,
    displacements: dict[str, np.ndarray] | None,
) -> dict[str, Any]:
    """``max |w(x+L)−w(x)|`` when a displacement field was returned."""
    if points is None or not displacements:
        return {"id": "periodic_mismatch", "status": "ok", "value": None}
    coords = np.asarray(points, dtype=float)
    present = [name for name in LOAD_CASES if name in displacements]
    if not present:
        return {"id": "periodic_mismatch", "status": "ok", "value": None}
    for name in present:
        field = np.asarray(displacements[name], dtype=float)
        if field.shape != coords.shape:
            return {"id": "periodic_mismatch", "status": "warn", "value": None}
    try:
        z_ties = z_face_ties(coords)
    except ValueError:
        return {"id": "periodic_mismatch", "status": "warn", "value": None}
    worst = 0.0
    for name in present:
        fluctuation = _fluctuation(coords, displacements[name], name)
        predicted = np.einsum("nk,nkd->nd", z_ties.weights, fluctuation[z_ties.masters])
        z_gap = float(np.max(np.abs(fluctuation[z_ties.slave] - predicted)))
        x_gap = _pair_gap(coords, fluctuation, 0)
        y_gap = _pair_gap(coords, fluctuation, 1)
        worst = max(worst, z_gap, x_gap, y_gap)
    if worst <= _GAP_OK:
        status = "ok"
    elif worst <= _GAP_WARN:
        status = "warn"
    else:
        status = "fail"
    return {"id": "periodic_mismatch", "status": status, "value": float(worst)}


def build_diagnostics(case: Any, prep: Any, solved: Any) -> dict[str, Any]:
    """Checks and ties for one solve. Cache stores this dict on the record."""
    mesh_points = np.asarray(prep.mesh.points, dtype=float)
    backend_ties = getattr(solved, "ties", None)
    if backend_ties:
        ties = [dict(row) for row in backend_ties]
    else:
        ties = tie_report(mesh_points)
    raw = getattr(solved, "raw_stiffness", None)
    if raw is None:
        raw = solved.stiffness
    points = getattr(solved, "points", None)
    displacements = getattr(solved, "displacements", None)
    checks = [
        raw_asymmetry(raw),
        positive_definite(solved.stiffness),
        voigt_reuss(
            solved.stiffness,
            case.core,
            case.resin,
            prep.geometry.effective_resin_vf,
            face=case.face,
        ),
        unmatched_nodes(ties),
        periodic_mismatch(points, displacements),
        kerf_pinch(case),
    ]
    return {"checks": checks, "ties": ties}


def strict_failures(diagnostics: dict[str, Any] | None) -> list[str]:
    """Names of warn/fail checks and ties that are not ok. Empty when clean."""
    block = diagnostics or {}
    names: list[str] = []
    for row in block.get("checks") or []:
        if row.get("status") in ("warn", "fail"):
            names.append(f"{row.get('id')}={row.get('status')}")
    for row in block.get("ties") or []:
        if row.get("ok") is False:
            names.append(f"tie:{row.get('face')}=fail")
    return names
