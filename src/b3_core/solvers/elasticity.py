"""Shared elasticity: load cases, stiffness, engineering constants.

Backends import this module and never each other. Property dicts use the
canonical names ``Ex, Ey, Ez, …``. The legacy ``Exx, Eyy, Ezz`` aliases are
included on the same dict through 0.3 so existing solver results keep their
keys; ``RunRecord.flat`` is the supported legacy writer.
"""

from __future__ import annotations

from typing import Any

import numpy as np

LOAD_CASES = ("xx", "yy", "zz", "yz", "xz", "xy")

# Canonical keys (b3_mat.OrthotropicMaterial names, plus reciprocal Poisson).
PROPERTY_KEYS = (
    "Ex",
    "Ey",
    "Ez",
    "Gxy",
    "Gxz",
    "Gyz",
    "nuxy",
    "nuxz",
    "nuyx",
    "nuyz",
    "nuzx",
    "nuzy",
)

# Present on solver property dicts in 0.3. Not part of the cache payload.
LEGACY_PROPERTY_KEYS = (
    "Exx",
    "Eyy",
    "Ezz",
    "Gxy",
    "Gxz",
    "Gyz",
    "nuxy",
    "nuxz",
    "nuyx",
    "nuyz",
    "nuzx",
    "nuzy",
)

_ORTHO_PAIRS = (
    ("Ex", "E1"),
    ("Ey", "E2"),
    ("Ez", "E3"),
    ("Gxy", "G12"),
    ("Gxz", "G13"),
    ("Gyz", "G23"),
    ("nuxy", "nu12"),
    ("nuxz", "nu13"),
    ("nuyz", "nu23"),
)


def properties_from_stiffness(stiffness):
    """Engineering constants and compliance from a 6×6 Voigt stiffness (Pa)."""
    compliance = np.linalg.inv(np.asarray(stiffness, dtype=float))
    ex = float(1.0 / compliance[0, 0])
    ey = float(1.0 / compliance[1, 1])
    ez = float(1.0 / compliance[2, 2])
    props = {
        "Ex": ex,
        "Ey": ey,
        "Ez": ez,
        "Gyz": float(1.0 / compliance[3, 3]),
        "Gxz": float(1.0 / compliance[4, 4]),
        "Gxy": float(1.0 / compliance[5, 5]),
        "nuxy": float(-compliance[1, 0] / compliance[0, 0]),
        "nuxz": float(-compliance[2, 0] / compliance[0, 0]),
        "nuyx": float(-compliance[0, 1] / compliance[1, 1]),
        "nuyz": float(-compliance[2, 1] / compliance[1, 1]),
        "nuzx": float(-compliance[0, 2] / compliance[2, 2]),
        "nuzy": float(-compliance[1, 2] / compliance[2, 2]),
    }
    props["Exx"] = ex
    props["Eyy"] = ey
    props["Ezz"] = ez
    return props, compliance


def isotropic_C(E: float, nu: float) -> np.ndarray:
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    C = np.zeros((6, 6))
    C[:3, :3] = lam
    C[0, 0] = C[1, 1] = C[2, 2] = lam + 2.0 * mu
    C[3, 3] = C[4, 4] = C[5, 5] = mu
    return C


def orthotropic_C(E1, E2, E3, G12, G13, G23, nu12, nu13, nu23) -> np.ndarray:
    """6×6 stiffness from orthotropic engineering constants (axes 1=x, 2=y, 3=z).

    Voigt order (xx, yy, zz, yz, xz, xy): shear diagonals are G23, G13, G12.
    """
    S = np.zeros((6, 6))
    S[0, 0], S[1, 1], S[2, 2] = 1.0 / E1, 1.0 / E2, 1.0 / E3
    S[0, 1] = S[1, 0] = -nu12 / E1
    S[0, 2] = S[2, 0] = -nu13 / E1
    S[1, 2] = S[2, 1] = -nu23 / E2
    S[3, 3], S[4, 4], S[5, 5] = 1.0 / G23, 1.0 / G13, 1.0 / G12
    return np.linalg.inv(S)


def constituent_dict(mat: Any) -> dict:
    """Material model or dict, with both canonical and legacy orthotropic keys."""
    if hasattr(mat, "model_dump"):
        raw = mat.model_dump()
    elif isinstance(mat, dict):
        raw = dict(mat)
    else:
        raw = dict(mat)
    out = dict(raw)
    for new, old in _ORTHO_PAIRS:
        if out.get(new) is None and out.get(old) is not None:
            out[new] = out[old]
        if out.get(old) is None and out.get(new) is not None:
            out[old] = out[new]
    return out


def material_C(mat: Any) -> np.ndarray:
    """6×6 stiffness from an isotropic (E, nu) or orthotropic material."""
    d = constituent_dict(mat)
    if d.get("E1") is not None:
        return orthotropic_C(
            d["E1"],
            d["E2"],
            d["E3"],
            d["G12"],
            d["G13"],
            d["G23"],
            d["nu12"],
            d["nu13"],
            d["nu23"],
        )
    return isotropic_C(d["E"], d["nu"])


def face_material_dict(face: Any) -> dict | None:
    """Face sheet as a material dict, or None when there is no sheet."""
    if face is None:
        return None
    if isinstance(face, dict):
        raw = dict(face)
    elif hasattr(face, "model_dump"):
        raw = face.model_dump()
        material = getattr(face, "material", None)
        if material is not None:
            raw.update(constituent_dict(material))
    else:
        raw = dict(face)
    if not raw:
        return None
    thickness = raw.get("thickness") or 0.0
    has_elastic = (
        raw.get("E") is not None
        or raw.get("E1") is not None
        or raw.get("Ex") is not None
    )
    if not has_elastic and float(thickness) <= 0.0:
        return None
    if not has_elastic:
        raw.setdefault("E", 12_000_000_000.0)
        raw.setdefault("nu", 0.3)
    return constituent_dict(raw)


def scoring_payload(scoring: Any) -> dict | None:
    if scoring is None:
        return None
    if isinstance(scoring, dict):
        return scoring or None
    if hasattr(scoring, "model_dump"):
        return scoring.model_dump()
    return dict(scoring)
