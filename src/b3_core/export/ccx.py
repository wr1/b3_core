"""CalculiX material cards from homogenized stiffness."""

from __future__ import annotations

import numpy as np

# CalculiX *ELASTIC,TYPE=ORTHO field order (ccx manual):
# D1111, D1122, D2222, D1133, D2233, D3333, D1212, D1313, D2323
# C is Voigt (xx, yy, zz, yz, xz, xy) as used by CoreModel.stiffness.


def ortho_dijkl(C: np.ndarray) -> tuple[float, ...]:
    """Nine CalculiX ORTHO constants from a 6×6 Voigt stiffness (Pa)."""
    C = np.asarray(C, dtype=float)
    if C.shape != (6, 6):
        msg = f"stiffness must be 6x6, got {C.shape}"
        raise ValueError(msg)
    return (
        float(C[0, 0]),  # D1111
        float(C[0, 1]),  # D1122
        float(C[1, 1]),  # D2222
        float(C[0, 2]),  # D1133
        float(C[1, 2]),  # D2233
        float(C[2, 2]),  # D3333
        float(C[5, 5]),  # D1212 = C_xyxy = Gxy
        float(C[4, 4]),  # D1313 = C_xzxz = Gxz
        float(C[3, 3]),  # D2323 = C_yzyz = Gyz
    )


def _fmt(v: float) -> str:
    return f"{float(v):.8g}"


def ccx_ortho_card(
    C: np.ndarray,
    *,
    name: str = "core_hom",
    rho: float | None = None,
    temperature: float | None = 293.0,
    provenance_comment: str | None = None,
) -> str:
    """CalculiX ``*elastic,type=ortho`` block from C_eff (Pa).

    Matches the Dijkl order used by ``b3_mat.calculix`` (swap yz/xy shear
    relative to this package's Voigt ``yz, xz, xy``). Optional ``*density``.
    """
    d1111, d1122, d2222, d1133, d2233, d3333, d1212, d1313, d2323 = ortho_dijkl(C)
    last = _fmt(d2323)
    if temperature is not None:
        last = f"{last},{_fmt(temperature)}"
    lines = [
        f"*material,name={name}",
        "*elastic,type=ortho",
        ",".join(
            _fmt(v) for v in (d1111, d1122, d2222, d1133, d2233, d3333, d1212, d1313)
        )
        + ",",
        last,
    ]
    if rho is not None:
        lines.extend(["*density", _fmt(rho)])
    text = "\n".join(lines) + "\n"
    if provenance_comment:
        text += provenance_comment
        if not provenance_comment.endswith("\n"):
            text += "\n"
    return text
