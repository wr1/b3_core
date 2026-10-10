"""Neat-basis split, foam scaling, and a typical resin."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

from b3_core.fit.spec import Target, TargetSet
from b3_core.models import Material


def load_foam_scaling() -> dict[str, Any]:
    text = (
        files("b3_core.data").joinpath("foam_scaling.json").read_text(encoding="utf-8")
    )
    return json.loads(text)


def scaling_family(family: str) -> dict[str, Any]:
    table = load_foam_scaling()
    if family not in table:
        known = ", ".join(sorted(table))
        raise ValueError(f"unknown foam family {family!r}; known: {known}")
    return table[family]


def estimate_foam(rho: float, *, family: str = "PVC") -> Material:
    """Gibson–Ashby estimate. ``reference`` carries the ± band."""
    row = scaling_family(family)
    density = float(rho)
    relative = density / float(row["solid_rho"])
    modulus = float(row["solid_E_Pa"]) * relative ** float(row["E_exponent"])
    band = float(row["band"])
    return Material(
        E=modulus,
        nu=float(row["nu"]),
        rho=density,
        source="estimated",
        reference=f"{row['reference']} ±{band:.0%} at rho={density:g}",
    )


def resin_typical(kind: str = "epoxy") -> Material:
    """A typical neat resin. Only ``epoxy`` is tabulated."""
    if kind != "epoxy":
        raise ValueError(f"unknown resin {kind!r}; use 'epoxy'")
    return Material(
        E=3.5e9,
        nu=0.35,
        rho=1100.0,
        source="neat",
        reference="typical epoxy",
    )


def _material_from_neat(fields: dict[str, float]) -> Material:
    rho = float(fields.get("rho_infused") or fields.get("rho") or 0.0)
    if rho <= 0.0:
        raise ValueError("a neat basis row needs rho or rho_infused")
    mechanical = {
        key: float(value)
        for key, value in fields.items()
        if key in {"E", "Ex", "Ey", "Ez", "Gxy", "Gxz", "Gyz", "nuxy", "nuxz", "nuyz"}
    }
    ortho = ("Ex", "Ey", "Ez", "Gxy", "Gxz", "Gyz", "nuxy", "nuxz", "nuyz")
    if all(key in mechanical for key in ortho):
        return Material(
            rho=rho, source="neat", **{key: mechanical[key] for key in ortho}
        )
    modulus = mechanical.get("E", mechanical.get("Ex"))
    if modulus is None:
        raise ValueError("a neat basis row needs E or Ex")
    nu = float(fields.get("nu", fields.get("nuxy", 0.3)))
    return Material(E=modulus, nu=nu, rho=rho, source="neat")


def split_basis(datasheet_rows: list[Any]) -> tuple[dict[str, Material], TargetSet]:
    """Neat rows become materials. Infused rows stay targets.

    A neat row with ``source="resin"`` fills the resin. Every other neat row
    fills the core. ``nu`` defaults to 0.3 when the neat card omits it.
    """
    core_fields: dict[str, float] = {}
    resin_fields: dict[str, float] = {}
    infused: list[Target] = []
    for row in datasheet_rows:
        target = row if isinstance(row, Target) else Target.model_validate(row)
        if target.basis != "neat":
            infused.append(target)
            continue
        if target.value is None:
            continue
        bucket = resin_fields if target.source == "resin" else core_fields
        bucket[target.quantity] = float(target.value)
    materials: dict[str, Material] = {}
    if core_fields:
        materials["core"] = _material_from_neat(core_fields)
    if resin_fields:
        materials["resin"] = _material_from_neat(resin_fields)
    return materials, TargetSet(targets=infused)
