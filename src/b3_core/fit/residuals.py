"""Measured-versus-model residuals. Missing targets are kept and unused."""

from __future__ import annotations

from typing import Any

from b3_core.fit.spec import ResidualReport, Target, TargetSet


def model_quantity(target: Target) -> str:
    """Rewrite a test-axis name with ``axis_map`` (test axis → model axis)."""
    name = target.quantity
    if not target.axis_map:
        return name
    converted = name
    for source, dest in sorted(target.axis_map.items(), key=lambda item: -len(item[0])):
        converted = converted.replace(source, dest)
    if converted.startswith("G") and len(converted) == 3:
        axes = "".join(sorted(converted[1:]))
        converted = "G" + axes
    return converted


def _properties(record_or_props: Any) -> dict[str, float]:
    if isinstance(record_or_props, dict):
        return {
            key: float(value)
            for key, value in record_or_props.items()
            if _number(value)
        }
    props: dict[str, float] = {}
    result = getattr(record_or_props, "result", None)
    if result is not None:
        for key, value in result.properties.items():
            if _number(value):
                props[key] = float(value)
    geometry = getattr(record_or_props, "geometry", None) or {}
    for key, value in geometry.items():
        if _number(value):
            props.setdefault(key, float(value))
    return props


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _sd(target: Target) -> float | None:
    if target.value is None:
        return None
    if target.sd is not None:
        return float(target.sd)
    rel = 0.05 if target.rel_sd is None else float(target.rel_sd)
    return abs(rel * float(target.value))


def residuals(record_or_props: Any, targets: TargetSet) -> ResidualReport:
    """``z = (model − measured) / sd``. Default ``rel_sd`` is 5 %."""
    props = _properties(record_or_props)
    rows: list[dict] = []
    used_z: list[float] = []
    chi2 = 0.0
    for target in targets.targets:
        key = model_quantity(target)
        measured = target.value
        model = props.get(key)
        sd = _sd(target)
        used = (
            target.basis == "infused"
            and measured is not None
            and model is not None
            and sd is not None
            and sd > 0.0
        )
        z = None
        if used and measured is not None and model is not None and sd is not None:
            z = (float(model) - float(measured)) / sd
            used_z.append(abs(z))
            chi2 += float(target.weight) * z * z
        rows.append(
            {
                "quantity": target.quantity,
                "model_quantity": key,
                "measured": measured,
                "model": model,
                "abs": None
                if model is None or measured is None
                else float(model) - float(measured),
                "rel": None
                if model is None or not measured
                else (float(model) - float(measured)) / float(measured),
                "z": z,
                "used": used,
            }
        )
    return ResidualReport(
        rows=rows,
        chi2=float(chi2),
        n_used=sum(1 for row in rows if row["used"]),
        max_abs_z=max(used_z) if used_z else 0.0,
    )
