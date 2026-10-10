"""Parameter catalogue and the writer that applies it to a case."""

from __future__ import annotations

import re
from typing import Any

from b3_core.fit.spec import ParamSet, ParamSpec, Prior
from b3_core.loaders import normalize_case
from b3_core.models import CaseInput, Material

_GROUP = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\.\{([A-Za-z0-9_,]+)\}$")
_WILD = re.compile(r"^(xgr|ygr)\[\*\]\.([A-Za-z_][A-Za-z0-9_]*)$")
_FIELD = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)$")


def _scale_prior(rel: float) -> Prior:
    return Prior(kind="normal", mean=1.0, sd=float(rel))


def default_bounds(case: Any, *, family: str = "PVC", rel: float = 0.2) -> ParamSet:
    """Catalogue. Every parameter starts fixed. The caller unfixes some."""
    case_in, _workdir = normalize_case(case)
    from b3_core.fit.materials import scaling_family

    scaling_family(family)
    cell = case_in.core.cell_size
    cell_value = (
        0.6 if cell is None or not isinstance(cell, (int, float)) else float(cell)
    )
    span = float(rel)
    return ParamSet(
        params=[
            ParamSpec(
                name="foam_E_scale",
                path="core.{E,Ex,Ey}",
                transform="scale",
                value=1.0,
                lower=1.0 - span,
                upper=1.0 + span,
                fixed=True,
                prior=_scale_prior(span),
                unit="-",
            ),
            ParamSpec(
                name="foam_Ez_scale",
                path="core.Ez",
                transform="scale",
                value=1.0,
                lower=1.0 - span,
                upper=1.0 + span,
                fixed=True,
                prior=_scale_prior(span),
                unit="-",
            ),
            ParamSpec(
                name="foam_G_scale",
                path="core.{Gxy,Gxz,Gyz}",
                transform="scale",
                value=1.0,
                lower=1.0 - span,
                upper=1.0 + span,
                fixed=True,
                prior=_scale_prior(span),
                unit="-",
            ),
            ParamSpec(
                name="halo_cell_size",
                path="core.cell_size",
                transform="abs",
                value=cell_value,
                lower=0.0,
                upper=2.0,
                fixed=True,
                prior=Prior(kind="uniform"),
                unit="mm",
            ),
            ParamSpec(
                name="kerf_width",
                path="xgr[*].width,ygr[*].width",
                transform="scale",
                value=1.0,
                lower=1.0 - span,
                upper=1.0 + span,
                fixed=True,
                prior=_scale_prior(span),
                unit="-",
            ),
            ParamSpec(
                name="resin_E",
                path="resin.{E,Ex}",
                transform="scale",
                value=1.0,
                lower=1.0 - span,
                upper=1.0 + span,
                fixed=True,
                prior=_scale_prior(span),
                unit="-",
            ),
        ]
    )


def _split_path(path: str) -> list[str]:
    """Split on commas outside ``{groups}``."""
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for char in path:
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
            continue
        buf.append(char)
    if buf:
        parts.append("".join(buf).strip())
    return [part for part in parts if part]


def _physical(param: ParamSpec, transform_value: float, current: float | None) -> float:
    if param.transform == "scale":
        base = 1.0 if current is None else float(current)
        return base * float(transform_value)
    if param.transform == "log":
        import math

        return math.exp(float(transform_value))
    return float(transform_value)


def _set_material_field(material: Material, field: str, value: float) -> None:
    if not hasattr(material, field):
        raise ValueError(f"material has no field {field!r}")
    current = getattr(material, field)
    if current is None and field != "cell_size":
        return
    setattr(material, field, value)


def _apply_one(case: CaseInput, spec: str, value: float, param: ParamSpec) -> None:
    token = spec.strip()
    group = _GROUP.match(token)
    if group:
        owner = getattr(case, group.group(1))
        for field in group.group(2).split(","):
            current = getattr(owner, field, None)
            if current is None and (param.fixed or field != "cell_size"):
                continue
            _set_material_field(
                owner,
                field,
                _physical(param, value, None if current is None else float(current)),
            )
        return
    wild = _WILD.match(token)
    if wild:
        grooves = getattr(case, wild.group(1))
        field = wild.group(2)
        for groove in grooves:
            current = float(getattr(groove, field))
            setattr(groove, field, _physical(param, value, current))
        return
    field_match = _FIELD.match(token)
    if field_match:
        owner = getattr(case, field_match.group(1))
        field = field_match.group(2)
        current = getattr(owner, field)
        if current is None and param.fixed:
            return
        number = None if current is None else float(current)
        _set_material_field(owner, field, _physical(param, value, number))
        return
    raise ValueError(f"cannot apply parameter path {spec!r}")


def apply(case: Any, paramset: ParamSet, x: Any = None) -> CaseInput:
    """Copy ``case`` and write parameter values. ``x`` is the free vector."""
    case_in, _workdir = normalize_case(case)
    updated = case_in.model_copy(deep=True)
    free_values = None if x is None else [float(value) for value in x]
    cursor = 0
    for param in paramset.params:
        if free_values is None or param.fixed:
            value = float(param.value)
        else:
            value = free_values[cursor]
            cursor += 1
        for spec in _split_path(param.path):
            _apply_one(updated, spec, value, param)
    if free_values is not None and cursor != len(free_values):
        raise ValueError(
            f"x has length {len(free_values)}; {cursor} free parameters were written"
        )
    return updated


def values_of(paramset: ParamSet, x: Any = None) -> list[ParamSpec]:
    """Return params with ``value`` set to the applied free vector."""
    free_values = None if x is None else [float(value) for value in x]
    cursor = 0
    written: list[ParamSpec] = []
    for param in paramset.params:
        if free_values is None or param.fixed:
            written.append(param)
            continue
        written.append(param.model_copy(update={"value": free_values[cursor]}))
        cursor += 1
    return written
