"""The only case loaders. ``_``-prefixed keys are stripped here."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

from b3_core.models import CaseInput


def _strip(data: dict) -> dict:
    return {key: value for key, value in data.items() if not str(key).startswith("_")}


def load_case(path: str) -> tuple[dict, str]:
    """Load JSON or YAML. Returns ``(dict, directory)``."""
    suffix = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as handle:
        if suffix in (".yaml", ".yml"):
            data = yaml.safe_load(handle) or {}
        elif suffix == ".json":
            data = json.load(handle)
        else:
            raise ValueError(
                f"unsupported case file type {suffix!r}; use .yaml, .yml, or .json"
            )
    if not isinstance(data, dict):
        raise ValueError(f"case file {path} did not contain a mapping")
    return data, os.path.dirname(path)


def from_dict(data: dict) -> Any:
    """Validate a mapping as a :class:`~b3_core.cases.CoreCase`."""
    from b3_core.cases import CoreCase

    return CoreCase(input=CaseInput(**_strip(data)))


def from_path(path: str | Path) -> Any:
    """Load a JSON or YAML file as a :class:`~b3_core.cases.CoreCase`."""
    from b3_core.cases import CoreCase

    data, dirname = load_case(str(path))
    return CoreCase(input=CaseInput(**_strip(data)), workdir=dirname or None)


def normalize_case(case_data: Any) -> tuple[CaseInput, str]:
    """Return ``(CaseInput, workdir)``.

    Accepts a path, a dict, a :class:`CaseInput`, or a :class:`~b3_core.cases.CoreCase`
    (``Textile``). Workdir is the file's directory for a path, the textile
    workdir when one was set, otherwise ``"."``.
    """
    from b3_core.cases import CoreCase

    if isinstance(case_data, CoreCase):
        workdir = case_data.workdir or "."
        return case_data.input, str(workdir)

    if isinstance(case_data, CaseInput):
        return case_data, "."

    if isinstance(case_data, (str, Path)):
        data, dirname = load_case(str(case_data))
        return CaseInput(**_strip(data)), (dirname or ".")

    if isinstance(case_data, dict):
        return CaseInput(**_strip(case_data)), "."

    raise TypeError(
        "expected a path, dict, CaseInput, CpropInput, CoreCase, or Textile, "
        f"got {type(case_data).__name__}"
    )
