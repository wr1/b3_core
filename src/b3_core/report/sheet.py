"""Full datasheet: Python writes data.json, Typst only reads it."""

from __future__ import annotations

import json
import shutil
import subprocess
from importlib.resources import files
from pathlib import Path
from typing import Any

from b3_core.report.prose import Prose
from b3_core.report.tables import version_block
from b3_core.units import curvature_tick

_FOOTNOTE = "cross-checked on flat; curved values FEniCSx"


class DatasheetRejected(RuntimeError):
    """``--full`` without a passing acceptance stamp, and without ``--draft``."""


def _curved(data: dict[str, Any]) -> bool:
    for key in ("kx", "ky"):
        try:
            if abs(float(data.get(key) or 0.0)) > 0.0:
                return True
        except (TypeError, ValueError):
            continue
    for row in data.get("grid_rows") or []:
        if abs(float(row.get("kx") or 0.0)) > 0 or abs(float(row.get("ky") or 0.0)) > 0:
            return True
    return False


def _cell(value: Any) -> str | int | float:
    """Typst ``str()`` rejects booleans. Keep numbers; stringify the rest."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        return value
    return json.dumps(value)


def assemble_data(
    *,
    prose: Prose | dict,
    properties: dict[str, Any] | None = None,
    convention: list[dict[str, Any]] | None = None,
    provenance: dict[str, Any] | None = None,
    acceptance: dict[str, Any] | None = None,
    figures: dict[str, str] | None = None,
    kx: float = 0.0,
    ky: float = 0.0,
    grid_rows: list[dict[str, Any]] | None = None,
    draft: bool = False,
) -> dict[str, Any]:
    """The JSON the template is allowed to read. No hand-typed numbers."""
    body = prose if isinstance(prose, Prose) else Prose.model_validate(prose)
    accepted = bool(acceptance and acceptance.get("ok"))
    checks = []
    for item in (acceptance or {}).get("checks") or []:
        checks.append(
            {
                "id": str(item.get("id")),
                "ok": bool(item.get("ok")),
                "value": _cell(item.get("value")),
            }
        )
    provenance_text = {
        key: _cell(value)
        for key, value in (provenance or {}).items()
        if not isinstance(value, (dict, list))
    }
    payload = {
        "title": body.title,
        "prose": body.model_dump(),
        "properties": {key: _cell(value) for key, value in (properties or {}).items()},
        "convention": convention or [],
        "provenance": provenance_text,
        "acceptance": {"ok": accepted, "checks": checks},
        "figures": figures or {},
        "kx": float(kx),
        "ky": float(ky),
        "kx_tick": curvature_tick(kx),
        "ky_tick": curvature_tick(ky),
        "grid_rows": grid_rows or [],
        "draft": bool(draft),
        "accepted": accepted,
        "version": {key: _cell(value) for key, value in version_block().items()},
        "footnote": "",
    }
    if _curved(payload) and not accepted:
        payload["footnote"] = _FOOTNOTE
    return payload


def generate_full(
    directory: str | Path,
    data: dict[str, Any],
    *,
    draft: bool = False,
    template: str = "full.typ",
    compile: bool = True,
) -> Path:
    """Write ``data.json`` and, unless refused, the Typst PDF.

    A curved or unstamped sheet compiles only when ``draft`` is true or the
    acceptance object says ``ok``.
    """
    dest = Path(directory)
    dest.mkdir(parents=True, exist_ok=True)
    accepted = bool(data.get("acceptance", {}).get("ok")) or bool(data.get("accepted"))
    if not draft and not accepted:
        raise DatasheetRejected(
            "acceptance stamp is missing or not ok; pass draft=True to build anyway"
        )
    data = dict(data)
    data["draft"] = bool(draft) or not accepted
    (dest / "data.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    source = files("b3_core.templates").joinpath(template).read_text(encoding="utf-8")
    typst_path = dest / template
    typst_path.write_text(source, encoding="utf-8")
    pdf = dest / "datasheet.pdf"
    if compile:
        _compile(typst_path, pdf)
    return pdf


def _compile(src: Path, pdf: Path) -> None:
    try:
        import typst as typst_mod
    except ImportError:
        typst_mod = None
    if typst_mod is not None and hasattr(typst_mod, "compile"):
        typst_mod.compile(str(src), output=str(pdf))
        return
    if shutil.which("typst") is None:
        raise RuntimeError(
            "typst is not installed. Install the report extra or put typst on PATH."
        )
    proc = subprocess.run(
        ["typst", "compile", src.name, pdf.name],
        cwd=src.parent,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"typst compile failed ({proc.returncode}):\n{proc.stderr}")
