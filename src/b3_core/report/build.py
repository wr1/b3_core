"""Report skeleton from a project file. Prose is templated; numbers are JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from b3_core.report.tables import (
    curvature_label,
    markdown_table,
    typst_table,
    version_block,
)


def _read(path: str | Path) -> Any:
    text = Path(path).read_text(encoding="utf-8")
    if str(path).endswith((".yaml", ".yml")):
        return yaml.safe_load(text)
    return json.loads(text)


def _rows_from_grid(grid: Any) -> list[list[str]]:
    rows = []
    source = grid.get("rows") if isinstance(grid, dict) else grid
    for row in source or []:
        if not row.get("ok", True):
            continue
        rows.append(
            [
                curvature_label(float(row.get("kx") or 0.0)),
                curvature_label(float(row.get("ky") or 0.0)),
                _fmt(row.get("Ex")),
                _fmt(row.get("Ez")),
                _fmt(row.get("Gxz")),
                _fmt(row.get("rho_infused")),
            ]
        )
    return rows


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def build_report(
    project: Any, out: str | Path, *, template: str | None = None
) -> dict[str, str]:
    """Write ``report.md`` and ``report.typ``. ``template`` is copied beside them."""
    if isinstance(project, str | Path):
        project = _read(project)
    project = dict(project or {})
    dest = Path(out)
    dest.mkdir(parents=True, exist_ok=True)
    grid = project.get("grid")
    if isinstance(grid, str):
        grid = _read(grid)
    version = version_block()
    (dest / "version.json").write_text(
        json.dumps(version, indent=2) + "\n", encoding="utf-8"
    )
    headers = ["kx", "ky", "Ex", "Ez", "Gxz", "rho infused"]
    body = _rows_from_grid(grid)
    title = str(project.get("title") or "Core report")
    md = f"# {title}\n\n" + markdown_table(headers, body, version=version)
    (dest / "report.md").write_text(md, encoding="utf-8")
    typst = (
        '#set page(paper: "a4", margin: 14mm)\n'
        "#set text(size: 10pt)\n"
        f"= {title}\n\n" + typst_table(headers, body, version=version)
    )
    (dest / "report.typ").write_text(typst, encoding="utf-8")
    written = {
        "report_md": str(dest / "report.md"),
        "report_typ": str(dest / "report.typ"),
        "version": str(dest / "version.json"),
    }
    if template:
        target = dest / Path(template).name
        target.write_text(Path(template).read_text(encoding="utf-8"), encoding="utf-8")
        written["template"] = str(target)
    return written
