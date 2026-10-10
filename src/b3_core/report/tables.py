"""Markdown and Typst table fragments. Numbers come from the caller."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from b3_core import __version__
from b3_core.provenance import code_revision
from b3_core.units import curvature_tick


def version_block() -> dict[str, Any]:
    commit, dirty = code_revision()
    return {
        "b3_core": __version__,
        "commit": commit,
        "dirty": dirty,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _md_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("@", "\\@")


def markdown_table(
    headers: list[str], rows: list[list[Any]], *, version: dict | None = None
) -> str:
    block = version or version_block()
    lines = [
        (
            f"b3_core {block['b3_core']} · commit {block['commit']} · "
            f"dirty {block['dirty']} · {block['generated_utc']}"
        ),
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_md_cell(cell) for cell in row) + " |")
    return "\n".join(lines) + "\n"


def _typst_string(value: Any) -> str:
    text = str(value).replace("\\", "\\\\").replace('"', '\\"').replace("@", "\\@")
    return f'"{text}"'


def typst_table(
    headers: list[str], rows: list[list[Any]], *, version: dict | None = None
) -> str:
    """One parenthesised tuple per row, then flattened into cells."""
    block = version or version_block()
    body = ",\n  ".join(
        "(" + ", ".join(_typst_string(cell) for cell in row) + ")" for row in rows
    )
    header = ", ".join(f"[{_md_cell(name)}]" for name in headers)
    note = (
        f"b3_core {block['b3_core']} · {block['commit']} · "
        f"dirty {block['dirty']} · {block['generated_utc']}"
    )
    return (
        f"#text(size: 8pt)[{_typst_string(note)[1:-1]}]\n"
        "#{\n"
        f"  let rows = (\n  {body},\n  )\n"
        "  table(\n"
        f"    columns: {len(headers)},\n"
        f"    table.header({header}),\n"
        "    ..rows.flatten(),\n"
        "  )\n"
        "}\n"
    )


def curvature_label(k_per_mm: float) -> str:
    return curvature_tick(k_per_mm)
