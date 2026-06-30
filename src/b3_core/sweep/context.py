"""Parametric sweep paths and homogenisation helpers."""

from __future__ import annotations

import copy
import glob
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from b3_core.core.cprop import cprop

MODULI = ["Exx", "Eyy", "Ezz", "Gxy"]
PATTERNS = ["plain", "uniaxial", "crossed", "two_sided"]
THICKNESSES = [20, 25, 30, 40, 50]
KX = [-0.008, -0.004, 0.0, 0.004, 0.008]
REF_THICKNESS = 30.0
LIGAMENT = 3.0


def default_root() -> Path:
    return Path(__file__).resolve().parents[3] / "examples" / "param_sweeps"


@dataclass(frozen=True)
class SweepContext:
    root: Path

    @property
    def out(self) -> Path:
        return self.root / "out"

    @property
    def img(self) -> Path:
        return self.root / "img"

    @property
    def bases(self) -> Path:
        return self.root / "bases"

    @property
    def mfem_patterns(self) -> Path:
        return self.root.parent / "mfem_patterns"


def load_base(ctx: SweepContext, name: str) -> dict:
    return json.loads((ctx.bases / f"{name}.json").read_text())


def load_pattern(ctx: SweepContext, name: str) -> dict:
    return json.loads((ctx.mfem_patterns / f"{name}.json").read_text())


def tag_float(value: float, *, prefix: str = "") -> str:
    s = f"{value:+.4f}".replace("+", "p").replace("-", "m").replace(".", "_")
    return f"{prefix}{s}" if prefix else s


def tag_thickness(t: float) -> str:
    return f"thickness_{int(t)}"


def tag_kx(kx: float) -> str:
    return f"kx_{tag_float(kx)}"


def tag_pattern(name: str) -> str:
    return f"pattern_{name}"


def run_case(base: dict, overrides: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    case = copy.deepcopy(base)
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(case.get(key), dict):
            case[key] = {**case[key], **val}
        else:
            case[key] = val
    case_path = out_dir / "case.json"
    case_path.write_text(json.dumps(case, indent=2))
    try:
        return cprop(str(case_path))
    except FileExistsError:
        cached = glob.glob(str(out_dir / "run*.json"))
        if not cached:
            raise
        return json.loads(Path(cached[0]).read_text())


def scale_groove_depth(depth: float, thickness: float, *, ref: float = REF_THICKNESS) -> float:
    sign = -1.0 if depth < 0 else 1.0
    mag = abs(depth) * thickness / ref
    cap = max(thickness - LIGAMENT, 0.5)
    return sign * min(mag, cap)


def scale_depths_for_thickness(
    cuts: list[list[float]], thickness: float, *, ref: float = REF_THICKNESS
) -> list[list[float]]:
    return [
        [offset, pitch, scale_groove_depth(depth, thickness, ref=ref), width]
        for offset, pitch, depth, width in cuts
    ]


def case_for_thickness(base: dict, thickness: float) -> dict:
    case = copy.deepcopy(base)
    case["thickness"] = thickness
    case["xgr"] = scale_depths_for_thickness(case.get("xgr", []), thickness)
    case["ygr"] = scale_depths_for_thickness(case.get("ygr", []), thickness)
    return case


def case_for_curvature(base: dict, kx: float) -> dict:
    case = copy.deepcopy(base)
    case["curvature"] = {"kx": kx, "ky": 0.0}
    return case


def collect_sweep(ctx: SweepContext, prefix: str) -> list[tuple[str, dict, dict]]:
    rows: list[tuple[str, dict, dict, str]] = []
    if not ctx.out.exists():
        return []
    for d in sorted(ctx.out.iterdir()):
        if not d.is_dir() or not d.name.startswith(prefix):
            continue
        case_path = d / "case.json"
        if not case_path.exists():
            continue
        cached = glob.glob(str(d / "run*.json"))
        if not cached:
            continue
        case = json.loads(case_path.read_text())
        result = json.loads(Path(cached[0]).read_text())
        rows.append((d.name, case, result, d.name))
    return [(tag, case, result) for tag, case, result, _ in rows]


def _gpa(val: float) -> str:
    return f"{val / 1e9:.4f}"


def print_moduli_table(
    console: Console,
    *,
    title: str,
    rows: list[tuple[str, dict]],
    extra_cols: list[tuple[str, Any]] | None = None,
) -> None:
    table = Table(title=title)
    table.add_column("case", justify="left", style="bold")
    if extra_cols:
        for name, _ in extra_cols:
            table.add_column(name, justify="right")
    table.add_column("resin_vf", justify="right")
    table.add_column("rho_inf", justify="right")
    for key in MODULI:
        table.add_column(key, justify="right")

    for label, out in rows:
        cells = [label]
        if extra_cols:
            for _, fn in extra_cols:
                cells.append(str(fn(out)))
        cells.extend(
            [
                f"{out['resin_vf']:.4f}",
                f"{out['rho_infused']:.0f}",
                *[_gpa(out[k]) for k in MODULI],
            ]
        )
        table.add_row(*cells)
    console.print(table)


def max_rel_err(output: dict) -> tuple[float, bool]:
    validation = output.get("ccx_validation") or {}
    props = validation.get("properties") or {}
    errs = [p["rel_error"] for p in props.values()]
    return (max(errs) if errs else 0.0), bool(validation.get("passed", True))


def parse_thickness_tag(tag: str) -> float | None:
    m = re.match(r"thickness_(\d+)$", tag)
    return float(m.group(1)) if m else None


def parse_kx_tag(tag: str) -> float | None:
    m = re.match(r"kx_(m|p)(\d)_(\d{4})$", tag)
    if not m:
        return None
    sign = -1 if m.group(1) == "m" else 1
    return sign * float(f"{m.group(2)}.{m.group(3)}")


def parse_pattern_tag(tag: str) -> str | None:
    m = re.match(r"pattern_(.+)$", tag)
    return m.group(1) if m else None