"""One-page Typst datasheet for a grooved-core homogenisation run.

Title banner, three header tables (RVE/geometry, materials, analysis), three
figure panels (groove cross-section cuts with the mesh, a 3D isometric of the
internal structure, and the directional Young's-modulus surface), and a footer
with the engineering constants + 6x6 effective stiffness.

The figures are produced by the shared :mod:`b3_core.viz` layer; the composition
is Typst (vector tables + embedded PNG panels). Data comes from a cached
:class:`b3_core.viz.CoreModel` (MFEM backend).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from b3_core import __version__
from b3_core.viz import slices
from b3_core.viz.model import CoreModel
from b3_core.viz.scene import CoreScene

# Headless: the figure panels never need an interactive display.
matplotlib.use("Agg")

logger = logging.getLogger(__name__)

_FIG_DPI = 150


@dataclass
class DatasheetSpec:
    """Everything the Typst page needs, assembled by `collect_spec`."""

    title: str
    config_path: str
    version: str
    rve_rows: list[tuple[str, str]]
    material_rows: list[tuple[str, str]]
    analysis_rows: list[tuple[str, str]]
    engineering_constants: dict[str, float] | None = None
    c_eff_gpa: "object" = None
    mesh_n_cells: int | None = None
    figure_cuts: Path | None = None
    figure_iso: Path | None = None
    figure_modulus: Path | None = None
    figure_col_fracs: tuple[float, ...] = (0.34, 0.33, 0.33)


# --------------------------------------------------------------------------- #
# spec assembly
# --------------------------------------------------------------------------- #
def _groove_rows(prefix: str, grooves: list[list[float]]) -> list[tuple[str, str]]:
    rows = []
    for k, g in enumerate(grooves, 1):
        off, pitch, depth, width = g
        rows.append(
            (
                f"{prefix}-groove {k}",
                f"off {off:.3g}, pitch {pitch:.3g}, depth {depth:.3g}, w {width:.3g}",
            )
        )
    return rows


def collect_spec(inp: dict, mesh, geom: dict, details, *, name: str, config_name: str):
    """Build the three header-table row lists + the result block from a run."""
    import numpy as np

    rve_rows: list[tuple[str, str]] = [
        (
            "RVE size [mm]",
            f"{inp['dx']:.4g} × {inp['dy']:.4g} × {inp['thickness']:.4g}",
        ),  # noqa: RUF001
        ("x-grooves", str(len(inp["xgr"]))),
        *_groove_rows("x", inp["xgr"]),
        ("y-grooves", str(len(inp["ygr"]))),
        *_groove_rows("y", inp["ygr"]),
    ]
    cur = inp.get("curvature") or {}
    if cur.get("kx") or cur.get("ky"):
        rve_rows.append(
            (
                "curvature [1/mm]",
                f"kx {cur.get('kx', 0):.4g}, ky {cur.get('ky', 0):.4g}",
            )
        )
    face = inp.get("face") or {}
    rve_rows.append(("face thickness [mm]", f"{face.get('thickness', 0.0):.3g}"))
    rve_rows.append(("mesh refine (madd)", ", ".join(f"{m:.2g}" for m in inp["madd"])))

    def _mat(label, m):
        return (label, f"E {m['E'] / 1e9:.3g} GPa, ν {m['nu']:.3g}, ρ {m['rho']:.0f}")  # noqa: RUF001

    material_rows = [
        _mat("core", inp["core"]),
        _mat("resin", inp["resin"]),
        ("resin Vf", f"{geom['resin_vf']:.3f}"),
        ("ρ infused [kg/m³]", f"{geom['rho_infused']:.1f}"),  # noqa: RUF001
        ("groove area factor", f"{geom['area_increase']:.3f}"),
    ]
    analysis_rows = [
        ("backend", "mfem (periodic)"),
        ("element type", inp["element_type"]),
        ("mesh cells", f"{mesh.n_cells}"),
        ("BC", "periodic, 6 unit strains"),
        ("units", "geom mm · props SI"),
    ]
    p = details.properties
    eng = {
        "E_x": p["Exx"],
        "E_y": p["Eyy"],
        "E_z": p["Ezz"],
        "G_xy": p["Gxy"],
        "G_xz": p["Gxz"],
        "G_yz": p["Gyz"],
        "nu_xy": p["nuxy"],
        "nu_xz": p["nuxz"],
        "nu_yz": p["nuyz"],
    }
    return DatasheetSpec(
        title=f"Grooved core — {name}",
        config_path=config_name,
        version=f"b3_core {__version__}",
        rve_rows=rve_rows,
        material_rows=material_rows,
        analysis_rows=analysis_rows,
        engineering_constants=eng,
        c_eff_gpa=np.asarray(details.stiffness) * 1e-9,
        mesh_n_cells=mesh.n_cells,
    )


# --------------------------------------------------------------------------- #
# Typst composition (helpers adapted from b3_tex/datasheet.py)
# --------------------------------------------------------------------------- #
def _typst_escape(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("#", "\\#")
        .replace("$", "\\$")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace("_", "\\_")
    )


def _render_table(header: str, rows: list[tuple[str, str]]) -> str:
    cells = ", ".join(
        f"[#text(size: 6.5pt)[{_typst_escape(c)}]]" for a, b in rows for c in (a, b)
    )
    return (
        f'#text(weight: "bold", size: 7.5pt)[{header}]\n'
        "#table(\n  columns: (auto, 1fr),\n  inset: 1.5pt,\n  stroke: 0.3pt,\n"
        f"  align: (left + top),\n  {cells}\n)\n"
    )


def _figure_image(path: Path) -> str:
    return f'#image("{path.name}", width: 100%, height: 100%, fit: "contain")\n'


def _png_aspect(path: Path) -> float:
    arr = plt.imread(path)
    return float(arr.shape[1] / arr.shape[0])


def build_typst(spec: DatasheetSpec) -> str:
    rve = _render_table("RVE / Geometry", spec.rve_rows)
    materials = _render_table("Materials", spec.material_rows)
    analysis = _render_table("Analysis", spec.analysis_rows)

    title_block = (
        f'#align(center)[#text(size: 10pt, weight: "bold")[{_typst_escape(spec.title)}]'
        f" #text(size: 6pt)[· {_typst_escape(spec.config_path)} · "
        f"{_typst_escape(spec.version)}]]\n"
        "#v(1pt)\n#grid(\n  columns: (1fr, 1fr, 1fr),\n  column-gutter: 0.14cm,\n"
        f"  [{rve}],\n  [{materials}],\n  [{analysis}],\n)\n"
    )

    if spec.engineering_constants is not None and spec.c_eff_gpa is not None:
        ec = spec.engineering_constants
        eng_block = (
            '#text(weight: "bold", size: 7pt)[Engineering constants]\n'
            "#text(size: 6.5pt)["
            f"$E$ = ({ec['E_x'] / 1e9:.1f}, {ec['E_y'] / 1e9:.1f}, {ec['E_z'] / 1e9:.1f}) GPa; "
            f"$G$ = ({ec['G_xy'] / 1e9:.2f}, {ec['G_xz'] / 1e9:.2f}, {ec['G_yz'] / 1e9:.2f}) GPa; "
            f"$nu$ = ({ec['nu_xy']:.2f}, {ec['nu_xz']:.2f}, {ec['nu_yz']:.2f})]\n"
        )
        labels = ["11", "22", "33", "23", "13", "12"]
        c = spec.c_eff_gpa
        matrix_rows = [("", *labels)]
        for i in range(6):
            matrix_rows.append((labels[i], *(f"{c[i, j]:.2f}" for j in range(6))))
        mcells = ", ".join(
            f"[#text(size: 6pt)[{_typst_escape(str(x))}]]"
            for row in matrix_rows
            for x in row
        )
        matrix_block = (
            "#table(\n  columns: (auto,) + (auto,) * 6,\n  inset: 1.5pt,\n"
            f"  stroke: 0.3pt,\n  {mcells}\n)\n"
        )
        footer = (
            "#grid(\n  columns: (1fr, 1fr),\n  column-gutter: 0.25cm,\n"
            "  align: (left + top, left + top),\n"
            f"  [{eng_block}],\n"
            f'  [#text(weight: "bold", size: 7pt)[$C_"eff"$ [GPa]] #v(0.5pt) {matrix_block}],\n)\n'
        )
    else:
        footer = '#text(size: 7pt)[(homogenisation skipped — no $C_"eff"$)]\n'

    fracs = list(spec.figure_col_fracs)
    panels = []
    for i, fpath in enumerate((spec.figure_cuts, spec.figure_iso, spec.figure_modulus)):
        if fpath and fpath.is_file():
            panels.append((_figure_image(fpath), fracs[i] if i < len(fracs) else 1.0))

    head = (
        '#set page(paper: "a4", flipped: true, margin: (x: 0.32cm, y: 0.2cm))\n'
        '#set text(size: 7.5pt, font: "Liberation Sans")\n'
        "#set par(leading: 0.42em)\n"
    )
    if not panels:
        return head + title_block + "#v(0.5pt)\n" + footer + "\n"

    cols = ", ".join(f"{frac}fr" for _, frac in panels)
    cells = "".join(f"  [{img}],\n" for img, _ in panels)
    aligns = ", ".join(["center + horizon"] * len(panels))
    figure_row = (
        "#block(width: 100%, height: 100%)[#grid(\n"
        f"  columns: ({cols}),\n  rows: (1fr,),\n  column-gutter: 0.15cm,\n"
        f"  align: ({aligns}),\n{cells})]\n"
    )
    return (
        head + "#grid(\n  rows: (auto, 1fr, auto),\n  row-gutter: 0.12cm,\n"
        f"  [{title_block}],\n  [{figure_row}],\n  [{footer}],\n)\n"
    )


def compile_datasheet(
    typst_src: str,
    out_pdf: Path,
    *,
    out_png: Path | None = None,
    root: Path | None = None,
) -> None:
    """Compile Typst to PDF (and optionally PNG); `root` holds the figure PNGs."""
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    root = Path(root or out_pdf.parent)
    src = root / "datasheet.typ"
    src.write_text(typst_src)
    proc = subprocess.run(
        ["typst", "compile", src.name, str(out_pdf.resolve())],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"typst compile failed ({proc.returncode}):\n{proc.stderr}")
    if out_png is not None:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        proc2 = subprocess.run(
            [
                "typst",
                "compile",
                "--format",
                "png",
                "--ppi",
                "150",
                src.name,
                "datasheet-{p}.png",
            ],
            cwd=root,
            capture_output=True,
            text=True,
        )
        if proc2.returncode != 0:
            raise RuntimeError(
                f"typst png export failed ({proc2.returncode}):\n{proc2.stderr}"
            )
        shutil.copy(next(root.glob("datasheet-*.png")), out_png)


# --------------------------------------------------------------------------- #
# top-level entry point
# --------------------------------------------------------------------------- #
def generate(
    json_path: str | Path,
    out_pdf: str | Path | None = None,
    *,
    out_png: str | Path | None = None,
    name: str | None = None,
    workdir: str | Path | None = None,
    skip_compile: bool = False,
) -> DatasheetSpec:
    """Build a datasheet for a CpropInput JSON case (figures via b3_core.viz)."""
    model = CoreModel.from_json(json_path)
    if name:
        model.name = name

    root = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="b3core_ds_"))
    root.mkdir(parents=True, exist_ok=True)
    fig_cuts, fig_iso, fig_mod = (
        root / "cuts.png",
        root / "iso.png",
        root / "modulus.png",
    )

    fig, a_cuts = slices.plot_orthogonal_cuts(model)
    fig.savefig(fig_cuts, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    iso = CoreScene(model).add_phases().add_axes().isometric()
    iso.screenshot(fig_iso)
    iso.close()
    mod = CoreScene(model).add_modulus_surface().add_axes().isometric()
    mod.screenshot(fig_mod)
    mod.close()

    a_iso, a_mod = _png_aspect(fig_iso), _png_aspect(fig_mod)
    total = a_cuts + a_iso + a_mod

    spec = collect_spec(
        model.inp,
        model.mesh,
        model.geom,
        model.details,
        name=model.name,
        config_name=model.config_path or Path(json_path).name,
    )
    spec.figure_cuts, spec.figure_iso, spec.figure_modulus = fig_cuts, fig_iso, fig_mod
    spec.figure_col_fracs = (a_cuts / total, a_iso / total, a_mod / total)

    if out_pdf is not None and not skip_compile:
        compile_datasheet(build_typst(spec), out_pdf, out_png=out_png, root=root)
    return spec
