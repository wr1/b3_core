#!/usr/bin/env python3

import json as json_module
from pathlib import Path

from treeparse import argument, cli, command, option

from b3_core.core.cprop import cprop


def cmd_json(path: str):
    cprop(path)


def cmd_datasheet(path: str, output: str, png: str):
    from b3_core.datasheet import generate

    out_pdf = output or str(Path(path).with_suffix(".pdf"))
    generate(path, out_pdf, out_png=(png or None))
    print(f"Wrote {out_pdf}" + (f" and {png}" if png else ""))


def _parse_grooves(s: str):
    return [[float(x) for x in g.split(",")] for g in s.split(";") if g.strip()]


def cmd_run(
    dx: float,
    dy: float,
    thickness: float,
    xgr: str,
    ygr: str,
    core_e: float,
    core_nu: float,
    core_rho: float,
    resin_e: float,
    resin_nu: float,
    resin_rho: float,
    face_thickness: float,
    kx: float,
    ky: float,
    element_type: str,
    backend: str,
    validate_with_ccx: bool,
    output_dir: str,
):
    cfg = {
        "dx": dx,
        "dy": dy,
        "thickness": thickness,
        "xgr": _parse_grooves(xgr),
        "ygr": _parse_grooves(ygr),
        "core": {"E": core_e, "nu": core_nu, "rho": core_rho},
        "resin": {"E": resin_e, "nu": resin_nu, "rho": resin_rho},
        "element_type": element_type,
        "backend": backend,
        "validate_with_ccx": validate_with_ccx,
    }
    if face_thickness > 0:
        cfg["face"] = {"thickness": face_thickness}
    if kx or ky:
        cfg["curvature"] = {"kx": kx, "ky": ky}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = out / "_cli_input.json"
    cfg_path.write_text(json_module.dumps(cfg, indent=2))
    cprop(str(cfg_path))


def main():
    app = cli(
        name="b3_core",
        help="FEA homogenisation of grooved-core sandwich panels.",
        commands=[
            command(
                name="json",
                help="Run a case from a JSON config file.",
                callback=cmd_json,
                arguments=[
                    argument(name="path", arg_type=str, help="Path to the case JSON."),
                ],
            ),
            command(
                name="datasheet",
                help="Render a one-page datasheet (PDF/PNG) for a JSON case.",
                callback=cmd_datasheet,
                arguments=[
                    argument(name="path", arg_type=str, help="Path to the case JSON."),
                ],
                options=[
                    option(flags=["--output", "-o"], arg_type=str, default="",
                           help="Output PDF path (default: <case>.pdf)."),
                    option(flags=["--png"], arg_type=str, default="",
                           help="Also export a PNG to this path."),
                ],
            ),
            command(
                name="run",
                help="Run a case defined directly via CLI flags.",
                callback=cmd_run,
                options=[
                    option(flags=["--dx"], arg_type=float, default=50.0,
                           help="Panel width along x (mm)."),
                    option(flags=["--dy"], arg_type=float, default=50.0,
                           help="Panel width along y (mm)."),
                    option(flags=["--thickness", "-t"], arg_type=float, default=30.0,
                           help="Panel core thickness (mm)."),
                    option(flags=["--xgr"], arg_type=str, default="",
                           help="x-grooves; ';'-separated 'off,pitch,depth,width' "
                                "(use --xgr=... when values start with '-')."),
                    option(flags=["--ygr"], arg_type=str, default="",
                           help="y-grooves; ';'-separated 'off,pitch,depth,width' "
                                "(use --ygr=... when values start with '-')."),
                    option(flags=["--core-e"], arg_type=float, default=4e9,
                           help="Core Young's modulus (Pa)."),
                    option(flags=["--core-nu"], arg_type=float, default=0.3,
                           help="Core Poisson ratio."),
                    option(flags=["--core-rho"], arg_type=float, default=100.0,
                           help="Core density (kg/m3)."),
                    option(flags=["--resin-e"], arg_type=float, default=4e9,
                           help="Resin Young's modulus (Pa)."),
                    option(flags=["--resin-nu"], arg_type=float, default=0.3,
                           help="Resin Poisson ratio."),
                    option(flags=["--resin-rho"], arg_type=float, default=1100.0,
                           help="Resin density (kg/m3)."),
                    option(flags=["--face-thickness"], arg_type=float, default=0.0,
                           help="Face skin thickness (mm); 0 = no skin."),
                    option(flags=["--kx"], arg_type=float, default=0.0,
                           help="Mold curvature 1/Rx (1/mm) tapering x-grooves; "
                                "0 = flat (use --kx=... for negatives)."),
                    option(flags=["--ky"], arg_type=float, default=0.0,
                           help="Mold curvature 1/Ry (1/mm) tapering y-grooves; "
                                "0 = flat (use --ky=... for negatives)."),
                    option(flags=["--element-type"], arg_type=str, default="C3D8",
                           choices=["C3D8", "C3D20"], help="CCX element type."),
                    option(flags=["--backend"], arg_type=str, default="ccx",
                           choices=["ccx", "fenicsx", "mfem"], help="Solver backend."),
                    option(flags=["--validate-with-ccx"], arg_type=bool, default=False,
                           help="Compare the selected backend against CCX results."),
                    option(flags=["--output-dir", "-o"], arg_type=str, default=".",
                           help="Where to write run<HASH> outputs."),
                ],
            ),
        ],
    )
    app.run()


if __name__ == "__main__":
    main()
