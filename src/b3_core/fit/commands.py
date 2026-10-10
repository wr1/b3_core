"""CLI wrappers. A fit result exits 0 only for a finished status."""

import json
from pathlib import Path

from treeparse import argument, command, group, option

from b3_core.core.run import CASE_ARG, error_payload

_OK = {"converged", "rsm_only", "no_free_params"}
_JSON = option(
    flags=["--json", "--agent-json"],
    dest="as_json",
    flag=True,
    help="Print the schema JSON.",
)


def _load_model(path: str, model):
    text = Path(path).read_text(encoding="utf-8")
    if path.endswith((".yaml", ".yml")):
        import yaml

        return model.model_validate(yaml.safe_load(text))
    return model.model_validate(json.loads(text))


def _emit(payload, *, as_json: bool, status: str | None = None) -> None:
    if hasattr(payload, "model_dump"):
        text = json.dumps(payload.model_dump(mode="json"), indent=2)
    else:
        text = json.dumps(payload, indent=2)
    print(text if as_json else text)
    if status is not None and status not in _OK:
        raise SystemExit(1)


def cmd_fit_bounds(
    path: str, family: str = "PVC", rel: float = 0.2, as_json: bool = False
):
    from b3_core.fit.params import default_bounds

    _emit(default_bounds(path, family=family, rel=rel), as_json=as_json)


def cmd_fit_residuals(
    path: str, targets: str, backend: str = "fenicsx", as_json: bool = False
):
    from b3_core.api import run_case
    from b3_core.fit.residuals import residuals
    from b3_core.fit.spec import TargetSet

    try:
        target_set = _load_model(targets, TargetSet)
        record = run_case(path, backend=backend)
        _emit(residuals(record, target_set), as_json=as_json)
    except Exception as exc:
        print(json.dumps(error_payload(exc)))
        raise SystemExit(1) from exc


def cmd_fit_estimate_halo(
    path: str,
    rho_infused: float = 0.0,
    uptake: float = 0.0,
    sd: float = 0.0,
    as_json: bool = False,
):
    from b3_core.fit.volume import estimate_halo

    try:
        estimate = estimate_halo(
            path,
            rho_infused=None if rho_infused == 0.0 else rho_infused,
            uptake_kg_m2=None if uptake == 0.0 else uptake,
            sd=None if sd == 0.0 else sd,
        )
    except Exception as exc:
        print(json.dumps(error_payload(exc)))
        raise SystemExit(1) from exc
    _emit(estimate, as_json=as_json)


def cmd_fit_refine(
    path: str,
    params: str,
    targets: str,
    backend: str = "fenicsx",
    cache: str = "",
    as_json: bool = False,
):
    from b3_core.cache import DiskCache
    from b3_core.fit.refine import refine
    from b3_core.fit.spec import ParamSet, TargetSet

    try:
        result = refine(
            path,
            _load_model(params, ParamSet),
            _load_model(targets, TargetSet),
            backend=backend,
            cache=DiskCache(cache) if cache else None,
        )
    except Exception as exc:
        print(json.dumps(error_payload(exc)))
        raise SystemExit(1) from exc
    _emit(result, as_json=as_json, status=result.status)


def cmd_fit_sensitivity(
    path: str,
    params: str,
    targets: str,
    backend: str = "fenicsx",
    as_json: bool = False,
):
    from b3_core.fit.identify import sensitivity
    from b3_core.fit.spec import ParamSet, TargetSet

    try:
        report = sensitivity(
            path,
            _load_model(params, ParamSet),
            _load_model(targets, TargetSet),
            backend=backend,
        )
    except Exception as exc:
        print(json.dumps(error_payload(exc)))
        raise SystemExit(1) from exc
    _emit(report, as_json=as_json)


def cmd_fit_design(
    path: str,
    params: str,
    method: str = "ccd",
    n: int = 0,
    seed: int = 0,
    out: str = "",
    backend: str = "fenicsx",
    cache: str = "",
    workers: int = 1,
    as_json: bool = False,
):
    from b3_core.cache import DiskCache
    from b3_core.fit.design import design, evaluate
    from b3_core.fit.spec import ParamSet

    try:
        paramset = _load_model(params, ParamSet)
        points = design(paramset, method=method, n=None if n == 0 else n, seed=seed)
        table = evaluate(
            path,
            paramset,
            points,
            backend=backend,
            cache=DiskCache(cache) if cache else None,
            workers=workers,
        )
        payload = {
            "table": table.model_dump(mode="json"),
            "params": paramset.model_dump(mode="json"),
        }
        if out:
            Path(out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _emit(payload, as_json=as_json)
    except Exception as exc:
        print(json.dumps(error_payload(exc)))
        raise SystemExit(1) from exc


def cmd_fit_rsm(
    table: str,
    targets: str,
    out: str = "",
    as_json: bool = False,
):
    from b3_core.fit.rsm import fit_quadratic, solve_rsm
    from b3_core.fit.spec import DesignTable, ParamSet, TargetSet

    try:
        payload = json.loads(Path(table).read_text(encoding="utf-8"))
        design_table = DesignTable.model_validate(payload["table"])
        paramset = ParamSet.model_validate(payload["params"])
        target_set = _load_model(targets, TargetSet)
        surface = fit_quadratic(design_table, target_set)
        start = solve_rsm(surface, target_set, paramset)
        result = {
            "surface": surface.model_dump(mode="json"),
            "x": [float(value) for value in start],
            "status": "rsm_only",
        }
        if out:
            Path(out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        _emit(result, as_json=as_json, status="rsm_only")
    except Exception as exc:
        print(json.dumps(error_payload(exc)))
        raise SystemExit(1) from exc


def cmd_fit_run(path: str, out: str = "", as_json: bool = False):
    from b3_core.fit.run import run_fit
    from b3_core.fit.spec import FitSpec

    try:
        result = run_fit(_load_model(path, FitSpec))
    except Exception as exc:
        print(json.dumps(error_payload(exc)))
        raise SystemExit(1) from exc
    if out:
        dest = Path(out)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "fit_result.json").write_text(
            json.dumps(result.model_dump(mode="json"), indent=2), encoding="utf-8"
        )
        (dest / "calibrated_case.json").write_text(
            json.dumps(result.calibrated_case, indent=2), encoding="utf-8"
        )
    _emit(result, as_json=as_json, status=result.status)


def fit_subgroup():
    targets = option(
        flags=["--targets"],
        arg_type=str,
        required=True,
        help="TargetSet JSON or YAML.",
    )
    params = option(
        flags=["--params"],
        arg_type=str,
        required=True,
        help="ParamSet JSON or YAML.",
    )
    backend = option(
        flags=["--backend"],
        arg_type=str,
        default="fenicsx",
        help="Pinned solver. Default fenicsx.",
    )
    return group(
        name="fit",
        help="Calibrate a case to a sparse datasheet. The skill chooses the free set.",
        commands=[
            command(
                name="bounds",
                help="Parameter catalogue, every parameter fixed.",
                callback=cmd_fit_bounds,
                arguments=[CASE_ARG],
                options=[
                    option(
                        flags=["--family"],
                        arg_type=str,
                        default="PVC",
                        help="Foam family.",
                    ),
                    option(
                        flags=["--rel"],
                        arg_type=float,
                        default=0.2,
                        help="Half-width of the scale bounds.",
                    ),
                    _JSON,
                ],
            ),
            command(
                name="residuals",
                help="Solve the case and compare it to the targets.",
                callback=cmd_fit_residuals,
                arguments=[CASE_ARG],
                options=[targets, backend, _JSON],
            ),
            command(
                name="estimate-halo",
                help="Halo width from density or uptake. No finite-element solve.",
                callback=cmd_fit_estimate_halo,
                arguments=[CASE_ARG],
                options=[
                    option(
                        flags=["--rho-infused"],
                        arg_type=float,
                        default=0.0,
                        help="Target infused density [kg/m³]. 0 means unused.",
                    ),
                    option(
                        flags=["--uptake"],
                        arg_type=float,
                        default=0.0,
                        help="Target resin uptake [kg/m²]. 0 means unused.",
                    ),
                    option(
                        flags=["--sd"],
                        arg_type=float,
                        default=0.0,
                        help="Absolute band added around the estimate. 0 means a point.",
                    ),
                    _JSON,
                ],
            ),
            command(
                name="refine",
                help="Bounded fit on true solves. Exit 1 unless the status is finished.",
                callback=cmd_fit_refine,
                arguments=[CASE_ARG],
                options=[
                    params,
                    targets,
                    backend,
                    option(
                        flags=["--cache"],
                        arg_type=str,
                        default="",
                        help="Disk cache directory.",
                    ),
                    _JSON,
                ],
            ),
            command(
                name="sensitivity",
                help="Two solves per free parameter. Reports weak and collinear names.",
                callback=cmd_fit_sensitivity,
                arguments=[CASE_ARG],
                options=[params, targets, backend, _JSON],
            ),
            command(
                name="design",
                help="Design points and one solve per point.",
                callback=cmd_fit_design,
                arguments=[CASE_ARG],
                options=[
                    params,
                    option(
                        flags=["--method"],
                        arg_type=str,
                        default="ccd",
                        help="ccd, sobol, or lhs.",
                    ),
                    option(
                        flags=["--n"],
                        arg_type=int,
                        default=0,
                        help="Point count. 0 keeps the method default.",
                    ),
                    option(
                        flags=["--seed"],
                        arg_type=int,
                        default=0,
                        help="Sampler seed.",
                    ),
                    option(
                        flags=["--out"],
                        arg_type=str,
                        default="",
                        help="Write the design table JSON here.",
                    ),
                    backend,
                    option(
                        flags=["--cache"],
                        arg_type=str,
                        default="",
                        help="Disk cache directory.",
                    ),
                    option(
                        flags=["--workers"],
                        arg_type=int,
                        default=1,
                        help="Process pool size. Children set OMP_NUM_THREADS=1.",
                    ),
                    _JSON,
                ],
            ),
            command(
                name="rsm",
                help="Quadratic surface from a design table. Status rsm_only.",
                callback=cmd_fit_rsm,
                arguments=[
                    argument(
                        name="table",
                        arg_type=str,
                        help="JSON from fit design (table and params).",
                    ),
                ],
                options=[
                    targets,
                    option(
                        flags=["--out"],
                        arg_type=str,
                        default="",
                        help="Write the surface JSON here.",
                    ),
                    _JSON,
                ],
            ),
            command(
                name="run",
                help="Chain the stages in a FitSpec. Writes calibrated_case.json.",
                callback=cmd_fit_run,
                arguments=[CASE_ARG],
                options=[
                    option(
                        flags=["--out"],
                        arg_type=str,
                        default="",
                        help="Directory for fit_result.json and calibrated_case.json.",
                    ),
                    _JSON,
                ],
            ),
        ],
    )
