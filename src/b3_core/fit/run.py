"""Chain the stages named on a FitSpec. This adds no free-parameter policy."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from b3_core.fit.design import design, evaluate
from b3_core.fit.identify import sensitivity
from b3_core.fit.refine import refine
from b3_core.fit.rsm import fit_quadratic, solve_rsm
from b3_core.fit.spec import FitResult, FitSpec
from b3_core.fit.volume import estimate_halo
from b3_core.loaders import normalize_case


def _stage(stages: dict, name: str) -> Any:
    if name not in stages:
        return None
    return stages[name]


def _cache(spec: FitSpec) -> Any:
    if not spec.cache:
        return None
    from b3_core.cache import DiskCache

    return DiskCache(spec.cache)


def _apply_halo(spec: FitSpec, messages: list[str]) -> None:
    rho = next(
        (
            target.value
            for target in spec.targets.targets
            if target.quantity == "rho_infused" and target.value
        ),
        None,
    )
    uptake = next(
        (
            target.value
            for target in spec.targets.targets
            if target.quantity == "resin_uptake_kg_m2" and target.value
        ),
        None,
    )
    if rho is None and uptake is None:
        messages.append("halo_from_mass skipped: no density or uptake target")
        return
    estimate = estimate_halo(
        spec.case,
        rho_infused=rho,
        uptake_kg_m2=uptake,
    )
    for param in spec.params.params:
        if param.name != "halo_cell_size":
            continue
        param.value = float(estimate.value)
        param.lower = float(estimate.lower)
        param.upper = float(estimate.upper)
    messages.append(
        "halo_from_mass "
        f"value={estimate.value:.4g} feasible={estimate.feasible} "
        f"interval=[{estimate.lower:.4g}, {estimate.upper:.4g}]"
    )


def run_fit(spec: FitSpec, *, solve: Any = None) -> FitResult:
    """halo → design → rsm → refine → identify, only for stages that are set."""
    case_in, _workdir = normalize_case(spec.case)
    messages: list[str] = []
    stages = dict(spec.stages or {})
    if _stage(stages, "halo_from_mass"):
        _apply_halo(spec, messages)
    cache = _cache(spec)
    surface = None
    x0 = None
    design_stage = _stage(stages, "design")
    if design_stage and spec.params.free():
        method = "ccd"
        count = None
        seed = 0
        if isinstance(design_stage, dict):
            method = str(design_stage.get("method") or "ccd")
            count = design_stage.get("n")
            seed = int(design_stage.get("seed") or 0)
        points = design(spec.params, method=method, n=count, seed=seed)
        table = evaluate(
            case_in,
            spec.params,
            points,
            backend=spec.backend,
            cache=cache,
            workers=int(spec.workers),
            solve=solve,
        )
        if _stage(stages, "rsm") is not False and table.rows:
            options = stages.get("rsm") if isinstance(stages.get("rsm"), dict) else {}
            options = options or {}
            surface = fit_quadratic(
                table,
                spec.targets,
                log=bool(options.get("log", True)),
                ridge=float(options.get("ridge", 1e-2)),
            )
            x0 = solve_rsm(surface, spec.targets, spec.params)
            messages.append(f"rsm start={np.asarray(x0).tolist()}")
    refine_stage = stages.get("refine", {})
    if refine_stage is False:
        if surface is None:
            raise ValueError("refine is off and no response surface was built")
        from b3_core.fit.params import apply, values_of
        from b3_core.fit.refine import fit_provenance, mark_calibrated
        from b3_core.fit.residuals import residuals

        record = (solve or _default)(
            apply(case_in, spec.params, x0), backend=spec.backend, cache=cache
        )
        report = residuals(record, spec.targets)
        result = FitResult(
            status="rsm_only",
            params=values_of(spec.params, x0),
            residuals=report,
            rsm=surface.model_dump(mode="json"),
            solves={"true": len(table.rows), "cache_hits": 0, "wall_s": 0.0},
            calibrated_case=mark_calibrated(apply(case_in, spec.params, x0)).model_dump(
                mode="json"
            ),
            provenance=fit_provenance(case_in, spec.backend),
            messages=messages,
        )
    else:
        options = refine_stage if isinstance(refine_stage, dict) else {}
        result = refine(
            case_in,
            spec.params,
            spec.targets,
            x0,
            backend=spec.backend,
            cache=cache,
            tol_sigma=float(options.get("tol_sigma", 1.0)),
            max_solves=int(options.get("max_solves", 30)),
            jac="fd",
            solve=solve,
        )
        if surface is not None:
            result = result.model_copy(update={"rsm": surface.model_dump(mode="json")})
        result.messages = [*messages, *result.messages]
    identify = stages.get("identify")
    if identify and spec.params.free() and result.status != "no_free_params":
        step = 0.02
        if isinstance(identify, dict) and identify.get("step"):
            step = float(identify["step"])
        report = sensitivity(
            case_in,
            spec.params,
            spec.targets,
            step=step,
            solve=solve,
            backend=spec.backend,
            cache=cache,
        )
        result = result.model_copy(update={"identifiability": report})
    digest = hashlib.sha256(
        json.dumps(spec.model_dump(mode="json"), sort_keys=True).encode()
    ).hexdigest()
    provenance = dict(result.provenance)
    provenance["fitspec_hash"] = digest
    return result.model_copy(update={"provenance": provenance})


def _default(case: Any, *, backend: str, cache: Any) -> Any:
    from b3_core.api import run_case

    return run_case(case, backend=backend, cache=cache)
