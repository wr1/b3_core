"""Numpy anisotropic homogeniser."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from b3_core.solvers.elasticity import (
    LOAD_CASES,
    constituent_dict,
    face_material_dict,
    properties_from_stiffness,
    scoring_payload,
)
from b3_core.solvers.numpy_fe.assembly import (
    SHAPE_N,
    UNIT,
    canonicalize,
    homogenize_aniso,
)
from b3_core.solvers.protocol import (
    Capabilities,
    SolveRequest,
    SolveResult,
    as_solve_result,
)
from b3_core.solvers.sampling import per_gp_stiffness, phase_attributes


@dataclass(frozen=True)
class AnisoResult:
    properties: dict
    stiffness: np.ndarray
    compliance: np.ndarray
    displacements: dict | None = None
    points: np.ndarray | None = None
    # Per-element total strain under each unit load case (n_elem, 6 cases, 6 Voigt),
    # element attribute and volume — inputs to the resin failure check.
    elem_strain: np.ndarray | None = None
    elem_attr: np.ndarray | None = None
    elem_volume: np.ndarray | None = None
    raw_stiffness: np.ndarray | None = None


def runnumpy(
    mesh,
    resin,
    core,
    face=None,
    *,
    score_field=None,
    scoring=None,
    return_details=False,
):
    """Drop-in for runmfem using the numpy anisotropic homogeniser.

    Accepts isotropic or orthotropic material dicts. With a ``score_field``
    (stochastic resin halo), foam Gauss points get a rule-of-mixtures stiffness
    ``P*C_resin + (1-P)*C_foam``, P = the field's resin probability; the sampling
    strategy ("exact" / "local_cloud" IDW) comes from ``scoring['sampling']``.
    Mirrors the MFEM result and adds the per-element strain field for failure.
    """
    grid = mesh.scale((1e-3, 1e-3, 1e-3), inplace=False)
    if hasattr(grid, "cast_to_unstructured_grid"):
        grid = grid.cast_to_unstructured_grid()
    cell_block = grid.cells.reshape((-1, 9))
    if not np.all(cell_block[:, 0] == 8):
        raise ValueError("numpy backend supports linear hexahedral cells only")
    points = np.asarray(grid.points, dtype=np.float64)
    cells = canonicalize(points, np.asarray(cell_block[:, 1:], dtype=np.int64))

    attr = phase_attributes(grid, face)
    # Shared constitutive map: neat phases, then the graded resin halo on foam
    # cells (identical to the MFEM backend via b3_core.solvers.sampling).
    gp_mm = np.einsum("gn,enj->egj", SHAPE_N, points[cells]) * 1000.0
    gp_C = per_gp_stiffness(
        attr,
        gp_mm,
        core=core,
        resin=resin,
        face=face,
        score_field=score_field,
        scoring=scoring,
        points_m=points,
        cells=cells,
    )

    stiffness, info = homogenize_aniso(points, cells, gp_C)
    properties, compliance = properties_from_stiffness(stiffness)
    raw = np.asarray(info["raw_stiffness"], dtype=float)

    if not return_details:
        return AnisoResult(properties, stiffness, compliance, raw_stiffness=raw)

    displacements = {}
    W = info["W"]
    w_nodal = info.get("w_nodal")
    master_of = info["master_of"]
    for k, case in enumerate(LOAD_CASES):
        e0 = np.zeros((3, 3))
        v = UNIT[k]
        e0[0, 0], e0[1, 1], e0[2, 2] = v[0], v[1], v[2]
        e0[1, 2] = e0[2, 1] = 0.5 * v[3]
        e0[0, 2] = e0[2, 0] = 0.5 * v[4]
        e0[0, 1] = e0[1, 0] = 0.5 * v[5]
        if w_nodal is not None:
            w_node = w_nodal[:, :, k]
        else:
            w_node = W[3 * master_of[:, None] + np.array([0, 1, 2]), k]
        displacements[case] = points @ e0 + w_node
    return AnisoResult(
        properties,
        stiffness,
        compliance,
        displacements,
        points,
        info["elem_strain"],
        attr,
        info["vol"],
        raw,
    )


class NumpyBackend:
    name = "numpy"
    capabilities = Capabilities(
        orthotropic=True,
        halo=True,
        face_layer=True,
        displacements=True,
        element_types=frozenset({"C3D8"}),
        interpolated_periodicity=True,
    )

    def is_available(self) -> bool:
        return True

    def solve(self, req: SolveRequest) -> SolveResult:
        result = runnumpy(
            req.mesh,
            constituent_dict(req.resin),
            constituent_dict(req.core),
            face_material_dict(req.face),
            score_field=req.score_field,
            scoring=scoring_payload(req.scoring),
            return_details=True,
        )
        return as_solve_result(result, details=req.details)
