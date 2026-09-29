"""Numpy anisotropic homogeniser."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from b3_core.solvers.elasticity import (
    LOAD_CASES,
    constituent_dict,
    face_material_dict,
    material_C,
    properties_from_stiffness,
    scoring_payload,
)
from b3_core.solvers.numpy_fe.assembly import UNIT, canonicalize, homogenize_aniso
from b3_core.solvers.numpy_fe.halo_sampling import gauss_point_resin_P
from b3_core.solvers.protocol import (
    Capabilities,
    SolveRequest,
    SolveResult,
    as_solve_result,
)


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

    resin_cells = np.asarray(grid.cell_data["resin"], dtype=bool)
    face_cells = np.asarray(grid.cell_data["face"], dtype=bool)
    attr = np.ones(grid.n_cells, dtype=np.int64)
    attr[resin_cells] = 2
    if face is not None and face_cells.any():
        attr[face_cells] = 3

    C_core, C_resin = material_C(core), material_C(resin)
    C_face = None
    if (attr == 3).any():
        face_mat = dict(face) if face else {}
        face_mat.setdefault("E", 12_000_000_000.0)
        face_mat.setdefault("nu", 0.3)
        C_face = material_C(face_mat)

    # Neat per-Gauss-point stiffness, then the graded resin halo on foam cells.
    gp_C = np.broadcast_to(C_core, (grid.n_cells, 8, 6, 6)).copy()
    gp_C[attr == 2] = C_resin
    if C_face is not None:
        gp_C[attr == 3] = C_face
    if score_field is not None and getattr(score_field, "active", False):
        foam = np.flatnonzero(attr == 1)
        if len(foam):
            sampling = (scoring or {}).get("sampling") or {}
            P = gauss_point_resin_P(
                points,
                cells[foam],
                score_field,
                strategy=sampling.get("strategy", "exact"),
                resolution=int(sampling.get("resolution", 3)),
                idw_power=float(sampling.get("idw_power", 2.0)),
            )  # (n_foam, 8)
            p = P[:, :, None, None]
            gp_C[foam] = p * C_resin + (1.0 - p) * C_core

    stiffness, info = homogenize_aniso(points, cells, gp_C)
    properties, compliance = properties_from_stiffness(stiffness)

    if not return_details:
        return AnisoResult(properties, stiffness, compliance)

    displacements = {}
    master_of, W = info["master_of"], info["W"]
    for k, case in enumerate(LOAD_CASES):
        e0 = np.zeros((3, 3))
        v = UNIT[k]
        e0[0, 0], e0[1, 1], e0[2, 2] = v[0], v[1], v[2]
        e0[1, 2] = e0[2, 1] = 0.5 * v[3]
        e0[0, 2] = e0[2, 0] = 0.5 * v[4]
        e0[0, 1] = e0[1, 0] = 0.5 * v[5]
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
    )


class NumpyBackend:
    name = "numpy"
    capabilities = Capabilities(
        orthotropic=True,
        halo=True,
        face_layer=True,
        displacements=True,
        element_types=frozenset({"C3D8"}),
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
