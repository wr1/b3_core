"""Stochastic resin halo for grid-scored cores.

Grid-scoring (knife/saw) opens foam cells along saw-cut surfaces; unsawn
top/bottom faces keep closed cells. Cell size is a distribution per surface
type, so the probability that a material point holds resin is the survival
function evaluated at distance to the nearest relevant cut surface:

    P(x) = max( S_saw(d_to_groove), S_face(d_to_outer_face) )

Local stiffness is a rule of mixtures ``P*C_resin + (1-P)*C_foam`` at foam
Gauss points (numpy backend).
"""

from __future__ import annotations

from typing import Any

import numpy as np

_DEFAULT_FACE_SCALE = 0.25


def survival(cell_size):
    """Return ``(S, reach)`` for a cell-size spec.

    - scalar ``cs``: uniform [0, cs] -> ``S(d) = 1 - d/cs`` (linear), ``reach=cs``.
    - dict ``{mean, std, dist}``: ``S`` is the survival function of a ``lognormal``
      (default) or ``normal`` cell-size distribution (renormalised so ``S(0)=1``);
      ``reach`` is the ~max cell size (ppf 0.999 / mean+3*std).
    - ``None``: no halo (``S=0``, ``reach=0``).
    """
    if cell_size is None:
        return (lambda d: np.zeros(np.shape(d)), 0.0)
    if not isinstance(cell_size, dict):
        cs = float(cell_size)
        return (lambda d: np.clip(1.0 - np.asarray(d, float) / cs, 0.0, 1.0), cs)

    mean = float(cell_size["mean"])
    std = float(cell_size.get("std", 0.0))
    if std <= 0.0:
        return (lambda d: np.clip(1.0 - np.asarray(d, float) / mean, 0.0, 1.0), mean)

    from scipy import stats

    dist = cell_size.get("dist", "lognormal")
    if dist == "normal":
        rv = stats.norm(mean, std)
        reach = mean + 3.0 * std
    elif dist == "lognormal":
        sigma = float(np.sqrt(np.log(1.0 + (std / mean) ** 2)))
        mu = float(np.log(mean) - 0.5 * sigma**2)
        rv = stats.lognorm(s=sigma, scale=np.exp(mu))
        reach = float(rv.ppf(0.999))
    else:
        raise ValueError(f"cell_size dist must be 'lognormal' or 'normal', got {dist!r}")

    s0 = float(rv.sf(0.0)) or 1.0
    return (lambda d: np.clip(rv.sf(np.asarray(d, float)) / s0, 0.0, 1.0), float(reach))


def _scale_cell_size(cell_size: float | dict[str, Any], scale: float):
    """Scale a cell-size spec by ``scale`` (for thinner face halos)."""
    if cell_size is None or scale <= 0.0:
        return None
    if not isinstance(cell_size, dict):
        return float(cell_size) * scale
    out = dict(cell_size)
    out["mean"] = float(out["mean"]) * scale
    if "std" in out:
        out["std"] = float(out["std"]) * scale
    return out


def parse_surface_halo(inp: dict) -> dict[str, dict[str, Any]]:
    """Resolve per-surface halo specs from ``core.cell_size`` and ``scoring.surfaces``.

    Returns ``{saw_cut: {cell_size, enabled, S, reach}, face: {...}}``.
    """
    core_cs = (inp.get("core") or {}).get("cell_size")
    scoring = inp.get("scoring") or {}
    surfaces_in = scoring.get("surfaces") or {}
    saw_cfg = surfaces_in.get("saw_cut") or {}
    face_cfg = surfaces_in.get("face") or {}

    saw_cs = saw_cfg["cell_size"] if "cell_size" in saw_cfg else core_cs
    saw_enabled = saw_cfg.get("enabled", True) and saw_cs is not None
    if not saw_enabled:
        saw_cs = None

    face_enabled = face_cfg.get("enabled", True)
    if "cell_size" in face_cfg:
        face_cs = face_cfg["cell_size"]
        if face_cs is None:
            face_enabled = False
    elif face_enabled and saw_cs is not None:
        scale = float(face_cfg.get("scale", _DEFAULT_FACE_SCALE))
        face_cs = _scale_cell_size(saw_cs, scale)
    else:
        face_cs = None

    if not face_enabled:
        face_cs = None

    out: dict[str, dict[str, Any]] = {}
    for key, cs, enabled in (
        ("saw_cut", saw_cs, saw_enabled),
        ("face", face_cs, face_enabled and face_cs is not None),
    ):
        s_fn, reach = survival(cs) if enabled and cs is not None else (lambda d: np.zeros(np.shape(d)), 0.0)
        out[key] = {
            "cell_size": cs,
            "enabled": enabled and reach > 0.0,
            "S": s_fn,
            "reach": reach,
        }
    return out


class ScoreField:
    """Resin-presence probability field for a grid-scored RVE.

    Evaluates per-surface survival functions at arbitrary points (mm), independent
    of the FE mesh, for sampling at integration points.
    """

    def __init__(self, inp: dict):
        from b3_core.core.mesh import create_grooves

        thk = float(inp["thickness"])
        kx = (inp.get("curvature") or {}).get("kx", 0.0)
        ky = (inp.get("curvature") or {}).get("ky", 0.0)
        bx, tx, hx, sx = create_grooves(inp["xgr"], float(inp["dx"]), meshadd=[0.0], kappa=kx)
        by, ty, hy, sy = create_grooves(inp["ygr"], float(inp["dy"]), meshadd=[0.0], kappa=ky)

        self.grooves = []
        for axis, (b, t, h, s) in ((0, (bx, tx, hx, sx)), (1, (by, ty, hy, sy))):
            for c0, hw, d, sl in zip(0.5 * (b + t), 0.5 * (t - b), h, s, strict=True):
                if hw > 0 and abs(d) != 0:
                    self.grooves.append((axis, float(c0), float(hw), float(sl), float(d)))
        self.thickness = thk
        self.surfaces = parse_surface_halo(inp)
        # Legacy aliases (saw-cut only).
        saw = self.surfaces["saw_cut"]
        self.cell_size = saw["cell_size"]
        self.S = saw["S"]
        self.reach = max(
            (s["reach"] for s in self.surfaces.values() if s["enabled"]),
            default=0.0,
        )

    @property
    def active(self) -> bool:
        saw = self.surfaces["saw_cut"]
        face = self.surfaces["face"]
        if saw["enabled"] and self.grooves:
            return True
        return face["enabled"]

    def distance_to_saw_cut(self, points: np.ndarray) -> np.ndarray:
        """Distance (mm) from each point to the nearest groove wall/root.

        Wall half-width uses the same root-hinged ``hw(z)`` law as the mesh
        morph (incl. pinch clamp), so the halo grades off the *open/closed*
        kerf surface. Evaluate at **physical** post-morph coordinates — after
        the interval-affine morph, walls sit at ``c0 ± hw(z)``.
        """
        from b3_core.core.mesh import _MIN_HW

        pts = np.asarray(points, dtype=float)
        z = pts[:, 2]
        d_min = np.full(len(pts), np.inf)
        th = self.thickness
        for axis, c0, hw0, slope, depth in self.grooves:
            if depth > 0:
                z0, z1 = 0.0, depth
                inside = (z >= z0) & (z <= z1)
                zeta = depth - z
            else:
                z0, z1 = th + depth, th
                inside = (z >= z0) & (z <= z1)
                zeta = z - (th + depth)
            # Same law as mesh._hw_at (root-hinged taper + pinch floor).
            hw = np.where(
                inside,
                np.maximum(_MIN_HW, hw0 + slope * zeta),
                hw0,
            )
            du = np.maximum(0.0, np.abs(pts[:, axis] - c0) - hw)
            dz = np.maximum(0.0, np.maximum(z0 - z, z - z1))
            d_min = np.minimum(d_min, np.hypot(du, dz))
        return d_min

    def distance_to_face(self, points: np.ndarray) -> np.ndarray:
        """Distance (mm) from each point to the nearest unsawn top/bottom face."""
        pts = np.asarray(points, dtype=float)
        z = pts[:, 2]
        return np.minimum(z, self.thickness - z)

    def distance_to_cut(self, points: np.ndarray) -> np.ndarray:
        """Backward-compatible alias for saw-cut distance only."""
        return self.distance_to_saw_cut(points)

    def resin_probability(self, points: np.ndarray) -> np.ndarray:
        """P(resin) in [0,1] at each point (max over active surface halos)."""
        if not self.active:
            return np.zeros(len(np.asarray(points)))
        pts = np.asarray(points, dtype=float)
        p = np.zeros(len(pts))
        saw = self.surfaces["saw_cut"]
        if saw["enabled"] and self.grooves:
            p = np.maximum(p, saw["S"](self.distance_to_saw_cut(pts)))
        face = self.surfaces["face"]
        if face["enabled"]:
            p = np.maximum(p, face["S"](self.distance_to_face(pts)))
        return p


def effective_resin_vf(mesh, field, resin_vf: float):
    """``(effective_resin_vf, halo_vf)``: neat resin plus the halo's resin content.

    Adds the volume integral of P(resin) over the **foam** cells (evaluated at
    cell centres) to the neat-kerf ``resin_vf``. Returns the inputs unchanged when
    no field is active.
    """
    if field is None or not field.active:
        return resin_vf, 0.0
    ma = mesh.compute_cell_sizes()
    vol = np.abs(ma.cell_data["Volume"])
    foam = ~(np.asarray(mesh.cell_data["resin"], bool)
             | np.asarray(mesh.cell_data["face"], bool))
    p = field.resin_probability(mesh.cell_centers().points)
    halo_vf = float((p[foam] * vol[foam]).sum() / vol.sum())
    return resin_vf + halo_vf, halo_vf