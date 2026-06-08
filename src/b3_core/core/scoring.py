"""Stochastic resin halo for grid-scored cores.

Grid-scoring (knife/saw) opens foam cells along each cut; opened cells fill with
resin, closed cells stay foam. Cell size is a distribution, so the probability
that a material point holds resin is the survival function of that distribution
evaluated at the point's distance to the nearest cut surface:

    P(x) = S( distance(x -> nearest groove wall/root) )      P=1 at the cut, ->0 at reach

Local stiffness is then a rule of mixtures ``P*C_resin + (1-P)*C_foam`` evaluated
at integration points by the numpy backend. Cut surfaces are the groove walls and
root only — the un-sawn outer top/bottom faces emit no halo (no groove there).
"""

from __future__ import annotations

import numpy as np


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


class ScoreField:
    """Resin-presence probability field for a grid-scored RVE.

    Rebuilds the nominal groove rectangles from the input spec and evaluates the
    survival-function probability at arbitrary points (mm). Independent of the FE
    mesh, so it can be sampled at integration points.
    """

    def __init__(self, inp: dict):
        from b3_core.core.mesh import create_grooves

        thk = float(inp["thickness"])
        kx = (inp.get("curvature") or {}).get("kx", 0.0)
        ky = (inp.get("curvature") or {}).get("ky", 0.0)
        # nominal groove geometry (meshadd=[0] -> no FE-refinement duplicates)
        bx, tx, hx, sx = create_grooves(inp["xgr"], float(inp["dx"]), meshadd=[0.0], kappa=kx)
        by, ty, hy, sy = create_grooves(inp["ygr"], float(inp["dy"]), meshadd=[0.0], kappa=ky)

        self.grooves = []  # (axis, centre, half_width, slope, depth)
        for axis, (b, t, h, s) in ((0, (bx, tx, hx, sx)), (1, (by, ty, hy, sy))):
            for c0, hw, d, sl in zip(0.5 * (b + t), 0.5 * (t - b), h, s, strict=True):
                if hw > 0 and abs(d) > 0:   # skip the no-groove placeholder bands
                    self.grooves.append((axis, float(c0), float(hw), float(sl), float(d)))
        self.thickness = thk
        self.cell_size = (inp.get("core") or {}).get("cell_size")
        self.S, self.reach = survival(self.cell_size)

    @property
    def active(self) -> bool:
        return bool(self.grooves) and self.reach > 0.0

    def distance_to_cut(self, points: np.ndarray) -> np.ndarray:
        """Distance (mm) from each point to the nearest groove wall/root."""
        pts = np.asarray(points, dtype=float)
        z = pts[:, 2]
        d_min = np.full(len(pts), np.inf)
        for axis, c0, hw0, slope, depth in self.grooves:
            if depth > 0:
                z0, z1, zeta = 0.0, depth, depth - z
            else:
                z0, z1, zeta = self.thickness + depth, self.thickness, z - (self.thickness + depth)
            hw = np.clip(hw0 + slope * zeta, 0.0, None)
            du = np.maximum(0.0, np.abs(pts[:, axis] - c0) - hw)
            dz = np.maximum(0.0, np.maximum(z0 - z, z - z1))
            d_min = np.minimum(d_min, np.hypot(du, dz))
        return d_min

    def resin_probability(self, points: np.ndarray) -> np.ndarray:
        """P(resin) in [0,1] at each point (0 when the field is inactive)."""
        if not self.active:
            return np.zeros(len(np.asarray(points)))
        return self.S(self.distance_to_cut(points))


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
