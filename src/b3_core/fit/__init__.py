"""Calibration primitives. The skill decides which parameters are free."""

from b3_core.fit.design import design, evaluate
from b3_core.fit.identify import sensitivity
from b3_core.fit.materials import estimate_foam, resin_typical, split_basis
from b3_core.fit.params import apply, default_bounds
from b3_core.fit.refine import refine
from b3_core.fit.residuals import residuals
from b3_core.fit.rsm import fit_quadratic, solve_rsm
from b3_core.fit.run import run_fit
from b3_core.fit.volume import estimate_halo

__all__ = [
    "apply",
    "default_bounds",
    "design",
    "estimate_foam",
    "estimate_halo",
    "evaluate",
    "fit_quadratic",
    "refine",
    "resin_typical",
    "residuals",
    "run_fit",
    "sensitivity",
    "solve_rsm",
    "split_basis",
]
