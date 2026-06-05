"""Directional elastic-property maths from a 6x6 effective stiffness.

Self-contained (numpy only): converts a Voigt stiffness ``C`` (order
xx, yy, zz, yz, xz, xy — matching the MFEM backend) to a 4th-order compliance
tensor and evaluates direction-dependent engineering properties — the
information a single number like ``E_x`` hides. ``modulus_surface`` turns that
into a 3D surface that bulges along stiff directions: the clearest single view
of grooved-core anisotropy.
"""

from __future__ import annotations

import numpy as np

from b3_core.viz._deps import require_pyvista

# Voigt index -> tensor index pair, in the backend's (xx,yy,zz,yz,xz,xy) order.
_VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def compliance(C: np.ndarray) -> np.ndarray:
    """Compliance ``S = C^-1`` (6x6)."""
    return np.linalg.inv(np.asarray(C, dtype=float))


def _compliance_tensor4(S: np.ndarray) -> np.ndarray:
    """Lift a 6x6 Voigt compliance to the symmetric 3x3x3x3 tensor.

    Shear rows/cols carry the Reuss factors (1/2 per shear index) so that the
    full-tensor contraction reproduces the engineering-strain convention.
    """
    T = np.zeros((3, 3, 3, 3))
    for p, (i, j) in enumerate(_VOIGT):
        fp = 1.0 if p < 3 else 0.5
        for q, (k, m) in enumerate(_VOIGT):
            fq = 1.0 if q < 3 else 0.5
            v = S[p, q] * fp * fq
            for a, b in ((i, j), (j, i)):
                for c, d in ((k, m), (m, k)):
                    T[a, b, c, d] = v
    return T


def _unit(n: np.ndarray) -> np.ndarray:
    n = np.asarray(n, dtype=float)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def youngs_modulus(C: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Directional Young's modulus ``E(n)`` (Pa). ``n`` is ``(...,3)``."""
    S4 = _compliance_tensor4(compliance(C))
    n = _unit(n)
    denom = np.einsum("...i,...j,...k,...l,ijkl->...", n, n, n, n, S4)
    return 1.0 / denom


def linear_compressibility(C: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Directional linear compressibility ``beta(n)`` (1/Pa)."""
    S4 = _compliance_tensor4(compliance(C))
    n = _unit(n)
    return np.einsum("...i,...j,ijkk->...", n, n, S4)


def shear_modulus(C: np.ndarray, n: np.ndarray, m: np.ndarray) -> float:
    """Shear modulus ``G`` for the orthonormal direction pair ``(n, m)`` (Pa)."""
    S4 = _compliance_tensor4(compliance(C))
    n, m = _unit(n), _unit(m)
    return 1.0 / (4.0 * np.einsum("i,j,k,l,ijkl->", n, m, n, m, S4))


def poisson_ratio(C: np.ndarray, n: np.ndarray, m: np.ndarray) -> float:
    """Poisson's ratio for loading along ``n`` measured along ``m``."""
    S4 = _compliance_tensor4(compliance(C))
    n, m = _unit(n), _unit(m)
    lateral = np.einsum("i,j,k,l,ijkl->", n, n, m, m, S4)
    axial = np.einsum("i,j,k,l,ijkl->", n, n, n, n, S4)
    return -lateral / axial


def engineering_constants(C: np.ndarray) -> dict[str, float]:
    """Orthotropic engineering constants from the 6x6 stiffness (Pa / -)."""
    S = compliance(C)
    return {
        "E_x": 1.0 / S[0, 0], "E_y": 1.0 / S[1, 1], "E_z": 1.0 / S[2, 2],
        "G_yz": 1.0 / S[3, 3], "G_xz": 1.0 / S[4, 4], "G_xy": 1.0 / S[5, 5],
        "nu_xy": -S[1, 0] / S[0, 0], "nu_xz": -S[2, 0] / S[0, 0],
        "nu_yz": -S[2, 1] / S[1, 1],
    }


_PLANE_AXES = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}


def polar_modulus(C: np.ndarray, plane: str = "xy", n: int = 361):
    """In-plane directional Young's modulus ``E(theta)`` (Pa) for a plane.

    Returns ``(theta, E)`` with ``theta`` in radians over ``[0, 2pi]``.
    """
    u, v = _PLANE_AXES[plane]
    theta = np.linspace(0.0, 2.0 * np.pi, n)
    dirs = np.zeros((n, 3))
    dirs[:, u] = np.cos(theta)
    dirs[:, v] = np.sin(theta)
    return theta, youngs_modulus(C, dirs)


def modulus_surface(C: np.ndarray, *, kind: str = "E", resolution: int = 120):
    """A pyvista surface of a directional property over all directions.

    The unit sphere is warped so the radius in direction ``n`` is proportional to
    the property there (normalised to its max), and a ``value_GPa`` scalar carries
    the unscaled magnitude for colouring. ``kind`` is ``"E"`` (Young's modulus) or
    ``"beta"`` (linear compressibility).
    """
    pv = require_pyvista()
    sphere = pv.Sphere(radius=1.0, theta_resolution=resolution, phi_resolution=resolution)
    dirs = _unit(sphere.points)
    if kind == "E":
        vals = youngs_modulus(C, dirs)
    elif kind == "beta":
        vals = linear_compressibility(C, dirs)
    else:
        raise ValueError(f"kind must be 'E' or 'beta', got {kind!r}")
    vmax = float(np.nanmax(np.abs(vals))) or 1.0
    surf = sphere.copy()
    surf.points = dirs * (np.abs(vals)[:, None] / vmax)
    surf["value_GPa"] = vals / 1e9
    return surf
