"""Lazy, centralised access to optional rendering dependencies.

Keeps the headless-GL bootstrap (``pv.start_xvfb``) and the optional trame stack
in one place instead of the try/except blocks that were duplicated across every
render script.
"""

from __future__ import annotations

import functools
import logging

logger = logging.getLogger(__name__)


def require_pyvista():
    """Import and return pyvista, or raise a clear error."""
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover - pyvista is a core dep
        raise RuntimeError("pyvista is required for 3D visualization") from exc
    return pv


@functools.lru_cache(maxsize=1)
def ensure_headless() -> None:
    """Start a virtual framebuffer for off-screen GL, once per process.

    No-op (and never raises) when a display is already present or xvfb is
    unavailable — pyvista will fall back to whatever GL context exists.
    """
    pv = require_pyvista()
    try:
        pv.start_xvfb()
    except Exception as exc:  # pragma: no cover - display already present / unsupported
        logger.debug("start_xvfb skipped: %s", exc)


def require_trame() -> None:
    """Raise a helpful error if the optional trame stack is not installed."""
    import importlib.util

    missing = [m for m in ("trame", "trame_vtk") if importlib.util.find_spec(m) is None]
    if missing:
        raise RuntimeError(
            "interactive web viewer needs the trame stack; install it with "
            "`uv sync --extra interactive` (or `pip install b3_core[interactive]`)"
        )
