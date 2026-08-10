"""Textile-as-code: named core textiles and modifiers as Python factories.

Primary way to define an RVE for homogenization — construct a :class:`Textile`
(or plain :class:`~b3_core.core.cprop.CpropInput`) in code, then pass it to
``homogenize`` / ``cprop``. JSON/YAML remain optional interchange for CLI and
frozen fixtures (``textile.to_json(path)``).

Example::

    from b3_core import homogenize
    from b3_core.cases import grid_scored, uniaxial

    r = homogenize(grid_scored(cell_size=0.6).with_curvature(kx=0.008))
    r2 = homogenize(uniaxial())
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from b3_core.core.cprop import CpropInput, Material

# ---------------------------------------------------------------------------
# Material presets (SI)
# ---------------------------------------------------------------------------

# Generic isotropic PVC foam (gallery / curved_panel examples)
PVC_100 = Material(E=130e6, nu=0.30, rho=100.0)
# Divinycell H60-class orthotropic card (Laustsen / DIAB GS30 notes)
H60_ORTHO = Material(
    E1=32e6,
    E2=32e6,
    E3=70e6,
    G12=19e6,
    G13=19e6,
    G23=19e6,
    nu12=0.3,
    nu13=0.3,
    nu23=0.3,
    rho=60.0,
)
EPOXY = Material(E=3e9, nu=0.35, rho=1100.0)
EPOXY_A = Material(E=3e9, nu=0.3, rho=1100.0)  # Laustsen "Resin A"


def _groove(offset: float, pitch: float, depth: float, width: float) -> list[float]:
    return [float(offset), float(pitch), float(depth), float(width)]


@dataclass(frozen=True)
class Textile:
    """Validated RVE case with fluent modifiers (textile-as-code).

    Holds a :class:`CpropInput`. Pass a ``Textile`` directly to ``homogenize``
    or ``cprop``. Use :meth:`to_json` only when you need a CLI snapshot.
    """

    input: CpropInput
    workdir: str | Path | None = None

    # -- fluent copy helpers -------------------------------------------------
    def with_curvature(self, kx: float = 0.0, ky: float = 0.0) -> Textile:
        return replace(
            self,
            input=self.input.model_copy(
                update={"curvature": {"kx": float(kx), "ky": float(ky)}}
            ),
        )

    def with_backend(
        self, backend: str, *, validate_with_ccx: bool | None = None
    ) -> Textile:
        upd: dict[str, Any] = {"backend": backend}
        if validate_with_ccx is not None:
            upd["validate_with_ccx"] = validate_with_ccx
        return replace(self, input=self.input.model_copy(update=upd))

    def with_halo(
        self,
        cell_size: float | dict,
        *,
        damage_cells: float = 1.0,
        face_scale: float = 0.25,
        face_enabled: bool = True,
        sampling: dict | None = None,
    ) -> Textile:
        """Attach resin-halo scoring (auto-routes solve to numpy)."""
        core = self.input.core.model_copy(update={"cell_size": cell_size})
        scoring: dict[str, Any] = {
            "damage_cells": damage_cells,
            "surfaces": {
                "saw_cut": {},
                "face": {"scale": face_scale, "enabled": face_enabled},
            },
            "sampling": sampling or {"strategy": "local_cloud", "resolution": 3},
        }
        return replace(
            self,
            input=self.input.model_copy(
                update={"core": core, "scoring": scoring, "backend": "numpy"}
            ),
        )

    def with_thickness(
        self, thickness: float, *, ligament: float | None = None
    ) -> Textile:
        """Set thickness; optionally keep a foam ligament (top-mouth grooves).

        If *ligament* is set and the first x-groove has negative depth, depth is
        rewritten as ``-(thickness - ligament)`` (curved-panel convention).
        """
        t = float(thickness)
        xgr = [list(g) for g in self.input.xgr]
        ygr = [list(g) for g in self.input.ygr]
        if ligament is not None and xgr and xgr[0][2] < 0:
            depth = -(t - float(ligament))
            xgr = [[g[0], g[1], depth, g[3]] for g in xgr]
        return replace(
            self,
            input=self.input.model_copy(
                update={"thickness": t, "xgr": xgr, "ygr": ygr}
            ),
        )

    def with_workdir(self, path: str | Path) -> Textile:
        return replace(self, workdir=path)

    def model_dump(self, **kwargs) -> dict:
        return self.input.model_dump(**kwargs)

    def to_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write a JSON snapshot for CLI / git fixtures (optional interchange)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.model_dump(), indent=indent) + "\n", encoding="utf-8"
        )
        return p


def _textile(
    *,
    dx: float,
    dy: float,
    thickness: float,
    xgr: list[list[float]],
    ygr: list[list[float]],
    core: Material,
    resin: Material,
    madd: list[float] | None = None,
    backend: str = "mfem",
    validate_with_ccx: bool = False,
    face: dict | None = None,
    curvature: dict | None = None,
    scoring: dict | None = None,
) -> Textile:
    inp = CpropInput(
        dx=dx,
        dy=dy,
        thickness=thickness,
        xgr=xgr,
        ygr=ygr,
        core=core,
        resin=resin,
        madd=list(madd if madd is not None else [0]),
        face=dict(face or {}),
        curvature=dict(curvature or {}),
        scoring=dict(scoring or {}),
        backend=backend,
        validate_with_ccx=validate_with_ccx,
    )
    return Textile(input=inp)


# ---------------------------------------------------------------------------
# Named textiles (mirror shipped examples)
# ---------------------------------------------------------------------------


def plain(
    *,
    dx: float = 50.0,
    dy: float = 50.0,
    thickness: float = 30.0,
    core: Material | None = None,
    resin: Material | None = None,
    backend: str = "mfem",
) -> Textile:
    """Ungrooved homogeneous core (baseline RVE)."""
    return _textile(
        dx=dx,
        dy=dy,
        thickness=thickness,
        xgr=[],
        ygr=[],
        core=core or PVC_100,
        resin=resin or EPOXY,
        madd=[0],
        backend=backend,
    )


def uniaxial(
    *,
    dx: float = 50.0,
    dy: float = 50.0,
    thickness: float = 30.0,
    pitch: float = 10.0,
    depth: float = 8.0,
    width: float = 3.0,
    offset: float = 10.0,
    core: Material | None = None,
    resin: Material | None = None,
    backend: str = "mfem",
    validate_with_ccx: bool = False,
) -> Textile:
    """Single x-family grooves (``examples/mfem_patterns/uniaxial``)."""
    return _textile(
        dx=dx,
        dy=dy,
        thickness=thickness,
        xgr=[_groove(offset, pitch, depth, width)],
        ygr=[],
        core=core or PVC_100,
        resin=resin or EPOXY,
        madd=[-0.3, 0, 0.3],
        backend=backend,
        validate_with_ccx=validate_with_ccx,
    )


def crossed(
    *,
    dx: float = 50.0,
    dy: float = 50.0,
    thickness: float = 30.0,
    pitch: float = 10.0,
    depth: float = 8.0,
    width: float = 3.0,
    offset: float = 10.0,
    core: Material | None = None,
    resin: Material | None = None,
    backend: str = "mfem",
    validate_with_ccx: bool = False,
) -> Textile:
    """Symmetric x+y groove families (``examples/mfem_patterns/crossed``)."""
    g = _groove(offset, pitch, depth, width)
    return _textile(
        dx=dx,
        dy=dy,
        thickness=thickness,
        xgr=[g],
        ygr=[list(g)],
        core=core or PVC_100,
        resin=resin or EPOXY,
        madd=[-0.3, 0, 0.3],
        backend=backend,
        validate_with_ccx=validate_with_ccx,
    )


def two_sided(
    *,
    dx: float = 50.0,
    dy: float = 50.0,
    thickness: float = 30.0,
    core: Material | None = None,
    resin: Material | None = None,
    backend: str = "mfem",
    validate_with_ccx: bool = False,
) -> Textile:
    """Top x-grooves + opposite-sign deep y-grooves (``mfem_patterns/two_sided``)."""
    return _textile(
        dx=dx,
        dy=dy,
        thickness=thickness,
        xgr=[_groove(10, 10, 8, 3)],
        ygr=[
            _groove(-2, 25, 17.0, 2.0),
            _groove(2, 25, -17.0, 2.0),
        ],
        core=core or PVC_100,
        resin=resin or EPOXY,
        madd=[-0.3, 0, 0.3],
        backend=backend,
        validate_with_ccx=validate_with_ccx,
    )


def curved_panel(
    *,
    dx: float = 50.0,
    dy: float = 50.0,
    thickness: float = 30.0,
    ligament: float = 3.0,
    pitch: float = 10.0,
    width: float = 3.0,
    offset: float = 10.0,
    kx: float = 0.0,
    ky: float = 0.0,
    core: Material | None = None,
    resin: Material | None = None,
    backend: str = "mfem",
) -> Textile:
    """Deep top-mouth x-grooves for mould curvature (``examples/curved_panel``).

    Depth is ``-(thickness - ligament)`` so a foam floor remains at the mould face.
    ``kx > 0`` opens, ``kx < 0`` closes (same sign convention as the docs).
    """
    depth = -(float(thickness) - float(ligament))
    return _textile(
        dx=dx,
        dy=dy,
        thickness=thickness,
        xgr=[_groove(offset, pitch, depth, width)],
        ygr=[],
        core=core or PVC_100,
        resin=resin or EPOXY,
        madd=[-0.4, -0.2, 0, 0.2, 0.4],
        backend=backend,
        curvature={"kx": float(kx), "ky": float(ky)},
    )


def grid_scored(
    *,
    pitch: float = 30.0,
    thickness: float = 20.0,
    depth: float = 18.0,
    width: float = 1.0,
    cell_size: float | dict | None = 0.6,
    core: Material | None = None,
    resin: Material | None = None,
    with_halo: bool = True,
) -> Textile:
    """DIAB-style grid-scored foam (``examples/diab_gs30_scored``).

    Orthotropic H60-class core + epoxy; numpy backend when halo is on.
    Pass ``cell_size=None`` and ``with_halo=False`` for sharp kerfs only
    (``diab_gs30`` without scoring).
    """
    dx = dy = float(pitch)
    t = _textile(
        dx=dx,
        dy=dy,
        thickness=thickness,
        xgr=[_groove(0, pitch, depth, width)],
        ygr=[_groove(0, pitch, depth, width)],
        core=core or H60_ORTHO,
        resin=resin or EPOXY_A,
        madd=[-0.15, 0, 0.15],
        backend="numpy",
        validate_with_ccx=False,
    )
    if with_halo and cell_size is not None:
        return t.with_halo(cell_size)
    if cell_size is not None and not with_halo:
        core_m = t.input.core.model_copy(update={"cell_size": cell_size})
        return replace(t, input=t.input.model_copy(update={"core": core_m}))
    return t


def from_dict(data: dict) -> Textile:
    """Validate a dict (e.g. loaded JSON) as a Textile."""
    payload = {k: v for k, v in data.items() if not str(k).startswith("_")}
    return Textile(input=CpropInput(**payload))


def from_path(path: str | Path) -> Textile:
    """Load JSON/YAML file as a Textile (interchange → code)."""
    from b3_core.core.cprop import load_case

    dct, dirname = load_case(str(path))
    return Textile(
        input=CpropInput(
            **{k: v for k, v in dct.items() if not str(k).startswith("_")}
        ),
        workdir=dirname or None,
    )


__all__ = [
    "Textile",
    "PVC_100",
    "H60_ORTHO",
    "EPOXY",
    "EPOXY_A",
    "plain",
    "uniaxial",
    "crossed",
    "two_sided",
    "curved_panel",
    "grid_scored",
    "from_dict",
    "from_path",
]
