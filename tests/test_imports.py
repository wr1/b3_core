"""Import boundaries: light package import and no private cross-imports."""

import ast
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "b3_core"


def test_deprecation_shims_import():
    """0.3 modules that only re-export must stay importable and measured."""
    import b3_core.io.runccx as runccx
    import b3_core.io.vts2ccx_planar as planar
    import b3_core.sweep.homogenise as homogenise

    assert callable(runccx.runccx)
    assert callable(planar.vtstoccx_planar)
    assert callable(homogenise.run_thickness)


def test_import_b3_core_does_not_load_heavy_stacks():
    code = (
        "import sys, b3_core\n"
        "banned = {'pyvista', 'mfem', 'vtk', 'matplotlib', 'frd2vtu'}\n"
        "hit = sorted(banned & set(sys.modules))\n"
        "print(','.join(hit))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == ""


def test_no_private_from_imports():
    bad: list[str] = []
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            module = node.module or ""
            if not module.startswith("b3_core."):
                continue
            for alias in node.names:
                if alias.name.startswith("_"):
                    bad.append(f"{path.relative_to(ROOT)}:{node.lineno} {alias.name}")
    assert bad == []


def test_viz_sweep_do_not_touch_private_attributes():
    roots = [SRC / "viz", SRC / "sweep", SRC / "physics_surrogate.py"]
    bad: list[str] = []
    for root in roots:
        paths = [root] if root.is_file() else root.rglob("*.py")
        for path in paths:
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1
            ):
                if re.search(r"\._[a-z]", line):
                    bad.append(f"{path.relative_to(ROOT)}:{lineno}")
    assert bad == []


def test_mesh_and_solver_entrypoints_stay_in_place():
    mesh_hits = []
    solver_hits = []
    for path in (ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(ROOT).as_posix()
        if "create_grooved_mesh(" in text:
            mesh_hits.append(rel)
        for needle in ("runnumpy(", "runmfem(", "runfenicsx("):
            if needle in text:
                solver_hits.append(rel)
    assert set(mesh_hits) == {
        "src/b3_core/core/mesh.py",
        "src/b3_core/pipeline.py",
    }
    assert set(solver_hits) <= {
        "src/b3_core/solvers/numpy_fe/backend.py",
        "src/b3_core/solvers/mfem.py",
        "src/b3_core/solvers/fenicsx.py",
        "src/b3_core/doctor.py",
        "src/b3_core/checks/bisect.py",
    }
