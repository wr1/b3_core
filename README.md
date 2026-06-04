# b3_core

Homogenized elastic properties for sandwich-panel cores with sawcuts and
machined grooves. The cuts fill with resin during infusion, altering the
effective stiffness and density of the core; this package predicts the
resulting orthotropic material and returns it as a `b3_mat.OrthotropicMaterial`
so it drops into the wider b3 section / beam pipeline.

## Approach

Structured 3D RVE mesh (PyVista) with cell-level tagging of core vs.
resin-filled groove. Six periodic-BC load cases (εxx, εyy, εzz, εxy, εxz, εyz)
run in parallel and the averaged reactions give the effective engineering
constants (Ex/Ey/Ez, Gxy/Gxz/Gyz, Poisson ratios) and infused density. Grooves
can be tapered by mould curvature (`kx`, `ky`) for curved panels.

Backends: **CalculiX** (default), plus optional **FEniCSx** and **MFEM**; set
`"validate_with_ccx": true` to cross-check a backend against ccx.

## Requirements

- Python ≥ 3.11, `uv`
- CalculiX (`ccx`) on PATH (default backend)
- `frd2vtu`, `treeparse`, `b3_mat` (resolved via `pyproject.toml`)

```bash
uv sync            # core + ccx backend
uv sync --extra mfem   # add the MFEM backend
```

## CLI

```bash
uv run b3_core json examples/simple.json          # run a JSON case
uv run b3_core run --dx 50 --dy 50 -t 30 \
    --xgr "5,10,15,3" --core-e 4e9 --resin-e 3.5e9 -o out/   # direct flags
```

Each run writes `run<HASH>.json` with the engineering constants, geometry
metrics, and (optionally) a backend-vs-ccx comparison.

## Python API

```python
from b3_core import homogenize

# json_data is a path to a CpropInput JSON file (or an equivalent dict)
result = homogenize("examples/simple.json")
print(result.material)                  # b3_mat.OrthotropicMaterial
print(result.resin_volume_fraction, result.surface_area_factor)
```

`homogenize` wraps the lower-level `cprop` pipeline; call `cprop(...)` directly
for the full raw output dict.

See `examples/curved_panel/` and `examples/mfem_patterns/` for galleries.

## License

MIT
