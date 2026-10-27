# Offline scripts — optional, not part of `make` or the main CLI

These are standalone runners for heavier or extra-dependency workflows. The
mainline path is `b3_core` + `make` (see repo `Makefile` and `b3_core --help`).

| Script | Needs | Output |
|--------|-------|--------|
| `sweep_gifs.py` | `[anim]` | `examples/param_sweeps/img/*.gif` |
| `sweep_full.py` | MFEM + `[anim]` | homogenise + response plots + gallery PNGs + GIFs |
| `explainer.py` | `[anim]`, MFEM | `examples/offline/out/explainer.{mp4,gif}` |
| `viz_scratch.py` | MFEM | `examples/offline/out/viz/` |
| `interactive_view.py` | `[interactive]` | HTML viewer |

```bash
uv sync --extra anim          # GIF / MP4 scripts
uv sync --extra interactive   # interactive_view.py

uv run python examples/offline/sweep_gifs.py
uv run python examples/offline/sweep_full.py
uv run python examples/offline/explainer.py examples/mfem_patterns/two_sided.json
```