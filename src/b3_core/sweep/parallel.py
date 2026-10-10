"""Process-pool map for sweep and fit. Children get one BLAS thread."""

from __future__ import annotations

import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable


def _limit_threads() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def map_workers(fn: Callable[[Any], Any], items: list[Any], workers: int) -> list[Any]:
    """Run ``fn`` on each item. ``workers`` of 1 stays in this process.

    Above 1, a fork pool is used and each child sets ``OMP_NUM_THREADS=1``.
    ``fn`` must be picklable (a module-level function).
    """
    count = int(workers)
    if count < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    if count == 1 or len(items) <= 1:
        return [fn(item) for item in items]
    try:
        pickle.dumps(fn)
    except (TypeError, AttributeError, pickle.PickleError) as exc:
        raise TypeError(
            "workers > 1 needs a module-level solve; a lambda cannot cross a process"
        ) from exc
    with ProcessPoolExecutor(max_workers=count, initializer=_limit_threads) as pool:
        return list(pool.map(fn, items, chunksize=1))
