from __future__ import annotations

from typing import Any

import numpy as np

from arm.models import run_learning_trials

from .results import summarize_run
from .variants import build_default_variants


def run_variants(
    x: np.ndarray,
    f: np.ndarray,
    variants: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    configs = variants or build_default_variants()
    results: dict[str, dict[str, Any]] = {}

    for name, config in configs.items():
        params = config.build(n_dims=x.shape[1])
        run = run_learning_trials(x, f, params)
        results[name] = {
            "config": config,
            "run": run,
            "summary": summarize_run(run, name=name),
        }
    return results


__all__ = ["run_variants"]
