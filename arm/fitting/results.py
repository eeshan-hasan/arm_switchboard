from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class RunSummary:
    name: str
    mean_accuracy: float
    mean_accuracy_max: float
    mean_loss: float
    final_alpha: np.ndarray
    n_trials: int


def summarize_run(result: dict[str, Any], name: str = "run") -> RunSummary:
    trajectories = result["trajectories"]
    return RunSummary(
        name=name,
        mean_accuracy=float(np.mean(trajectories["accuracy"])),
        mean_accuracy_max=float(np.mean(trajectories["accuracy_max"])),
        mean_loss=float(np.mean(trajectories["loss"])),
        final_alpha=np.asarray(result["state"]["initial_alpha"], dtype=float).copy(),
        n_trials=int(len(trajectories["x"])),
    )


__all__ = ["RunSummary", "summarize_run"]
