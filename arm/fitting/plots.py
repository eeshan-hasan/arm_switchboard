from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curve(
    result: dict[str, Any],
    *,
    ax: plt.Axes | None = None,
    label: str | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    accuracy = np.asarray(result["trajectories"]["accuracy"], dtype=float)
    ax.plot(np.arange(1, len(accuracy) + 1), accuracy, label=label or "accuracy")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Accuracy")
    ax.set_title("Learning Curve")
    if label:
        ax.legend()
    return ax


def plot_variant_scores(
    summaries: list[Any],
    *,
    metric: str = "mean_accuracy",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    labels = [summary.name for summary in summaries]
    scores = [getattr(summary, metric) for summary in summaries]
    ax.bar(labels, scores)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title("Variant Comparison")
    ax.tick_params(axis="x", rotation=20)
    return ax


__all__ = ["plot_learning_curve", "plot_variant_scores"]
