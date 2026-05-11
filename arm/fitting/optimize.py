from __future__ import annotations

from itertools import product
from typing import Any, Callable, Iterable


def expand_grid(parameter_grid: dict[str, Iterable[Any]]) -> list[dict[str, Any]]:
    if not parameter_grid:
        return [{}]

    keys = list(parameter_grid)
    values = [list(parameter_grid[key]) for key in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def grid_search(
    evaluate: Callable[[dict[str, Any]], dict[str, Any]],
    parameter_grid: dict[str, Iterable[Any]],
    *,
    score_key: str = "score",
    maximize: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    results = [evaluate(candidate) for candidate in expand_grid(parameter_grid)]
    if not results:
        raise ValueError("grid_search requires at least one candidate")

    best = max(results, key=lambda item: item[score_key]) if maximize else min(
        results,
        key=lambda item: item[score_key],
    )
    return best, results


__all__ = ["expand_grid", "grid_search"]
