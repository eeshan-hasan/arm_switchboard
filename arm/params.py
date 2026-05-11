from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping

import numpy as np


def _as_array(value: Any, *, n_dims: int, name: str, default: np.ndarray) -> np.ndarray:
    if value is None:
        return default.copy()

    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 1:
        array = np.repeat(array.item(), n_dims)
    if array.size != n_dims:
        raise ValueError(f"{name} must have length {n_dims}, got shape {array.shape}")
    return array.astype(float, copy=True)


@dataclass(slots=True)
class ARMParameters:
    initial_alpha: np.ndarray
    delta: np.ndarray
    lr: float = 0.1
    gamma_w: float = 1.0
    w_update_type: str = "perfect_instances"
    attention_update_type: str = "none"
    regularization_strength: float = 0.01
    regularization_p: float = 2.0
    reg_growth: float = 0.0
    exemplars: np.ndarray | None = None
    w: np.ndarray | None = None
    decay: float = 0.05
    guessing: float = 0.05
    partial_encoding: bool = True
    attention_parameterization: str = "sigmoid"
    loss_derivative: str = "sse"
    attention_update_dims: str = "all"
    initialization: str = "grid"
    initialization_association: float = 0.0
    n_points: int = 10
    alpha_clip: tuple[float, float] = (-3.0, 4.0)

    def to_dict(self) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, np.ndarray):
                output[field.name] = value.copy()
            else:
                output[field.name] = value
        return output


def default_params(n_dims: int = 2, **overrides: Any) -> ARMParameters:
    params = ARMParameters(
        initial_alpha=np.zeros(n_dims, dtype=float),
        delta=np.ones(n_dims, dtype=float),
    )
    merged = params.to_dict()
    merged.update(overrides)
    return ARMParameters(
        initial_alpha=_as_array(
            merged.get("initial_alpha"),
            n_dims=n_dims,
            name="initial_alpha",
            default=params.initial_alpha,
        ),
        delta=_as_array(
            merged.get("delta"),
            n_dims=n_dims,
            name="delta",
            default=params.delta,
        ),
        lr=float(merged.get("lr", params.lr)),
        gamma_w=float(merged.get("gamma_w", merged.get("params_w", params.gamma_w))),
        w_update_type=str(merged.get("w_update_type", params.w_update_type)),
        attention_update_type=str(
            merged.get("attention_update_type", params.attention_update_type)
        ),
        regularization_strength=float(
            merged.get("regularization_strength", params.regularization_strength)
        ),
        regularization_p=float(merged.get("regularization_p", params.regularization_p)),
        reg_growth=float(merged.get("reg_growth", params.reg_growth)),
        exemplars=merged.get("exemplars"),
        w=merged.get("w"),
        decay=float(merged.get("decay", params.decay)),
        guessing=float(merged.get("guessing", params.guessing)),
        partial_encoding=bool(merged.get("partial_encoding", params.partial_encoding)),
        attention_parameterization=str(
            merged.get(
                "attention_parameterization",
                params.attention_parameterization,
            )
        ),
        loss_derivative=str(merged.get("loss_derivative", params.loss_derivative)),
        attention_update_dims=str(
            merged.get("attention_update_dims", params.attention_update_dims)
        ),
        initialization=str(merged.get("initialization", params.initialization)),
        initialization_association=float(
            merged.get(
                "initialization_association",
                params.initialization_association,
            )
        ),
        n_points=int(merged.get("n_points", params.n_points)),
        alpha_clip=tuple(merged.get("alpha_clip", params.alpha_clip)),
    )


def coerce_params(
    params: ARMParameters | Mapping[str, Any] | None = None,
    *,
    n_dims: int = 2,
) -> dict[str, Any]:
    if params is None:
        return default_params(n_dims=n_dims).to_dict()
    if isinstance(params, ARMParameters):
        return default_params(n_dims=n_dims, **params.to_dict()).to_dict()
    if isinstance(params, Mapping):
        normalized = dict(params)
        if "alpha" in normalized and "initial_alpha" not in normalized:
            normalized["initial_alpha"] = normalized.pop("alpha")
        return default_params(n_dims=n_dims, **normalized).to_dict()
    raise TypeError("params must be a mapping, ARMParameters, or None")


__all__ = ["ARMParameters", "coerce_params", "default_params"]
