from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from arm.params import coerce_params


@dataclass(slots=True)
class ModelConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def build(self, *, n_dims: int = 2) -> dict[str, Any]:
        return coerce_params(self.params, n_dims=n_dims)


def make_model_config(name: str, description: str = "", **params: Any) -> ModelConfig:
    return ModelConfig(name=name, params=dict(params), description=description)


__all__ = ["ModelConfig", "make_model_config"]
