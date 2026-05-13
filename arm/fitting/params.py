
from dataclasses import dataclass
from typing import Callable

from .transforms import identity, log_10, logit, power_10, sigmoid


@dataclass(slots=True)
class Param:
    name: str
    dim: int
    default: tuple[float, ...]
    range: tuple[float, float]
    transform: Callable[[float], float]
    inv_transform: Callable[[float], float]


Params = {
    "initial_alpha": Param(
        name="initial_alpha",
        dim=2,
        default=(4, 4),
        range=(-3, 3),
        transform=sigmoid,
        inv_transform=logit,
    ),
    "initial_alpha_s": Param(
        name="initial_alpha_s",
        dim=1,
        default=(4),
        range=(-3, 3),
        transform=sigmoid,
        inv_transform=logit,
    ),
    "delta": Param(
        name="delta",
        dim=2,
        default=(100, 100),
        range=(-1, 4),
        transform=power_10,
        inv_transform=log_10,
    ),
    "decay": Param(
        name="decay",
        dim=1,
        default=(0.05,),
        range=(0.0, 0.95),
        transform=identity,
        inv_transform=identity,
    ),
    "lr": Param(
        name="lr",
        dim=1,
        default=(0.2,),
        range=(-4, 1),
        transform=power_10,
        inv_transform=log_10,
    ),
    "regularization_strength": Param(
        name="regularization_strength",
        dim=1,
        default=(0,),
        range=(5, 0),
        transform=power_10,
        inv_transform=log_10,
    ),
    "gamma_w": Param(
        name="gamma_w",
        dim=1,
        default=(1,),
        range=(0.0, 0.95),
        transform=identity,
        inv_transform=identity,
    ),
    "guessing": Param(
        name="guessing",
        dim=1,
        default=(0.01,),
        range=(0.0, 0.95),
        transform=identity,
        inv_transform=identity,
    ),
    "initialization_association": Param(
        name="initialization_association",
        dim=1,
        default=(1,),
        range=(-5, 3),
        transform=power_10,
        inv_transform=log_10,
    ),
}


__all__ = ["Param", "Params"]
