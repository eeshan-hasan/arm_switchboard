
from .params import Param
from .transforms import identity, log_10, logit, power_10, sigmoid

Params_Strength = {
    "initial_alpha": Param(
        name="initial_alpha",
        dim=2,
        default=(4, 4),
        bounds=(-3, 3),
        transform=identity,
        inv_transform=identity,
        init_value = 0
    ),
    "initial_alpha_s": Param(
        name="initial_alpha_s",
        dim=1,
        default=(4,),
        bounds=(-3, 3),
        transform=identity,
        inv_transform=identity,
        init_value = 0
    ),
    "delta": Param(
        name="delta",
        dim=2,
        default=(100, 100),
        bounds=(-1, 4),
        transform=power_10,
        inv_transform=log_10,
        init_value = 2
    ),
    "gamma_w": Param(
        name="decay",
        dim=1,
        default=(0.05,),
        bounds=(0.0, 0.95),
        transform=identity,
        inv_transform=identity,
        init_value = 0.05
    ),
    "lr": Param(
        name="lr",
        dim=1,
        default=(0.2,),
        bounds=(-4, 1),
        transform=power_10,
        inv_transform=log_10,
        init_value = 0.1
    ),
    "regularization_strength": Param(
        name="regularization_strength",
        dim=1,
        default=(0,),
        bounds=(-5, 0),
        transform=power_10,
        inv_transform=log_10,
        init_value = 0.1
    ),
    "gamma_w": Param(
        name="gamma_w",
        dim=1,
        default=(1,),
        bounds=(0.0, 0.95),
        transform=identity,
        inv_transform=identity,
        init_value = 0
    ),
    "guessing": Param(
        name="guessing",
        dim=1,
        default=(0.01,),
        bounds=(0.0, 0.95),
        transform=identity,
        inv_transform=identity,
        init_value = 0.01
    ),
    "initialization_association": Param(
        name="initialization_association",
        dim=1,
        default=(1,),
        bounds=(-1, 0),
        transform=identity, #The model transforms it
        inv_transform=identity,
        init_value = 1
    ),
    "beta": Param(
        name="beta",
        dim=1,
        default=(0,),
        bounds=(-1, 1),
        transform=power_10,
        inv_transform=log_10,
        init_value = 0
    ),
    "response_bias": Param(
        name="response_bias",
        dim=1,
        default=(0,),
        bounds=(-1, 1),
        transform=identity,
        inv_transform=identity,
        init_value = 0
    )
}


__all__ = ["Param", "Params"]
