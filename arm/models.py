from __future__ import annotations

from typing import Any

import numpy as np

from .params import coerce_params


verbose = False


def sigmoid(x: np.ndarray | float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def distances(x: np.ndarray, delta: np.ndarray, exemplars: np.ndarray) -> np.ndarray:
    return delta * np.abs(x - exemplars)


def calc_activations(
    x: np.ndarray,
    alpha: np.ndarray,
    delta: np.ndarray,
    exemplars: np.ndarray,
    alpha_trajectory: np.ndarray | list[Any] | None = None,
    partial_encoding: bool = False,
    attention_parameterization: str = "none",
) -> np.ndarray:
    d = distances(x, delta, exemplars)
    alpha_trajectory = [] if alpha_trajectory is None else alpha_trajectory

    if attention_parameterization == "none":
        transformed_alpha = 10 ** alpha
        transformed_alpha_history = 10 ** np.asarray(alpha_trajectory)
    elif attention_parameterization == "sigmoid":
        transformed_alpha = sigmoid(alpha)
        transformed_alpha_history = sigmoid(np.asarray(alpha_trajectory))
    else:
        raise ValueError(f"Unknown attention_parameterization: {attention_parameterization}")

    if partial_encoding and len(alpha_trajectory):
        activations = np.exp((-transformed_alpha) * d * transformed_alpha_history)
    else:
        activations = np.exp((-transformed_alpha) * d)
    return activations


def calc_activations_exemplars(activations: np.ndarray) -> np.ndarray:
    activations = np.prod(activations, axis=1)
    return activations.reshape(len(activations), -1)


def calc_evidence(activations: np.ndarray, w: np.ndarray) -> np.ndarray:
    if verbose:
        print("Activations:", activations)
        print("Weights:", w)
    return (activations * w).sum(axis=0) + 1e-100


def calc_evidence_decision(E: np.ndarray, guessing: float) -> np.ndarray:
    return (E / np.sum(E)) * (1 - guessing) + (guessing / len(E))


def update_exemplars(x: np.ndarray, exemplars: np.ndarray) -> np.ndarray:
    if len(exemplars) == 0:
        return np.array([x])
    return np.vstack([exemplars, x])


def update_weights_RW(
    w_i: np.ndarray,
    f_i: np.ndarray,
    act: np.ndarray,
    gamma_w: float = 1,
) -> np.ndarray:
    error = f_i - w_i
    return w_i + gamma_w * error * act


def update_weights_noisy_feedback(
    w_i: np.ndarray,
    f_i: np.ndarray,
    gamma_w: float = 1,
) -> np.ndarray:
    chance = 1 / len(f_i)
    f_assoc_feedback = gamma_w * (1 - chance) + chance
    not_f_assoc_feedback = (1 - f_assoc_feedback) / (len(f_i) - 1)
    return f_i * f_assoc_feedback + (1 - f_i) * not_f_assoc_feedback


def update_weights(
    w: np.ndarray,
    feedback_mat_active: np.ndarray,
    act: np.ndarray,
    gamma_w: float = 1,
    decay: float = 0,
    w_update_type: str = "perfect_instances",
) -> np.ndarray:
    w = w * (1 - decay)
    if w_update_type == "RW":
        for i in range(w.shape[0]):
            w[i] = update_weights_RW(w[i], feedback_mat_active[-1], act[i], gamma_w)
    elif w_update_type == "noisy_feedback":
        w[-1] = update_weights_noisy_feedback(w[-1], feedback_mat_active[-1], gamma_w)
    elif w_update_type == "perfect_instances":
        w[-1] = feedback_mat_active[-1]
    else:
        raise ValueError(f"Unknown w_update_type: {w_update_type}")

    if verbose:
        print(feedback_mat_active[-1], "Feedback for current exemplar")
    return w


def calc_attention_loss_gradient(
    w: np.ndarray,
    delta: np.ndarray,
    exemplars: np.ndarray,
    activations: np.ndarray,
    probe: np.ndarray,
    y_true: int,
    guessing: float,
    loss_derivative: str = "ce",
    partial_encoding: bool = True,
    alpha_trajectory: np.ndarray | list[Any] | None = None,
    attention_parameterization: str = "sigmoid",
) -> np.ndarray:
    activations = np.asarray(activations).reshape(-1, 1)
    exemplars = np.asarray(exemplars)
    delta = np.asarray(delta).reshape(1, -1)
    alpha_trajectory = [] if alpha_trajectory is None else alpha_trajectory

    cat_ev = calc_evidence(activations, w)
    tot_ev = np.sum(cat_ev)
    den = tot_ev**2
    decision_prob = calc_evidence_decision(cat_ev, guessing)

    if partial_encoding and len(alpha_trajectory):
        if attention_parameterization == "none":
            transformed_alpha_history = 10 ** np.asarray(alpha_trajectory)
        elif attention_parameterization == "sigmoid":
            transformed_alpha_history = sigmoid(np.asarray(alpha_trajectory))
        else:
            raise ValueError(
                f"Unknown attention_parameterization: {attention_parameterization}"
            )
        dervact = -activations * (
            delta * np.abs(exemplars - probe) * transformed_alpha_history
        )
    else:
        dervact = -activations * (delta * np.abs(exemplars - probe))

    evi_deriv = dervact.T @ w
    sum_evi_deriv = evi_deriv.sum(axis=1, keepdims=True)
    dP = (tot_ev * evi_deriv - sum_evi_deriv * cat_ev[None, :]) / den

    p_true = decision_prob[y_true]
    if loss_derivative == "ce":
        grad = -dP[:, y_true] / p_true
    elif loss_derivative == "sse":
        grad = -2 * (1 - p_true) * dP[:, y_true]
    else:
        raise ValueError(f"Unknown loss_derivative: {loss_derivative}")

    if verbose:
        print("grad:", grad)
    return grad


def calc_regularization(
    alpha: np.ndarray,
    p: float = 2,
    regularization_strength: float = 0.1,
) -> np.ndarray:
    return regularization_strength * p * (alpha ** (p - 1))


def update_attention(
    alpha: np.ndarray,
    lr: float,
    activations: np.ndarray,
    w: np.ndarray,
    delta: np.ndarray,
    exemplars: np.ndarray,
    x: np.ndarray,
    f: np.ndarray,
    D: np.ndarray,
    attention_update_type: str = "loss",
    regularization_p: float = 2,
    regularization_strength: float = 0.01,
    reg_growth: float = 1,
    competition_strength: float = 0.01,
    guessing: float = 0.05,
    attention_parameterization: str = "none",
    loss_derivative: str = "ce",
    partial_encoding: bool = True,
    alpha_trajectory: np.ndarray | list[Any] | None = None,
    attention_update_dims: str = "all",
    alpha_clip: tuple[float, float] = (-4, 3),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    del D, reg_growth, competition_strength
    alpha_trajectory = [] if alpha_trajectory is None else alpha_trajectory

    if attention_update_type == "none":
        dloss = np.zeros(shape=alpha.shape)
        dreg = np.zeros(shape=alpha.shape)
    elif attention_update_type == "loss":
        if attention_parameterization == "none":
            dloss = calc_attention_loss_gradient(
                w,
                delta,
                exemplars,
                activations,
                x,
                np.argmax(f),
                guessing,
                loss_derivative=loss_derivative,
                partial_encoding=partial_encoding,
                alpha_trajectory=alpha_trajectory,
                attention_parameterization=attention_parameterization,
            ) * (10 ** alpha)
        elif attention_parameterization == "sigmoid":
            sigmoid_alpha = sigmoid(alpha)
            dloss = calc_attention_loss_gradient(
                w,
                delta,
                exemplars,
                activations,
                x,
                np.argmax(f),
                guessing,
                loss_derivative=loss_derivative,
                partial_encoding=partial_encoding,
                alpha_trajectory=alpha_trajectory,
                attention_parameterization=attention_parameterization,
            ) * sigmoid_alpha * (1 - sigmoid_alpha)
        else:
            raise ValueError(
                f"Unknown attention_parameterization: {attention_parameterization}"
            )
        dreg = np.zeros(shape=alpha.shape)
    elif attention_update_type == "p_regularization":
        if attention_parameterization == "sigmoid":
            sigmoid_alpha = sigmoid(alpha)
            dloss = calc_attention_loss_gradient(
                w,
                delta,
                exemplars,
                activations,
                x,
                np.argmax(f),
                guessing,
                loss_derivative=loss_derivative,
                partial_encoding=partial_encoding,
                alpha_trajectory=alpha_trajectory,
                attention_parameterization=attention_parameterization,
            ) * sigmoid_alpha * (1 - sigmoid_alpha)
            dreg = calc_regularization(
                sigmoid(alpha),
                p=regularization_p,
                regularization_strength=regularization_strength,
            ) * sigmoid(alpha) * (1 - sigmoid(alpha))
        elif attention_parameterization == "none":
            dloss = calc_attention_loss_gradient(
                w,
                delta,
                exemplars,
                activations,
                x,
                np.argmax(f),
                guessing,
                loss_derivative=loss_derivative,
                partial_encoding=partial_encoding,
                alpha_trajectory=alpha_trajectory,
                attention_parameterization=attention_parameterization,
            ) * (10 ** alpha)
            dreg = calc_regularization(
                alpha + 1,
                p=regularization_p,
                regularization_strength=regularization_strength,
            )
        else:
            raise ValueError(
                f"Unknown attention_parameterization: {attention_parameterization}"
            )
    else:
        raise ValueError(f"Unknown attention_update_type: {attention_update_type}")

    if attention_update_dims == "all":
        alpha = alpha - lr * (dloss + dreg)
    elif attention_update_dims == "single":
        alpha = alpha - lr * (dloss + dreg).sum()
    else:
        raise ValueError(f"Unknown attention_update_dims: {attention_update_dims}")

    alpha = alpha.clip(alpha_clip[0], alpha_clip[1])
    if verbose:
        print("Updated alpha:", alpha)
    return alpha, dloss, dreg


def set_default(
    x: np.ndarray,
    f: np.ndarray,
    params: dict[str, Any] | None,
) -> tuple[Any, ...]:
    params = coerce_params(params, n_dims=x.shape[1])

    initialization = params.get("initialization", "grid")
    initialization_association = 10 ** params.get("initialization_association", 0)
    n_points = params.get("n_points", 10)
    alpha = params["initial_alpha"]
    delta = params["delta"]
    lr = params.get("lr", 0.1)
    attention_update_type = params.get("attention_update_type", "none")
    alpha_clip = params.get("alpha_clip", (-3, 4))
    gamma_w = params.get("gamma_w", 1)
    w_update_type = params.get("w_update_type", "perfect_instances")
    regularization_p = params.get("regularization_p", 1)
    regularization_strength = params.get("regularization_strength", 0.01)
    reg_growth = params.get("reg_growth", 0)

    exemplars_override = params.get("exemplars")
    if initialization == "point":
        exemplars = (
            exemplars_override
            if exemplars_override is not None
            else np.array([np.mean(x, axis=0)])
        )
    elif initialization == "grid":
        mins = np.min(x, axis=0)
        maxs = np.max(x, axis=0)
        grid_axes = [np.linspace(lo, hi, n_points) for lo, hi in zip(mins, maxs)]
        background_grid = np.stack(np.meshgrid(*grid_axes), axis=-1).reshape(-1, x.shape[1])
        exemplars = exemplars_override if exemplars_override is not None else background_grid
    else:
        raise ValueError(f"Unknown initialization: {initialization}")

    decay = params.get("decay", 0.05)
    guessing = params.get("guessing", 0.05)
    partial_encoding = params.get("partial_encoding", True)
    attention_parameterization = params.get("attention_parameterization", "sigmoid")
    loss_derivative = params.get("loss_derivative", "sse")
    attention_update_dims = params.get("attention_update_dims", "all")
    if attention_update_dims == "single":
        alpha = np.zeros(shape=delta.shape) + np.mean(alpha)

    if params.get("w") is None:
        if initialization == "point":
            w = np.zeros((len(x) + 1, len(np.unique(f))))
            w[0, :] = initialization_association / max(1, len(np.unique(f)))
            counter = 1
        else:
            grid_size = exemplars.shape[0]
            w = np.zeros((len(x) + grid_size, len(np.unique(f))))
            w[:grid_size, :] = initialization_association / max(1, grid_size)
            counter = grid_size
    else:
        w = params["w"]
        counter = len(w) + 1
        w = np.vstack([w, np.zeros((len(x), len(np.unique(f))))])

    return (
        alpha,
        delta,
        lr,
        gamma_w,
        w_update_type,
        attention_update_type,
        regularization_strength,
        regularization_p,
        reg_growth,
        exemplars,
        w,
        counter,
        decay,
        guessing,
        partial_encoding,
        attention_parameterization,
        loss_derivative,
        attention_update_dims,
        initialization,
        initialization_association,
        n_points,
        alpha_clip,
    )


def run_learning_trials(
    x: np.ndarray,
    f: np.ndarray,
    params: dict[str, Any] | None,
    save_trajectories: bool = True,
) -> dict[str, Any]:
    (
        alpha,
        delta,
        lr,
        gamma_w,
        w_update_type,
        attention_update_type,
        regularization_strength,
        regularization_p,
        reg_growth,
        exemplars,
        w,
        counter,
        decay,
        guessing,
        partial_encoding,
        attention_parameterization,
        loss_derivative,
        attention_update_dims,
        initialization,
        initialization_association,
        n_points,
        alpha_clip,
    ) = set_default(x, f, params)

    del counter, initialization_association

    all_w = []
    all_alpha = np.zeros((w.shape[0], alpha.shape[0]))
    if initialization == "point":
        all_alpha[0, :] = alpha
    else:
        all_alpha[: exemplars.shape[0], :] = alpha

    decisions = []
    dloss = []
    dreg = []
    feedback_mat = np.zeros((len(x), len(np.unique(f))))
    feedback_mat[np.arange(len(x)), f] = 1

    for idx in range(len(x)):
        current_idx = len(exemplars)
        probe = x[idx]
        w_active = w[:current_idx, :]
        all_alpha_active = all_alpha[:current_idx, :]
        activations_mat = calc_activations(
            probe,
            alpha,
            delta,
            exemplars,
            alpha_trajectory=all_alpha_active,
            partial_encoding=partial_encoding,
            attention_parameterization=attention_parameterization,
        )
        activations = calc_activations_exemplars(activations_mat)

        E = calc_evidence(activations, w_active)
        D = calc_evidence_decision(E, guessing)
        feedback_mat_active = feedback_mat[: idx + 1, :]
        f_trial = feedback_mat_active[-1]

        if current_idx == 1:
            prev_alpha = alpha
            dloss_trial = np.zeros(shape=alpha.shape)
            dreg_trial = np.zeros(shape=alpha.shape)
        else:
            prev_alpha = alpha
            alpha, dloss_trial, dreg_trial = update_attention(
                attention_update_type=attention_update_type,
                alpha=alpha,
                lr=lr,
                activations=activations,
                w=w_active,
                delta=delta,
                exemplars=exemplars,
                x=probe,
                f=f_trial,
                D=D,
                regularization_strength=regularization_strength,
                regularization_p=regularization_p,
                reg_growth=reg_growth,
                guessing=guessing,
                attention_parameterization=attention_parameterization,
                loss_derivative=loss_derivative,
                partial_encoding=partial_encoding,
                alpha_trajectory=all_alpha_active,
                attention_update_dims=attention_update_dims,
                alpha_clip=alpha_clip,
            )

        w[: current_idx + 1, :] = update_weights(
            w=w[: current_idx + 1, :],
            feedback_mat_active=feedback_mat_active,
            act=activations,
            gamma_w=gamma_w,
            decay=decay,
            w_update_type=w_update_type,
        )

        exemplars = update_exemplars(probe, exemplars)
        decisions.append(D)
        if save_trajectories:
            all_w.append(w.copy())
            all_alpha[current_idx, :] = prev_alpha
            dloss.append(dloss_trial)
            dreg.append(dreg_trial)

    decision_prob = np.array(decisions)
    if np.isnan(decision_prob.sum()):
        correct_prob = np.zeros(len(f))
        decisions_realized = np.random.binomial(1, 0.5, len(f))
        accuracy = decisions_realized == f
    else:
        correct_prob = decision_prob[np.arange(f.size), f]
        if decision_prob.shape[1] == 2:
            decisions_realized = np.random.binomial(1, decision_prob[:, 1])
        else:
            decisions_realized = decision_prob.argmax(axis=1)
        accuracy = decisions_realized == f

    accuracy_max = decision_prob.argmax(axis=1) == f
    correct_prob = np.clip(correct_prob, 1e-12, 1.0)
    loss = -np.log(correct_prob)

    trajectories = {
        "w": np.array(all_w),
        "alpha": np.array(all_alpha),
        "decision_prob": np.array(decision_prob),
        "decisions": np.array(decisions_realized),
        "x": x,
        "f": f,
        "accuracy_max": accuracy_max,
        "correct_prob": correct_prob,
        "accuracy": accuracy,
        "loss": loss,
        "dloss": np.array(dloss),
        "dreg": np.array(dreg),
    }
    state = {
        "w": w,
        "exemplars": exemplars,
        "initial_alpha": alpha,
        "delta": delta,
        "gamma_w": gamma_w,
        "decay": decay,
        "regularization_strength": regularization_strength,
        "regularization_p": regularization_p,
        "guessing": guessing,
        "lr": lr,
        "w_update_type": w_update_type,
        "attention_update_type": attention_update_type,
        "partial_encoding": partial_encoding,
        "loss_derivative": loss_derivative,
        "attention_update_dims": attention_update_dims,
    }
    return {"trajectories": trajectories, "state": state}


def make_predictions(x: np.ndarray, params: dict[str, Any]) -> dict[str, Any]:
    params = coerce_params(params, n_dims=x.shape[1])
    alpha = params["initial_alpha"]
    delta = params["delta"]
    exemplars = params["exemplars"]
    w = params["w"]
    guessing = params.get("guessing", 0.05)
    if exemplars is None or w is None:
        raise ValueError("params must include fitted `exemplars` and `w` for prediction")

    decisions = []
    for point in x:
        activations = calc_activations(point, alpha, delta, exemplars)
        activations = calc_activations_exemplars(activations)
        E = calc_evidence(activations, w)
        D = calc_evidence_decision(E, guessing)
        decisions.append(D)

    decisions = np.array(decisions)
    return {
        "trajectories": {"decision_prob": decisions, "x": x},
        "state": {
            "w": w,
            "exemplars": exemplars,
            "initial_alpha": alpha,
            "delta": delta,
        },
    }


__all__ = [
    "calc_activations",
    "calc_evidence",
    "calc_evidence_decision",
    "distances",
    "make_predictions",
    "run_learning_trials",
    "sigmoid",
]
