"""Differentiable StrengthModel with externally supplied latent attention.

The activation, evidence, decision, decay and association-update equations
match arm.models.strength_learning. There is no attention learning rule here.
"""

import numpy as np
import torch
from torch import nn


DEFAULT_PARAMS = {
    "delta": 50.0,
    "gamma_w": 0.1,
    "decay": 0.0,
    "guessing": 0.01,
    "initialization_association": 0.0,  # log10 total initial strength per category
    "beta": 1.0,
    "n_points": 30,
    "decision_rule": "luce",
    "w_update_type": "prediction_error",
    "attention_parameterization": "sigmoid",
}
NUMERIC_PARAMS = (
    "delta", "gamma_w", "decay", "guessing", "initialization_association", "beta"
)


def integer_labels(values, n_trials, name):
    values = np.asarray(values, dtype=float)
    if values.shape != (n_trials,) or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain one finite integer per trial")
    if np.any(values < 0) or np.any(values != np.floor(values)):
        raise ValueError(f"{name} must contain zero-based nonnegative integers")
    return values.astype(np.int64)


class EstimatedAlphaModel(nn.Module):
    """Evaluate one subject's full trial sequence in float64 on CPU.

    X has shape (trials, dimensions); feedback is zero-based. Each forward
    pass resets association weights. Gradients pass through every weight
    update, so early attention affects both current and later likelihoods.
    """

    def __init__(self, X, feedback, model_params=None, n_categories=None):
        super().__init__()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or min(X.shape) < 1 or not np.isfinite(X).all():
            raise ValueError("X must be a nonempty finite (trials, dimensions) array")
        feedback = integer_labels(feedback, len(X), "feedback")
        if n_categories is None:
            n_categories = max(2, int(feedback.max()) + 1)
        if (not isinstance(n_categories, (int, np.integer)) or
                n_categories < 2 or feedback.max() >= n_categories):
            raise ValueError("n_categories must include all feedback categories and be >= 2")
        self.n_categories = int(n_categories)
        self.n_trials, self.n_dimensions = X.shape
        supplied = dict(model_params or {})
        unknown = supplied.keys() - (DEFAULT_PARAMS.keys() | {"w"})
        if unknown:
            raise ValueError(f"Unknown model parameters: {sorted(unknown)}")
        self.model_params = {**DEFAULT_PARAMS, **supplied}
        for name in NUMERIC_PARAMS:
            value = np.asarray(self.model_params[name], dtype=float)
            if name == "delta":
                value = np.broadcast_to(value, (self.n_dimensions,)).copy()
            elif value.size == 1:
                value = value.reshape(())
            else:
                raise ValueError(f"{name} must be a scalar")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
            self.model_params[name] = value if name == "delta" else float(value)
        for name in ("delta", "beta"):
            if np.any(np.asarray(self.model_params[name]) <= 0):
                raise ValueError(f"{name} must be positive")
        for name in ("gamma_w", "decay", "guessing"):
            if not 0 <= self.model_params[name] <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.model_params["attention_parameterization"] not in {"sigmoid", "none"}:
            raise ValueError("attention_parameterization must be 'sigmoid' or 'none' (10**alpha)")
        if self.model_params["decision_rule"] not in {"luce", "softmax"}:
            raise ValueError("decision_rule must be 'luce' or 'softmax'")
        if self.model_params["w_update_type"] not in {
            "hebbian", "prediction_error", "prediction_error_2"
        }:
            raise ValueError("Unsupported w_update_type")
        n_points = self.model_params["n_points"]
        if not isinstance(n_points, (int, np.integer)) or n_points < 1:
            raise ValueError("n_points must be a positive integer")
        axes = [np.linspace(lo, hi, n_points) for lo, hi in zip(X.min(0), X.max(0))]
        hidden_units = np.stack(np.meshgrid(*axes), axis=-1).reshape(-1, X.shape[1])
        self.register_buffer("hidden_units", torch.tensor(hidden_units, dtype=torch.float64))
        self.register_buffer("distances", torch.tensor(
            np.abs(X[:, None, :] - hidden_units[None, :, :]), dtype=torch.float64
        ))
        self.register_buffer("feedback", torch.tensor(feedback, dtype=torch.long))
        self.register_buffer("feedback_matrix", torch.nn.functional.one_hot(
            self.feedback, num_classes=self.n_categories
        ).to(torch.float64))
        w = self.model_params.get("w")
        if w is not None:
            w = np.asarray(w, dtype=float)
            if (w.shape != (len(hidden_units), self.n_categories) or
                    not np.isfinite(w).all() or np.any(w < 0)):
                raise ValueError("w must be finite, nonnegative and shaped (grid_size, n_categories)")
            self.model_params["w"] = w.copy()
        self.register_buffer("initial_w", None if w is None else torch.tensor(w, dtype=torch.float64))

    def transform_attention(self, alpha):
        if self.model_params["attention_parameterization"] == "sigmoid":
            return torch.sigmoid(alpha)
        return torch.pow(10.0, alpha)

    def forward(self, alpha, params=None):
        """Return (trials, categories) probabilities BEFORE each feedback update.

        ``params`` can override numeric values with differentiable tensors.
        Do not detach those tensors or the recursively updated weights.
        """
        alpha = torch.as_tensor(alpha, dtype=torch.float64, device=self.distances.device)
        if alpha.shape != (self.n_trials, self.n_dimensions):
            raise ValueError("alpha must have shape (trials, dimensions)")
        params = {} if params is None else params
        if params.keys() - set(NUMERIC_PARAMS):
            raise ValueError("forward params may only override numeric model parameters")
        p = {name: torch.as_tensor(params.get(name, self.model_params[name]),
                                  dtype=alpha.dtype, device=alpha.device)
             for name in NUMERIC_PARAMS}
        if self.initial_w is None:
            w = torch.ones((len(self.hidden_units), self.n_categories),
                           dtype=alpha.dtype, device=alpha.device)
            w = w * (torch.pow(10.0, p["initialization_association"]) / len(self.hidden_units))
        else:
            w = self.initial_w
        attention = self.transform_attention(alpha)
        activations = torch.exp(-(
            self.distances * p["delta"] * attention[:, None, :]
        ).sum(dim=2))
        decisions = []
        for t in range(self.n_trials):
            a = activations[t, :, None]
            evidence = (a * w).sum(dim=0) + 1e-100
            if self.model_params["decision_rule"] == "luce":
                decision = evidence / evidence.sum()
            else:
                decision = torch.softmax(p["beta"] * evidence, dim=0)
            decision = (1 - p["guessing"]) * decision + p["guessing"] / self.n_categories
            decisions.append(decision)
            w = w * (1 - p["decay"])
            target = self.feedback_matrix[t]
            if self.model_params["w_update_type"] == "hebbian":
                w = w + p["gamma_w"] * target * a
            elif self.model_params["w_update_type"] == "prediction_error":
                w = torch.clamp(w + p["gamma_w"] * (target - w) * a, min=0)
            else:
                w = torch.clamp(w + p["gamma_w"] * (target - decision) * a, min=0)
        return torch.stack(decisions)
