"""Joint penalized-likelihood fitting with exact PyTorch gradients."""

from dataclasses import dataclass, asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import torch

from .basis import attention_basis
from .model import EstimatedAlphaModel, NUMERIC_PARAMS


# Bounds are in model units, except initialization_association, which is
# already log10 in the original StrengthModel. delta and beta are optimized
# in log10 coordinates for better numerical scaling.
DEFAULT_BOUNDS = {
    "delta": (0.1, 100.0),
    "gamma_w": (0.0, 0.95),
    "decay": (0.0, 0.95),
    "guessing": (0.0, 0.95),
    "initialization_association": (-1.0, 0.0),
    "beta": (0.1, 10.0),
}
LOG_PARAMS = {"delta", "beta"}


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass
class FitResult:
    """Point estimates and diagnostics; alpha_trajectory is BEFORE transformation."""

    params: dict
    coefficients: np.ndarray
    basis: np.ndarray
    alpha_trajectory: np.ndarray
    transformed_alpha: np.ndarray
    effective_sensitivity: np.ndarray
    decision_probs: np.ndarray
    observation_mask: np.ndarray
    neg_LL: float
    smoothness_penalty: float
    initial_prior_penalty: float
    objective: float
    initial_objective: float
    success: bool
    message: str
    n_iter: int
    n_observations: int
    n_parameters: int
    best_run: int
    runs: list
    settings: dict

    def to_frame(self):
        """Trial-aligned fitted attention and probabilities, in input order."""
        data = {"included_in_likelihood": self.observation_mask}
        for d in range(self.alpha_trajectory.shape[1]):
            data[f"alpha_raw_{d}"] = self.alpha_trajectory[:, d]
            data[f"attention_{d}"] = self.transformed_alpha[:, d]
            data[f"effective_sensitivity_{d}"] = self.effective_sensitivity[:, d]
        for c in range(self.decision_probs.shape[1]):
            data[f"model_prob_resp_{c + 1}"] = self.decision_probs[:, c]
        return pd.DataFrame(data)

    def save(self, filename):
        """Save coefficients, trajectories, probabilities and fit settings as JSON."""
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        filename.write_text(json.dumps(_jsonable(asdict(self)), indent=2, allow_nan=False))


class _FitProblem:
    """Pack spline coefficients and behavioral parameters into one optimizer vector."""

    def __init__(self, model, responses, *, trajectory, n_basis, lambda_alpha,
                 alpha_prior_mean, alpha_prior_sd, initial_alpha, alpha_bounds,
                 fit_params, parameter_bounds, mask):
        self.model = model
        self.basis = torch.tensor(attention_basis(model.n_trials, trajectory, n_basis),
                                  dtype=torch.float64)
        self.shape = (self.basis.shape[1], model.n_dimensions)
        self.n_alpha = int(np.prod(self.shape))
        if not np.isfinite(lambda_alpha) or lambda_alpha < 0:
            raise ValueError("lambda_alpha must be finite and nonnegative")
        if not np.isfinite(alpha_prior_sd) or alpha_prior_sd <= 0:
            raise ValueError("alpha_prior_sd must be finite and positive")
        self.lambda_alpha = float(lambda_alpha)
        self.alpha_prior_sd = float(alpha_prior_sd)
        mean = np.broadcast_to(np.asarray(alpha_prior_mean, dtype=float), (model.n_dimensions,))
        initial = np.broadcast_to(np.asarray(initial_alpha, dtype=float), (model.n_dimensions,))
        if not np.isfinite(mean).all() or not np.isfinite(initial).all():
            raise ValueError("initial_alpha and alpha_prior_mean must be finite")
        self.alpha_prior_mean = torch.tensor(mean, dtype=torch.float64)

        responses = np.asarray(responses, dtype=float)
        if responses.shape != (model.n_trials,):
            raise ValueError("responses must have one entry per trial")
        missing = np.isnan(responses) | (responses == -1)
        valid = (~np.isfinite(responses) | (responses != np.floor(responses)) |
                 (responses < 0) | (responses >= model.n_categories))
        if np.any(valid & ~missing):
            raise ValueError("responses must be zero-based labels; use -1 or NaN for missing")
        observed = ~missing
        if mask is not None:
            mask = np.asarray(mask)
            if mask.shape != (model.n_trials,) or mask.dtype != np.bool_:
                raise ValueError("mask must be a boolean array with one entry per trial")
            observed &= mask
        if not observed.any():
            raise ValueError("At least one observed response must be included in the likelihood")
        self.observed = torch.tensor(observed, dtype=torch.bool)
        self.responses = torch.tensor(np.where(missing, 0, responses).astype(np.int64))

        fit_params = tuple(fit_params)
        if len(set(fit_params)) != len(fit_params) or set(fit_params) - set(NUMERIC_PARAMS):
            raise ValueError(f"fit_params must contain unique names from {NUMERIC_PARAMS}")
        if "beta" in fit_params and model.model_params["decision_rule"] != "softmax":
            raise ValueError("beta only affects the softmax decision rule")
        if "initialization_association" in fit_params and model.initial_w is not None:
            raise ValueError("initialization_association is unused when w is supplied")
        overrides = dict(parameter_bounds or {})
        if overrides.keys() - set(fit_params):
            raise ValueError("parameter_bounds may only name parameters in fit_params")
        bounds = {**DEFAULT_BOUNDS, **overrides}
        self.bounds = []
        if alpha_bounds is None:
            self.bounds.extend([(None, None)] * self.n_alpha)
        else:
            lo, hi = alpha_bounds
            if not np.isfinite([lo, hi]).all() or lo >= hi:
                raise ValueError("alpha_bounds must be a finite increasing pair or None")
            if np.any(initial < lo) or np.any(initial > hi):
                raise ValueError("initial_alpha is outside alpha_bounds")
            self.bounds.extend([(float(lo), float(hi))] * self.n_alpha)
        self.x0 = list(np.tile(initial, (self.shape[0], 1)).ravel())
        self.slices = {}
        for name in fit_params:
            lo, hi = bounds[name]
            if not np.isfinite([lo, hi]).all() or lo >= hi:
                raise ValueError(f"Invalid bounds for {name}")
            if name in LOG_PARAMS and lo <= 0:
                raise ValueError(f"{name} bounds must be positive")
            if name in {"gamma_w", "decay", "guessing"} and not (0 <= lo < hi <= 1):
                raise ValueError(f"{name} bounds must be within [0, 1]")
            value = np.atleast_1d(model.model_params[name])
            if np.any(value < lo) or np.any(value > hi):
                raise ValueError(f"Starting {name} is outside its parameter_bounds")
            start = len(self.x0)
            self.slices[name] = slice(start, start + value.size)
            if name in LOG_PARAMS:
                value, lo, hi = np.log10(value), np.log10(lo), np.log10(hi)
            self.x0.extend(value.tolist())
            self.bounds.extend([(float(lo), float(hi))] * value.size)
        self.x0 = np.array(self.x0, dtype=float)

    def unpack(self, vector):
        coefficients = vector[:self.n_alpha].reshape(self.shape)
        alpha = self.basis @ coefficients
        params = {}
        for name, part in self.slices.items():
            value = vector[part]
            if name in LOG_PARAMS:
                value = torch.pow(10.0, value)
            params[name] = value if name == "delta" else value.squeeze(0)
        return coefficients, alpha, params

    def components(self, vector):
        coefficients, alpha, params = self.unpack(vector)
        probabilities = self.model(alpha, params)
        selected = probabilities[self.observed].gather(
            1, self.responses[self.observed, None]
        ).squeeze(1)
        nll = -torch.log(selected.clamp_min(1e-100)).sum()
        smoothness = self.lambda_alpha * (alpha[1:] - alpha[:-1]).square().sum()
        initial_prior = 0.5 * ((alpha[0] - self.alpha_prior_mean) / self.alpha_prior_sd).square().sum()
        return nll, smoothness, initial_prior, coefficients, alpha, params, probabilities

    def value_and_grad(self, vector):
        vector = torch.tensor(vector, dtype=torch.float64, requires_grad=True)
        nll, smoothness, initial_prior, *_ = self.components(vector)
        loss = nll + smoothness + initial_prior
        gradient, = torch.autograd.grad(loss, vector)
        if not torch.isfinite(loss) or not torch.isfinite(gradient).all():
            raise FloatingPointError("Nonfinite objective/gradient; check model scaling and bounds")
        return float(loss.detach()), gradient.detach().numpy().copy()


def fit_subject(X, feedback, responses, *, model_params=None,
                fit_params=("gamma_w",), trajectory="spline", n_basis=15,
                lambda_alpha=1.0, alpha_prior_mean=0.0, alpha_prior_sd=2.0,
                initial_alpha=0.0, alpha_bounds=(-6.0, 6.0), parameter_bounds=None,
                n_categories=None, mask=None, n_runs=1, seed=0,
                maxiter=300, ftol=1e-9, gtol=1e-5):
    """Jointly estimate latent attention and selected behavioral parameters.

    Minimize NLL + lambda_alpha * sum(diff(alpha)**2)
                 + .5 * sum(((alpha[0] - prior_mean) / prior_sd)**2).

    This is a penalized likelihood / joint MAP point estimate, not a marginal
    likelihood or posterior sampler. lambda_alpha is fixed, not estimated.
    All trials update weights, even when their responses are missing/masked.
    Input order defines time. n_runs denotes sequential optimization restarts;
    a run uses L-BFGS-B with full-sequence PyTorch gradients, not numerical DE.
    """
    for name, value in (("n_runs", n_runs), ("maxiter", maxiter)):
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not np.isfinite([ftol, gtol]).all() or ftol <= 0 or gtol <= 0:
        raise ValueError("ftol and gtol must be finite and positive")
    fit_params = tuple(fit_params)
    model = EstimatedAlphaModel(X, feedback, model_params, n_categories)
    problem = _FitProblem(
        model, responses, trajectory=trajectory, n_basis=n_basis, lambda_alpha=lambda_alpha,
        alpha_prior_mean=alpha_prior_mean, alpha_prior_sd=alpha_prior_sd,
        initial_alpha=initial_alpha, alpha_bounds=alpha_bounds, fit_params=fit_params,
        parameter_bounds=parameter_bounds, mask=mask,
    )
    rng = np.random.default_rng(seed)
    runs, candidates = [], []
    lower = np.array([-np.inf if lo is None else lo for lo, _ in problem.bounds])
    upper = np.array([np.inf if hi is None else hi for _, hi in problem.bounds])
    for run in range(n_runs):
        x0 = problem.x0.copy()
        if run:
            x0[:problem.n_alpha] += rng.normal(0, 0.5, problem.n_alpha)
            x0[problem.n_alpha:] = rng.uniform(lower[problem.n_alpha:], upper[problem.n_alpha:])
            x0 = np.clip(x0, lower, upper)
        initial_value, _ = problem.value_and_grad(x0)
        result = minimize(problem.value_and_grad, x0, jac=True, method="L-BFGS-B",
                          bounds=problem.bounds,
                          options={"maxiter": maxiter, "ftol": ftol, "gtol": gtol, "maxls": 30})
        if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
            raise FloatingPointError(f"Optimization run {run} returned nonfinite values")
        retained_initial = bool(result.fun > initial_value)
        if retained_initial:
            result.x = x0
            result.fun, result.jac = problem.value_and_grad(x0)
            result.success = False
            result.message = f"Retained initial point because optimization worsened it: {result.message}"
        candidates.append(result)
        runs.append({"run": run, "initial_objective": initial_value,
                     "objective": float(result.fun), "success": bool(result.success),
                     "message": str(result.message), "n_iter": int(result.nit),
                     "n_evaluations": int(result.nfev),
                     "gradient_inf_norm": float(np.max(np.abs(result.jac)))})
    best_run = int(np.argmin([res.fun for res in candidates]))
    best = candidates[best_run]
    with torch.no_grad():
        nll, smoothness, prior, coefficients, alpha, params, probs = problem.components(
            torch.tensor(best.x, dtype=torch.float64)
        )
        transformed = model.transform_attention(alpha).numpy()
    fitted_params = dict(model.model_params)
    for name, value in params.items():
        fitted_params[name] = value.numpy().copy() if name == "delta" else float(value)
    settings = {
        "trajectory": trajectory, "n_basis_requested": n_basis,
        "n_basis_used": problem.shape[0], "fit_params": fit_params,
        "lambda_alpha": lambda_alpha, "alpha_prior_mean": alpha_prior_mean,
        "alpha_prior_sd": alpha_prior_sd, "alpha_bounds": alpha_bounds,
        "initial_alpha": initial_alpha,
        "parameter_bounds": {name: (parameter_bounds or {}).get(name, DEFAULT_BOUNDS[name])
                             for name in fit_params},
        "n_categories": model.n_categories, "n_runs": n_runs, "seed": seed,
        "maxiter": maxiter, "ftol": ftol, "gtol": gtol,
    }
    return FitResult(
        params=fitted_params, coefficients=coefficients.numpy().copy(), basis=problem.basis.numpy().copy(),
        alpha_trajectory=alpha.numpy().copy(), transformed_alpha=transformed.copy(),
        effective_sensitivity=transformed * fitted_params["delta"],
        decision_probs=probs.numpy().copy(), observation_mask=problem.observed.numpy().copy(),
        neg_LL=float(nll), smoothness_penalty=float(smoothness), initial_prior_penalty=float(prior),
        objective=float(nll + smoothness + prior), initial_objective=runs[best_run]["initial_objective"],
        success=bool(best.success), message=str(best.message), n_iter=int(best.nit),
        n_observations=int(problem.observed.sum()), n_parameters=len(best.x),
        best_run=best_run, runs=runs, settings=settings,
    )
