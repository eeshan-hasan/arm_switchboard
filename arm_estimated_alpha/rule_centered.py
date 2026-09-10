"""Latent attention with Gaussian deviations from the original learning rule.

alpha[t+1] ~ Normal(alpha[t] - lr * (loss_gradient + regularizer_gradient), sd**2 I)

The local update gradient uses feedback and pre-update association weights,
exactly as the original Luce/sigmoid StrengthModel. The outer fitting gradient
also differentiates through this rule and all recursive association updates.
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import minimize
import torch

from .model import EstimatedAlphaModel, NUMERIC_PARAMS
from .fitting import FitResult, _FitProblem, DEFAULT_BOUNDS
from .config import EstimatedAlphaConfig


class RuleCenteredModel(EstimatedAlphaModel):
    """The original loss/p-regularization drift, evaluated on a latent path.

    Currently supports the Luce decision rule with sigmoid attention, including
    all three association-weight rules. Unsupported variants fail explicitly.
    """

    def __init__(self, X, feedback, model_params=None, n_categories=None,
                 regularization_p=1.0, loss_derivative="ce", update_clip=None):
        super().__init__(X, feedback, model_params, n_categories)
        if self.model_params["decision_rule"] != "luce":
            raise ValueError("The rule-centered update currently requires decision_rule='luce'")
        if self.model_params["attention_parameterization"] != "sigmoid":
            raise ValueError("The rule-centered update currently requires sigmoid attention")
        if not np.isfinite(regularization_p) or regularization_p <= 0:
            raise ValueError("regularization_p must be finite and positive")
        if loss_derivative not in {"ce", "sse"}:
            raise ValueError("loss_derivative must be 'ce' or 'sse'")
        if update_clip is not None:
            lo, hi = update_clip
            if not np.isfinite([lo, hi]).all() or lo >= hi:
                raise ValueError("update_clip must be a finite increasing pair or None")
        self.regularization_p = float(regularization_p)
        self.loss_derivative = loss_derivative
        self.update_clip = update_clip

    def rule_statistics(self, alpha, params=None, *, lr=0.1, regularization_strength=0.01):
        """Return probabilities, predicted next raw alpha, loss and reg gradients.

        No random noise is drawn during likelihood evaluation. Conditional on
        alpha, everything is deterministic. The Gaussian residual is scored by
        the fitting objective, independently of observed-response masking.
        """
        alpha = torch.as_tensor(alpha, dtype=torch.float64, device=self.distances.device)
        if alpha.shape != (self.n_trials, self.n_dimensions):
            raise ValueError("alpha must have shape (trials, dimensions)")
        params = {} if params is None else params
        if params.keys() - set(NUMERIC_PARAMS):
            raise ValueError("params may only override numeric behavioral parameters")
        p = {name: torch.as_tensor(params.get(name, self.model_params[name]),
                                  dtype=alpha.dtype, device=alpha.device) for name in NUMERIC_PARAMS}
        lr = torch.as_tensor(lr, dtype=alpha.dtype, device=alpha.device)
        regularization_strength = torch.as_tensor(regularization_strength,
                                                 dtype=alpha.dtype, device=alpha.device)
        if self.initial_w is None:
            w = torch.ones((len(self.hidden_units), self.n_categories),
                           dtype=alpha.dtype, device=alpha.device)
            w = w * torch.pow(10.0, p["initialization_association"]) / len(self.hidden_units)
        else:
            w = self.initial_w
        attention = torch.sigmoid(alpha)
        distances = self.distances * p["delta"]
        activations = torch.exp(-(distances * attention[:, None, :]).sum(2))
        decisions, means, loss_gradients, reg_gradients = [], [], [], []
        for t in range(self.n_trials):
            a = activations[t, :, None]
            evidence = (a * w).sum(0) + 1e-100
            total = evidence.sum()
            q = evidence / total
            decision = (1 - p["guessing"]) * q + p["guessing"] / self.n_categories
            target = self.feedback_matrix[t]
            # Partial derivative at fixed current w, with respect to transformed
            # attention. Outer autograd still sees w's dependence on past alpha.
            d_evidence = (-a * distances[t]).T @ w
            d_probability = (1 - p["guessing"]) * (
                d_evidence - q[None, :] * d_evidence.sum(1, keepdim=True)
            ) / total
            p_true = (decision * target).sum().clamp_min(1e-100)
            d_p_true = d_probability @ target
            if self.loss_derivative == "ce":
                local_gradient = -d_p_true / p_true
            else:
                local_gradient = -2 * (1 - p_true) * d_p_true
            loss_gradient = local_gradient * attention[t] * (1 - attention[t])
            # d/d(raw alpha) of reg_strength * sum(sigmoid(alpha)**p).
            reg_gradient = (regularization_strength * self.regularization_p
                            * attention[t].pow(self.regularization_p) * (1 - attention[t]))
            mean = alpha[t] - lr * (loss_gradient + reg_gradient)
            if self.update_clip is not None:
                mean = torch.clamp(mean, *self.update_clip)
            decisions.append(decision)
            means.append(mean)
            loss_gradients.append(loss_gradient)
            reg_gradients.append(reg_gradient)
            # Weight decay and learning occur AFTER prediction and attention drift.
            w = w * (1 - p["decay"])
            if self.model_params["w_update_type"] == "hebbian":
                w = w + p["gamma_w"] * target * a
            elif self.model_params["w_update_type"] == "prediction_error":
                w = torch.clamp(w + p["gamma_w"] * (target - w) * a, min=0)
            else:
                w = torch.clamp(w + p["gamma_w"] * (target - decision) * a, min=0)
        return tuple(torch.stack(values) for values in (decisions, means, loss_gradients, reg_gradients))


@dataclass
class RuleCenteredFitResult(FitResult):
    rule_params: dict
    expected_next_alpha: np.ndarray  # row t predicts alpha[t+1], T-1 rows
    transition_residuals: np.ndarray
    loss_gradients: np.ndarray
    regularization_gradients: np.ndarray
    transition_penalty: float
    transition_log_normalizer: float
    transition_nll: float
    transition_sd_at_bound: bool

    def to_frame(self):
        frame = super().to_frame()
        for d in range(self.alpha_trajectory.shape[1]):
            # Align predicted alpha[t] and epsilon[t-1] with destination trial t.
            frame[f"rule_predicted_alpha_raw_{d}"] = np.r_[np.nan, self.expected_next_alpha[:, d]]
            frame[f"transition_error_{d}"] = np.r_[np.nan, self.transition_residuals[:, d]]
        return frame


class _RuleCenteredProblem(_FitProblem):
    def __init__(self, model, responses, *, lr, regularization_strength,
                 fit_rule_params, rule_parameter_bounds, transition_sd,
                 fit_transition_sd, transition_sd_bounds, initial_coefficients=None,
                 **options):
        super().__init__(model, responses, lambda_alpha=0.0, **options)
        if model.n_trials < 2:
            raise ValueError("A rule-centered fit requires at least two trials")
        self.rule_values = {"lr": float(lr), "regularization_strength": float(regularization_strength),
                            "transition_sd": float(transition_sd)}
        for name, value in self.rule_values.items():
            if not np.isfinite(value) or value < 0 or (name == "transition_sd" and value == 0):
                raise ValueError(f"Invalid {name}; transition_sd must be positive, rule rates nonnegative")
        fit_rule_params = tuple(fit_rule_params)
        if (len(set(fit_rule_params)) != len(fit_rule_params) or
                set(fit_rule_params) - {"lr", "regularization_strength"}):
            raise ValueError("fit_rule_params may contain 'lr' and 'regularization_strength' once each")
        overrides = dict(rule_parameter_bounds or {})
        if overrides.keys() - set(fit_rule_params):
            raise ValueError("rule_parameter_bounds must name fitted rule parameters")
        bounds = {"lr": (1e-4, 100.0), "regularization_strength": (1e-3, 1.0), **overrides}
        names = list(fit_rule_params)
        if fit_transition_sd:
            if transition_sd_bounds is None:
                raise ValueError("Estimating transition_sd requires explicit positive transition_sd_bounds")
            names.append("transition_sd")
            bounds["transition_sd"] = transition_sd_bounds
        elif transition_sd_bounds is not None:
            raise ValueError("transition_sd_bounds is only used when fit_transition_sd=True")
        self.rule_slices = {}
        values = self.x0.tolist()
        self.rule_bounds = {}
        for name in names:
            lo, hi = bounds[name]
            value = self.rule_values[name]
            if not np.isfinite([lo, hi]).all() or not 0 < lo < hi:
                raise ValueError(f"Bounds for {name} must be finite, positive and increasing")
            if not lo <= value <= hi:
                raise ValueError(f"Initial {name} is outside its bounds")
            self.rule_slices[name] = len(values)
            self.rule_bounds[name] = (float(lo), float(hi))
            values.append(np.log10(value))
            self.bounds.append((np.log10(lo), np.log10(hi)))
        self.x0 = np.array(values, dtype=float)
        if initial_coefficients is not None:
            coefficients = np.asarray(initial_coefficients, dtype=float)
            if coefficients.shape != self.shape or not np.isfinite(coefficients).all():
                raise ValueError(f"initial_coefficients must be finite with shape {self.shape}")
            for value, (lo, hi) in zip(coefficients.ravel(), self.bounds[:self.n_alpha]):
                if (lo is not None and value < lo) or (hi is not None and value > hi):
                    raise ValueError("initial_coefficients are outside alpha_bounds")
            self.x0[:self.n_alpha] = coefficients.ravel()

    def details(self, vector):
        coefficients, alpha, params = super().unpack(vector)
        rule = {name: torch.as_tensor(value, dtype=vector.dtype) for name, value in self.rule_values.items()}
        for name, index in self.rule_slices.items():
            rule[name] = torch.pow(10.0, vector[index])
        probs, means, dloss, dreg = self.model.rule_statistics(
            alpha, params, lr=rule["lr"], regularization_strength=rule["regularization_strength"])
        selected = probs[self.observed].gather(1, self.responses[self.observed, None]).squeeze(1)
        nll = -torch.log(selected.clamp_min(1e-100)).sum()
        residuals = alpha[1:] - means[:-1]
        penalty = 0.5 * (residuals / rule["transition_sd"]).square().sum()
        # Required when fitting sigma; retained for fixed sigma for an explicit Gaussian NLL.
        normalizer = residuals.numel() * (torch.log(rule["transition_sd"]) + 0.5 * math.log(2 * math.pi))
        prior = 0.5 * ((alpha[0] - self.alpha_prior_mean) / self.alpha_prior_sd).square().sum()
        return dict(nll=nll, transition_penalty=penalty, transition_log_normalizer=normalizer,
                    transition_nll=penalty + normalizer, prior=prior,
                    coefficients=coefficients, alpha=alpha, params=params, rule=rule,
                    probabilities=probs, expected_next_alpha=means[:-1], residuals=residuals,
                    loss_gradients=dloss, regularization_gradients=dreg)

    def components(self, vector):
        d = self.details(vector)
        # Reuse the tested autograd value/gradient wrapper from _FitProblem.
        return d["nll"], d["transition_nll"], d["prior"]


def fit_rule_centered_subject(
    X, feedback, responses, *, model_params=None, fit_params=("gamma_w",),
    trajectory="spline", n_basis=15, lr=0.1, regularization_strength=0.01,
    regularization_p=1.0, loss_derivative="ce", update_clip=None,
    fit_rule_params=("lr", "regularization_strength"), rule_parameter_bounds=None,
    transition_sd=0.1, fit_transition_sd=False, transition_sd_bounds=None,
    alpha_prior_mean=0.0, alpha_prior_sd=2.0, initial_alpha=0.0,
    initial_coefficients=None, alpha_bounds=(-6.0, 6.0), parameter_bounds=None,
    n_categories=None, mask=None, n_runs=1, seed=0, maxiter=600,
    ftol=1e-9, gtol=1e-5, progress_every=0,
):
    """Joint MAP / constrained spline MAP around the original attention update.

    One Gaussian transition SD applies to every trial and dimension. It is fixed
    by default; optional joint estimation requires explicit positive bounds to
    avoid an unbounded mode as the latent path approaches a deterministic rule.
    The normalizer n*log(sd) is included. Scores are conditional in-sample fits.
    Use trajectory='trial' for a full latent path, or 'spline' for the same
    reduced representation used by the earlier varying-alpha model.
    """
    for name, value in (("n_runs", n_runs), ("maxiter", maxiter)):
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not np.isfinite([ftol, gtol]).all() or ftol <= 0 or gtol <= 0:
        raise ValueError("ftol and gtol must be positive and finite")
    if not isinstance(progress_every, (int, np.integer)) or progress_every < 0:
        raise ValueError("progress_every must be a nonnegative integer")
    fit_params, fit_rule_params = tuple(fit_params), tuple(fit_rule_params)
    model = RuleCenteredModel(X, feedback, model_params, n_categories,
                              regularization_p, loss_derivative, update_clip)
    problem = _RuleCenteredProblem(
        model, responses, trajectory=trajectory, n_basis=n_basis,
        alpha_prior_mean=alpha_prior_mean, alpha_prior_sd=alpha_prior_sd,
        initial_alpha=initial_alpha, initial_coefficients=initial_coefficients,
        alpha_bounds=alpha_bounds, fit_params=fit_params, parameter_bounds=parameter_bounds,
        mask=mask, lr=lr, regularization_strength=regularization_strength,
        fit_rule_params=fit_rule_params, rule_parameter_bounds=rule_parameter_bounds,
        transition_sd=transition_sd, fit_transition_sd=fit_transition_sd,
        transition_sd_bounds=transition_sd_bounds)
    rng = np.random.default_rng(seed)
    lower = np.array([-np.inf if lo is None else lo for lo, _ in problem.bounds])
    upper = np.array([np.inf if hi is None else hi for _, hi in problem.bounds])
    candidates, runs = [], []
    for run in range(n_runs):
        x0 = problem.x0.copy()
        if run:
            x0[:problem.n_alpha] += rng.normal(0, 0.5, problem.n_alpha)
            x0[problem.n_alpha:] = rng.uniform(lower[problem.n_alpha:], upper[problem.n_alpha:])
            x0 = np.clip(x0, lower, upper)
        initial_value, _ = problem.value_and_grad(x0)
        iteration = 0

        def progress(intermediate_result):
            nonlocal iteration
            iteration += 1
            if progress_every and iteration % progress_every == 0:
                print(f"  run={run} iteration={iteration} objective={intermediate_result.fun:.6f}", flush=True)

        result = minimize(problem.value_and_grad, x0, method="L-BFGS-B", jac=True,
                          bounds=problem.bounds, callback=progress,
                          options={"maxiter": maxiter, "ftol": ftol, "gtol": gtol,
                                   "maxls": 40, "maxcor": 30})
        if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
            raise FloatingPointError("Rule-centered optimization returned nonfinite values")
        if result.fun > initial_value:
            result.x = x0
            result.fun, result.jac = problem.value_and_grad(x0)
            result.success = False
            result.message = f"Retained initial point: {result.message}"
        candidates.append(result)
        runs.append(dict(run=run, initial_objective=initial_value, objective=float(result.fun),
                         success=bool(result.success), message=str(result.message),
                         n_iter=int(result.nit), n_evaluations=int(result.nfev),
                         gradient_inf_norm=float(np.abs(result.jac).max())))
    best_run = int(np.argmin([r.fun for r in candidates]))
    best = candidates[best_run]
    with torch.no_grad():
        d = problem.details(torch.tensor(best.x, dtype=torch.float64))
        transformed = model.transform_attention(d["alpha"]).numpy()
    fitted_params = dict(model.model_params)
    for name, value in d["params"].items():
        fitted_params[name] = value.numpy().copy() if name == "delta" else float(value)
    rule = {name: float(value) for name, value in d["rule"].items()}
    rule.update(regularization_p=regularization_p, loss_derivative=loss_derivative, update_clip=update_clip)
    sd_at_bound = bool(fit_transition_sd and any(np.isclose(rule["transition_sd"], b, rtol=1e-5, atol=1e-10)
                                                for b in transition_sd_bounds))
    settings = dict(
        prior="loss_regularization_centered_gaussian", trajectory=trajectory,
        n_basis_requested=n_basis, n_basis_used=problem.shape[0], fit_params=fit_params,
        fit_rule_params=fit_rule_params, fit_transition_sd=fit_transition_sd,
        transition_sd_bounds=transition_sd_bounds, rule_parameter_bounds=problem.rule_bounds,
        alpha_prior_mean=alpha_prior_mean, alpha_prior_sd=alpha_prior_sd, alpha_bounds=alpha_bounds,
        initial_alpha=initial_alpha, initial_coefficients=initial_coefficients,
        initial_rule_params=dict(lr=lr, regularization_strength=regularization_strength,
                                 transition_sd=transition_sd),
        parameter_bounds={name: (parameter_bounds or {}).get(name, DEFAULT_BOUNDS[name]) for name in fit_params},
        n_categories=model.n_categories, n_runs=n_runs, seed=seed, maxiter=maxiter, ftol=ftol, gtol=gtol,
    )
    return RuleCenteredFitResult(
        params=fitted_params, coefficients=d["coefficients"].numpy().copy(), basis=problem.basis.numpy().copy(),
        alpha_trajectory=d["alpha"].numpy().copy(), transformed_alpha=transformed.copy(),
        effective_sensitivity=transformed * fitted_params["delta"],
        decision_probs=d["probabilities"].numpy().copy(), observation_mask=problem.observed.numpy().copy(),
        neg_LL=float(d["nll"]), smoothness_penalty=0.0, initial_prior_penalty=float(d["prior"]),
        objective=float(d["nll"] + d["transition_nll"] + d["prior"]),
        initial_objective=runs[best_run]["initial_objective"], success=bool(best.success),
        message=str(best.message), n_iter=int(best.nit), n_observations=int(problem.observed.sum()),
        n_parameters=len(best.x), best_run=best_run, runs=runs, settings=settings,
        rule_params=rule, expected_next_alpha=d["expected_next_alpha"].numpy().copy(),
        transition_residuals=d["residuals"].numpy().copy(), loss_gradients=d["loss_gradients"].numpy().copy(),
        regularization_gradients=d["regularization_gradients"].numpy().copy(),
        transition_penalty=float(d["transition_penalty"]), transition_log_normalizer=float(d["transition_log_normalizer"]),
        transition_nll=float(d["transition_nll"]), transition_sd_at_bound=sd_at_bound)


class RuleCenteredConfig(EstimatedAlphaConfig):
    """The same cleaned-DataFrame adapter, using the rule-centered latent prior."""

    _fit_subject = staticmethod(fit_rule_centered_subject)
