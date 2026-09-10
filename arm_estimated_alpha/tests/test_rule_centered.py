"""Check the transition drift against the original rule and its outer gradients."""

import json
import math

import numpy as np
import pandas as pd
import pytest
import torch

from arm.models.strength_learning import StrengthModel
from arm_estimated_alpha import RuleCenteredModel, RuleCenteredConfig, fit_rule_centered_subject
from arm_estimated_alpha.rule_centered import _RuleCenteredProblem


X = np.array([[0.1, 0.2], [0.8, 0.7], [0.3, 0.6], [0.9, 0.2], [0.5, 0.8], [0.2, 0.9]])
F = np.array([0, 1, 0, 1, 1, 0])
Y = np.array([1, 1, 0, 1, 0, 0])
PARAMS = dict(delta=[3., 5.], gamma_w=0.18, decay=0.05, guessing=0.04,
              initialization_association=-0.5, n_points=3)


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("rule", ["hebbian", "prediction_error", "prediction_error_2"])
@pytest.mark.parametrize("p", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("loss", ["ce", "sse"])
def test_drift_matches_original_rule(rule, p, loss):
    alpha = np.array([[-0.5, 0.3], [0.1, -0.2], [0.3, 0.6],
                      [-0.1, 0.7], [0.9, 0.2], [0.2, -0.1]])

    class Reference(StrengthModel):
        cursor = 0

        def calc_activations(self, x):
            self.alpha = alpha[self.cursor].copy()
            self.cursor += 1
            return super().calc_activations(x)

        def update_attention(self, **kwargs):
            out = super().update_attention(**kwargs)
            expected.append(out)
            return out

    expected = []
    params = dict(PARAMS, w_update_type=rule)
    ref = Reference(X, F, dict(params, attention_update_type="p_regularization", lr=0.7,
                              regularization_strength=0.12, regularization_p=p,
                              loss_derivative=loss, alpha_clip=(-6, 6))).predict_proba()
    model = RuleCenteredModel(X, F, params, regularization_p=p, loss_derivative=loss, update_clip=(-6, 6))
    probs, means, dloss, dreg = model.rule_statistics(alpha, lr=0.7, regularization_strength=0.12)
    np.testing.assert_allclose(probs.numpy(), ref.decision_probs, rtol=1e-12, atol=1e-14)
    for actual, j in ((means, 0), (dloss, 1), (dreg, 2)):
        np.testing.assert_allclose(actual.numpy(), np.array([out[j] for out in expected]),
                                   rtol=1e-11, atol=1e-13)


def make_problem(**overrides):
    opts = dict(trajectory="spline", n_basis=3, alpha_prior_mean=0., alpha_prior_sd=2.,
                initial_alpha=0.2, alpha_bounds=(-6., 6.), fit_params=("gamma_w", "delta"),
                parameter_bounds=None, mask=None, lr=0.7, regularization_strength=0.12,
                fit_rule_params=("lr", "regularization_strength"), rule_parameter_bounds=None,
                transition_sd=0.2, fit_transition_sd=False, transition_sd_bounds=None)
    opts.update(overrides)
    return _RuleCenteredProblem(RuleCenteredModel(X, F, PARAMS), Y, **opts)


@pytest.mark.parametrize("trajectory,fit_sd", [("spline", False), ("trial", True)])
def test_outer_gradient_includes_gradient_of_the_update_rule(trajectory, fit_sd):
    problem = make_problem(trajectory=trajectory, fit_transition_sd=fit_sd,
                           transition_sd_bounds=(0.01, 1.) if fit_sd else None)
    vector = problem.x0.copy()
    vector[:problem.n_alpha] += np.linspace(-0.2, 0.3, problem.n_alpha)
    _, actual = problem.value_and_grad(vector)
    numerical = np.zeros_like(vector)
    for i in range(len(vector)):
        step = np.zeros_like(vector)
        step[i] = 1e-6
        numerical[i] = (problem.value_and_grad(vector + step)[0]
                        - problem.value_and_grad(vector - step)[0]) / 2e-6
    np.testing.assert_allclose(actual, numerical, rtol=2e-5, atol=2e-7)


def test_zero_drift_reduces_to_gaussian_random_walk():
    problem = make_problem(lr=0., regularization_strength=0., fit_rule_params=())
    vector = problem.x0.copy()
    vector[:problem.n_alpha] += np.linspace(-0.2, 0.3, problem.n_alpha)
    d = problem.details(torch.tensor(vector))
    expected = d["alpha"][1:] - d["alpha"][:-1]
    torch.testing.assert_close(d["residuals"], expected)
    assert float(d["transition_penalty"]) == pytest.approx(float(0.5 * (expected / 0.2).square().sum()))
    assert float(d["transition_log_normalizer"]) == pytest.approx(expected.numel() * math.log(0.2 * math.sqrt(2 * math.pi)))


def test_exact_legacy_trajectory_has_zero_transition_error():
    params = dict(PARAMS, initial_alpha=[0.3, -0.2], attention_update_type="p_regularization",
                  regularization_p=1., regularization_strength=0.12, lr=0.7, alpha_clip=(-6, 6))
    ref = StrengthModel(X, F, params).predict_proba()
    model = RuleCenteredModel(X, F, PARAMS, update_clip=(-6, 6))
    _, means, _, _ = model.rule_statistics(ref.alpha_trajectory, lr=0.7, regularization_strength=0.12)
    np.testing.assert_allclose(ref.alpha_trajectory[1:] - means.numpy()[:-1], 0, atol=1e-12)


def test_mask_changes_observed_likelihood_only():
    mask = np.array([False, False, False, False, False, True])
    full, masked = make_problem(), make_problem(mask=mask)
    d1 = full.details(torch.tensor(full.x0))
    d2 = masked.details(torch.tensor(masked.x0))
    torch.testing.assert_close(d1["transition_nll"], d2["transition_nll"])
    torch.testing.assert_close(d1["expected_next_alpha"], d2["expected_next_alpha"])
    assert float(d2["nll"]) == pytest.approx(-math.log(float(d1["probabilities"][-1, Y[-1]])))


def test_fitted_sigma_contains_normalization_term():
    problem = make_problem(fit_transition_sd=True, transition_sd_bounds=(0.01, 1.))
    x1 = problem.x0.copy()
    x1[:problem.n_alpha] += np.linspace(-0.2, 0.3, problem.n_alpha)
    x2 = x1.copy()
    x2[problem.rule_slices["transition_sd"]] += np.log10(2)
    d1, d2 = [problem.details(torch.tensor(x)) for x in (x1, x2)]
    assert float(d2["transition_penalty"]) == pytest.approx(float(d1["transition_penalty"]) / 4)
    assert float(d2["transition_log_normalizer"] - d1["transition_log_normalizer"]) == pytest.approx(10 * np.log(2))


def test_fit_serialization_and_dataframe_adapter(tmp_path):
    frame = pd.DataFrame({"subject_ID": 1, "stim.Orientation": X[:, 0] * 100,
                          "stim.Frequency": X[:, 1] * 100, "truth": F + 1, "resp": Y + 1})
    config = RuleCenteredConfig(model_params=PARAMS, n_basis=3, maxiter=120,
                                transition_sd=0.2, fit_rule_params=())
    config.fit(frame, single_subject=True, verbose=False)
    result = config.result
    assert result.objective < result.initial_objective
    assert result.objective == pytest.approx(result.neg_LL + result.transition_nll + result.initial_prior_penalty)
    assert result.smoothness_penalty == 0
    assert result.rule_params["transition_sd"] == 0.2
    np.testing.assert_allclose(result.alpha_trajectory[1:] - result.expected_next_alpha, result.transition_residuals)
    assert np.isnan(config.model_data.rule_predicted_alpha_raw_0.iloc[0])
    np.testing.assert_allclose(config.model_data.rule_predicted_alpha_raw_0.iloc[1:], result.expected_next_alpha[:, 0])
    config.save(tmp_path)
    saved = json.loads((tmp_path / "fit_0000.json").read_text())
    assert saved["settings"]["prior"] == "loss_regularization_centered_gaussian"
    assert saved["transition_nll"] == pytest.approx(result.transition_nll)


@pytest.mark.parametrize("options,match", [
    ({"transition_sd": 0.}, "transition_sd"),
    ({"fit_transition_sd": True}, "explicit positive"),
    ({"fit_transition_sd": True, "transition_sd_bounds": (0., 1.)}, "positive"),
    ({"fit_rule_params": ("delta",)}, "fit_rule_params"),
    ({"regularization_p": -1}, "regularization_p"),
])
def test_invalid_prior_is_rejected(options, match):
    with pytest.raises(ValueError, match=match):
        fit_rule_centered_subject(X, F, Y, model_params=PARAMS, **options)


def test_estimated_noise_scale_reports_the_lower_bound_for_zero_residuals():
    result = fit_rule_centered_subject(
        X, F, Y, model_params=PARAMS, n_basis=1, fit_params=(), fit_rule_params=(),
        lr=0., regularization_strength=0., transition_sd=0.2,
        fit_transition_sd=True, transition_sd_bounds=(0.02, 1.), maxiter=40)
    assert result.rule_params["transition_sd"] == pytest.approx(0.02)
    assert result.transition_sd_at_bound
    np.testing.assert_allclose(result.transition_residuals, 0, atol=1e-12)


def test_unsupported_update_parameterization_fails_explicitly():
    with pytest.raises(ValueError, match="decision_rule='luce'"):
        RuleCenteredModel(X, F, {**PARAMS, "decision_rule": "softmax"})
    with pytest.raises(ValueError, match="sigmoid"):
        RuleCenteredModel(X, F, {**PARAMS, "attention_parameterization": "none"})
