"""Numerical checks of the behavioral equations and complete-sequence gradients."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from arm.models.strength_learning import StrengthModel
from arm_estimated_alpha import (
    EstimatedAlphaConfig, EstimatedAlphaModel, attention_basis, fit_subject,
)
from arm_estimated_alpha.fitting import _FitProblem


X = np.array([[0.1, 0.2], [0.8, 0.7], [0.3, 0.6], [0.9, 0.2], [0.5, 0.8], [0.2, 0.9]])
FEEDBACK = np.array([0, 1, 0, 1, 1, 0])
RESPONSES = np.array([1, 1, 0, 1, 0, 0])
PARAMS = {"delta": [3.0, 5.0], "gamma_w": 0.18, "decay": 0.05,
          "guessing": 0.04, "initialization_association": -0.5, "beta": 1.3, "n_points": 3}


@pytest.fixture(autouse=True, scope="module")
def small_tensor_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("n_trials,n_basis", [(1, 15), (2, 15), (3, 2), (30, 15), (7, 1)])
def test_basis_covers_endpoints_and_constant_attention(n_trials, n_basis):
    basis = attention_basis(n_trials, n_basis=n_basis)
    assert basis.shape == (n_trials, min(n_trials, n_basis))
    assert (basis >= -1e-14).all()
    np.testing.assert_allclose(basis.sum(axis=1), 1)
    np.testing.assert_allclose(basis @ np.ones((basis.shape[1], 2)), 1)
    assert basis[0, 0] == 1
    assert basis[-1, -1] == 1


@pytest.mark.parametrize("rule", ["hebbian", "prediction_error", "prediction_error_2"])
@pytest.mark.parametrize("decision", ["luce", "softmax"])
@pytest.mark.parametrize("parameterization", ["sigmoid", "none"])
def test_probabilities_match_original_strength_model(rule, decision, parameterization):
    alpha = np.array([[-0.5, 0.3], [0.1, -0.2], [0.3, 0.6],
                      [-0.1, 0.7], [0.9, 0.2], [0.2, -0.1]])
    params = {**PARAMS, "w_update_type": rule, "decision_rule": decision,
              "attention_parameterization": parameterization}

    class ReplayStrengthModel(StrengthModel):
        """Feed the same externally supplied alpha into the existing equations."""
        cursor = 0

        def calc_activations(self, x):
            self.alpha = alpha[self.cursor].copy()
            self.cursor += 1
            return super().calc_activations(x)

    reference = ReplayStrengthModel(X, FEEDBACK, {**params, "attention_update_type": "none"}).predict_proba()
    model = EstimatedAlphaModel(X, FEEDBACK, params)
    actual = model(torch.tensor(alpha, dtype=torch.float64)).detach().numpy()
    np.testing.assert_allclose(actual, reference.decision_probs, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(actual.sum(axis=1), 1)
    # Every evaluation resets w; optimizer evaluations cannot leak state.
    np.testing.assert_array_equal(actual, model(alpha).detach().numpy())


def problem(**overrides):
    options = dict(trajectory="spline", n_basis=3, lambda_alpha=0.7,
                   alpha_prior_mean=0.1, alpha_prior_sd=2.0, initial_alpha=0.2,
                   alpha_bounds=(-6.0, 6.0), fit_params=("delta", "gamma_w", "decay", "guessing"),
                   parameter_bounds=None, mask=None)
    options.update(overrides)
    return _FitProblem(EstimatedAlphaModel(X, FEEDBACK, PARAMS), RESPONSES, **options)


@pytest.mark.parametrize("trajectory", ["spline", "trial"])
def test_joint_gradient_matches_finite_differences(trajectory):
    p = problem(trajectory=trajectory)
    vector = p.x0.copy()
    vector[:p.n_alpha] += np.linspace(-0.2, 0.3, p.n_alpha)
    _, gradient = p.value_and_grad(vector)
    numerical = np.empty_like(vector)
    eps = 1e-6
    for i in range(len(vector)):
        step = np.zeros_like(vector)
        step[i] = eps
        numerical[i] = (p.value_and_grad(vector + step)[0] - p.value_and_grad(vector - step)[0]) / (2 * eps)
    np.testing.assert_allclose(gradient, numerical, rtol=2e-5, atol=1e-7)


def test_late_response_has_gradient_through_early_weight_updates():
    model = EstimatedAlphaModel(X, FEEDBACK, PARAMS)
    alpha = torch.zeros((len(X), 2), dtype=torch.float64, requires_grad=True)
    loss = -torch.log(model(alpha)[-1, RESPONSES[-1]])
    gradient, = torch.autograd.grad(loss, alpha)
    assert gradient[0].abs().sum() > 1e-6


def test_mask_excludes_likelihood_but_preserves_learning_trials():
    mask = np.array([False, False, False, False, False, True])
    p = problem(mask=mask)
    parts = p.components(torch.tensor(p.x0, dtype=torch.float64))
    assert float(parts[0]) == pytest.approx(-np.log(float(parts[-1][-1, RESPONSES[-1]])))
    full = problem().components(torch.tensor(p.x0, dtype=torch.float64))
    torch.testing.assert_close(parts[-1], full[-1])


@pytest.mark.parametrize("trajectory", ["spline", "trial"])
def test_fit_improves_objective_and_saves_consistent_result(tmp_path, trajectory):
    result = fit_subject(X, FEEDBACK, RESPONSES, model_params=PARAMS, n_basis=3,
                         trajectory=trajectory, fit_params=("gamma_w", "delta"),
                         maxiter=80, n_runs=2, seed=4)
    assert result.objective < result.initial_objective - 1e-3
    assert result.objective == pytest.approx(
        result.neg_LL + result.smoothness_penalty + result.initial_prior_penalty)
    assert result.neg_LL == pytest.approx(-np.log(result.decision_probs[np.arange(len(X)), RESPONSES]).sum())
    np.testing.assert_allclose(result.basis @ result.coefficients, result.alpha_trajectory)
    np.testing.assert_allclose(result.effective_sensitivity, result.transformed_alpha * result.params["delta"])
    assert result.n_parameters == result.coefficients.size + 3
    assert result.objective == pytest.approx(min(run["objective"] for run in result.runs))
    result.save(tmp_path / "result.json")
    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved["settings"]["trajectory"] == trajectory
    np.testing.assert_allclose(saved["alpha_trajectory"], result.alpha_trajectory)
    replay = EstimatedAlphaModel(X, FEEDBACK, saved["params"])(saved["alpha_trajectory"])
    np.testing.assert_allclose(replay.detach().numpy(), result.decision_probs)


def test_missing_responses_and_single_trial():
    result = fit_subject(X[:1], FEEDBACK[:1], RESPONSES[:1], model_params=PARAMS, maxiter=3)
    assert result.smoothness_penalty == 0
    assert result.n_observations == 1
    responses = RESPONSES.astype(float)
    responses[0], responses[2] = np.nan, -1
    result = fit_subject(X, FEEDBACK, responses, model_params=PARAMS, n_basis=2, maxiter=3)
    assert result.n_observations == 4
    assert result.decision_probs.shape == (6, 2)


def test_dataframe_subjects_reset_and_keep_interleaved_order(tmp_path):
    frame = pd.DataFrame({"stim.Orientation": np.repeat(X[:, 0], 2) * 100,
                          "stim.Frequency": np.repeat(X[:, 1], 2) * 100,
                          "truth": np.repeat(FEEDBACK + 1, 2),
                          "resp": np.repeat(RESPONSES + 1, 2),
                          "subject_ID": ["A", "B"] * len(X)})
    frame.index = [1] * len(frame)  # Positional alignment must survive duplicate indexes.
    config = EstimatedAlphaConfig(model_params=PARAMS, n_basis=3, maxiter=10)
    config.fit(frame, verbose=False)
    np.testing.assert_array_equal(config.results["A"].alpha_trajectory, config.results["B"].alpha_trajectory)
    output = config.make_model_data()
    np.testing.assert_array_equal(output["subject_ID"], frame["subject_ID"])
    np.testing.assert_allclose(output["attention_0"][::2], config.results["A"].transformed_alpha[:, 0])
    assert config.neg_LL == pytest.approx(sum(r.neg_LL for r in config.results.values()))
    config.save(tmp_path)
    assert (tmp_path / "model_data.csv").exists()
    assert len(json.loads((tmp_path / "manifest.json").read_text())) == 2
    with pytest.raises(ValueError, match="exactly one subject"):
        config.fit(frame, single_subject=True)


@pytest.mark.parametrize("options,match", [
    ({"n_runs": 0}, "n_runs"),
    ({"lambda_alpha": -1}, "lambda_alpha"),
    ({"alpha_prior_sd": 0}, "alpha_prior_sd"),
    ({"fit_params": ("beta",)}, "softmax"),
    ({"mask": np.zeros(6, dtype=bool)}, "At least one"),
    ({"mask": np.ones(6)}, "boolean"),
    ({"fit_params": ("lr",)}, "fit_params"),
])
def test_invalid_fits_fail_clearly(options, match):
    with pytest.raises(ValueError, match=match):
        fit_subject(X, FEEDBACK, RESPONSES, model_params=PARAMS, **options)
