# Estimated attention trajectories

This package fits the latent-attention version of `arm.models.StrengthModel`.
Attention is inferred from the entire behavioral sequence. It is **not** updated
from prediction error or an attention learning rate. Association weights still
follow the existing trial-by-trial learning equations.

For latent deviations centered on the original loss/regularization update, use
`RuleCenteredConfig` instead. See [the rule-centered guide](../../estimated_alpha/RULE_CENTERED.md).

The default is a cubic B-spline trajectory with 15 bases per attention dimension:

```text
alpha = B @ C                                # raw attention, shape (T, d)
attention = sigmoid(alpha)                   # default activation parameterization
activation[t, j] = exp(-sum(attention[t] * delta * abs(X[t] - hidden[j])))

objective = behavioral_NLL
          + lambda_alpha * sum((alpha[1:] - alpha[:-1])**2)
          + 0.5 * sum(((alpha[0] - alpha_prior_mean) / alpha_prior_sd)**2)
```

The spline coefficients and selected behavioral parameters are optimized
**jointly** with L-BFGS-B and float64 PyTorch autograd. Gradients include the
complete recursion through association weights: changing early attention can
change later predictions. No finite-difference gradients or differential
evolution are used in this package.

The result is a penalized-likelihood / joint MAP **point estimate**, conditional
on fixed smoothness and prior settings. It is not a marginal-likelihood fit,
posterior sampler, or uncertainty-band estimator. `lambda_alpha` is deliberately
not fitted jointly with the latent trajectory.

## Using your cleaned data in a notebook

From the project root, add `src` to the import path (or install the package):

```python
import sys
sys.path.insert(0, "./src")

import pandas as pd
from arm_estimated_alpha import EstimatedAlphaConfig

data = pd.read_csv("Data/clean_data/test_human_data_2.csv")
subject_data = data[data.subject_ID == data.subject_ID.iloc[0]].copy()

config = EstimatedAlphaConfig(
    model_params={
        "delta": [50.0, 50.0],
        "gamma_w": 0.1,
        "w_update_type": "prediction_error",
        "decision_rule": "luce",
        "n_points": 30,
    },
    fit_params=("gamma_w",),
    trajectory="spline",
    n_basis=15,
    lambda_alpha=1.0,
    maxiter=300,
)
config.fit(subject_data, single_subject=True, n_runs=1)
result = config.result

print(result.neg_LL)       # behavioral NLL only
print(result.objective)    # NLL + smoothness + initial-state prior
print(result.success, result.message)
attention = result.transformed_alpha       # (trials, dimensions), sigmoid(alpha)
alpha_raw = result.alpha_trajectory         # raw values before sigmoid
probabilities = result.decision_probs       # (trials, categories)

config.save("Switchboard/EstimatedAlpha/subject_1")
```

Use the full cleaned trial sequence for substantive fits. The small test CSVs are
useful for checking the workflow; omitted trials cannot contribute their weight
updates to a fit. The defaults preserve each subject's supplied row order and
continue learning across phase boundaries, just as the existing StrengthModel
does. Sort explicitly before fitting if your data are not already chronological;
trial numbers that restart each phase must be ordered by phase as well.

For every subject independently:

```python
config.fit(data, n_runs=1)
result = config.results[data.subject_ID.iloc[0]]
model_data = config.make_model_data()
config.save("Switchboard/EstimatedAlpha/all_subjects")
```

Each subject starts with fresh association weights and a separate attention
trajectory. `fit()` returns the config. `config.neg_LL` sums behavioral NLLs;
`config.objective` sums penalized objectives. No unadjusted AIC/BIC is supplied
because regularized trajectories complicate effective parameter counts.

The adapter uses the existing conventions:

- Features: `stim.Orientation`, `stim.Frequency`, divided by 100, in that order.
- Feedback: `truth`, one-based category labels. Every trial requires feedback.
- Observed response: `resp`, one-based labels, with NaN or 0 indicating missing.
- Grouping: `subject_ID`. Names/feature scaling can be overridden in `fit()`.
- A positional boolean `mask` changes likelihood inclusion only. Missing or
  excluded responses do **not** remove the trial or its feedback/weight update.

The fit conditions on the supplied stimuli and feedback; it does not require
the observed choice to match the correct category. Reconstructed attention uses
all included responses, including future responses. It is retrospective
inference, not an out-of-sample or online prediction score.

## Fitting more behavioral parameters

`model_params` gives fixed values and starting values. `fit_params` selects which
of them are estimated jointly with attention. For example:

```python
config = EstimatedAlphaConfig(
    model_params={"delta": [10.0, 10.0], "gamma_w": 0.1,
                  "decision_rule": "softmax", "beta": 1.0},
    fit_params=("gamma_w", "delta", "beta", "decay", "guessing"),
    n_basis=15,
    lambda_alpha=1.0,
)
```

| Parameter | Default value | Default fitting bounds |
| --- | --- | --- |
| `delta` | 50 in each dimension | 0.1–100, optimized on log10 scale |
| `gamma_w` | 0.1 | 0–0.95 |
| `decay` | 0 | 0–0.95 |
| `guessing` | 0.01 | 0–0.95 |
| `initialization_association` | 0 | −1–0, already a log10 parameter |
| `beta` | 1 | 0.1–10, optimized on log10 scale; softmax only |

`initialization_association` has the original model's meaning: weights start at
`10**initialization_association / grid_size` per category. Supply an initial `w`
matrix in `model_params` if needed; in that case its initialization parameter
cannot also be estimated. Fixed numeric values are not restricted to the fitting
bounds, but must be valid model parameters. Starting fitted values must lie
inside their bounds. Override bounds with, for example,
`parameter_bounds={"delta": (0.1, 300.0)}`.

By default only `gamma_w` is fitted alongside attention. Free `delta` and free
attention can compensate for each other because activation depends on their
product. The saved `effective_sensitivity = delta * transformed_alpha` helps
inspect that tradeoff. Association learning adds another potential tradeoff.
Fixing some behavioral parameters or checking recovery with simulated data is
useful before interpreting a fitted trajectory psychologically.

Supported weight rules are `hebbian`, `prediction_error`, and
`prediction_error_2`, exactly as implemented in the current StrengthModel.
`rescorla_wagner` is not an implemented rule in that source model. Decision rules
are `luce` and `softmax`. The alternate attention parameterization is
`attention_parameterization="none"`, which retains the original `10**alpha`
meaning; it does not disable attention.

## Trajectory and regularization settings

- `trajectory="spline"`, `n_basis=15`: fit K coefficients per dimension. K is
  capped at the trial count; fewer than four bases use a lower polynomial degree.
  `n_basis=1` fits constant attention. Knots are equally spaced over trial index.
- `trajectory="trial"`: fit T raw attention vectors directly; `n_basis` is unused.
  This is more expensive and relies more strongly on regularization.
- `lambda_alpha=1.0`: fixed penalty strength on **raw alpha differences**, not
  sigmoid attention or coefficient differences. The penalty is a sum, not a
  mean, and does not rescale by elapsed time. Try several values as a sensitivity
  analysis. Larger values favor smoother raw trajectories. For a Gaussian random
  walk, this corresponds to `lambda_alpha = 1 / (2 * sigma_alpha**2)`.
- `alpha_prior_mean=0.0`, `alpha_prior_sd=2.0`: initial raw-attention Gaussian
  prior. Its normalizing constant is omitted because its scale is fixed.
- `initial_alpha=0.0`: constant starting raw trajectory, i.e. sigmoid attention
  0.5. These three alpha settings also accept a mean/start vector per dimension
  (the prior SD is scalar).
- `alpha_bounds=(-6, 6)`: bounds on the raw spline coefficients or trial states.
  The nonnegative spline basis sums to one, so trajectories share those bounds.
  They imply sigmoid attention approximately 0.0025–0.9975. Use `None` for
  unbounded raw parameters or supply another pair.
- `n_runs=1`: one optimization start, fitting the complete trajectory jointly.
  Additional starts perturb attention and randomize fitted behavioral parameters
  within their bounds. The smallest **penalized objective** selects the result.
  Starts and subjects execute sequentially; there are no nested process pools.

Convergence is reported explicitly. Reaching `maxiter` still returns the best
of the final restart solutions with `success=False`; inspect `message` and
`runs` before treating it as a finished fit. This is a local optimizer and does
not guarantee a global optimum. `n_parameters` counts all coefficients and fitted
behavioral values; it is not an effective degrees-of-freedom estimate.

## Arrays and plotting

```python
from arm_estimated_alpha import fit_subject

result = fit_subject(
    X, feedback, responses,  # X already scaled; zero-based labels for this API
    model_params={"delta": [10.0, 10.0], "gamma_w": 0.1},
    fit_params=("gamma_w", "delta"),
    trajectory="trial",
    lambda_alpha=5.0,
    n_runs=1,
)

import matplotlib.pyplot as plt
plt.plot(result.transformed_alpha[:, 0], label="Orientation")
plt.plot(result.transformed_alpha[:, 1], label="Frequency")
plt.xlabel("Trial index")
plt.ylabel("Estimated attention")
plt.ylim(0, 1)  # for sigmoid parameterization
plt.legend()
```

The array API uses zero-based feedback/responses; responses of −1 or NaN are
missing. `n_categories` defaults to at least two, inferred from feedback, and
can be supplied explicitly for multicategory experiments.

## Command line / cluster

The package lives alongside `arm` in `src`, and the existing setuptools `arm*`
package discovery includes it. Install/update the source package in your active
environment with `python -m pip install -e ./src`, or use `PYTHONPATH=src`:

```bash
PYTHONPATH=src python -m arm_estimated_alpha \
  Data/clean_data/test_human_data_2.csv \
  --subject 1 \
  --output Switchboard/EstimatedAlpha/subject_1 \
  --n-runs 1 --n-basis 15 --lambda-alpha 1 --maxiter 300 --threads 1
```

Use `--trajectory trial` for trial-level latent states, or
`--fit-params gamma_w delta` for joint delta fitting. An empty `--fit-params`
fits attention alone. `--help` lists all CLI options.

This implementation runs on CPU and stores the full sequence's autograd graph.
Cost grows with trials, dimensions, and hidden grid size (`n_points**dimensions`).
The CLI defaults to one PyTorch CPU thread to avoid thread oversubscription on
small matrix operations; it can be overridden. The library itself does not alter
global thread settings. On a cluster, distribute independent subjects across
jobs with different `--subject` and `--output` values. One optimization run does
not automatically use every core: association updates depend on preceding trials.

Saved output contains `manifest.json`, per-subject `fit_0000.json` files, and
`model_data.csv` with input rows and attention/probability columns. The manifest
maps files to original subject IDs. JSON contains the basis, coefficients,
trajectory, model values, likelihood mask, objective terms, and restart settings.

## Validation

```bash
PYTHONPATH=src python -m pytest src/arm_estimated_alpha/tests -q
```

Tests compare probabilities to the existing StrengthModel across all supported
weight/decision/attention transformations, compare joint gradients with central
finite differences, check gradients through earlier weight updates, verify
masked-response semantics, and exercise fitting, subject grouping, and saving.

Implementation references: [SciPy B-splines](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html)
and [PyTorch autograd](https://docs.pytorch.org/docs/stable/autograd.html).
