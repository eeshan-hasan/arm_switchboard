"""Temporal bases for raw (pre-sigmoid) attention."""

import numpy as np
from scipy.interpolate import BSpline


def attention_basis(n_trials, trajectory="spline", n_basis=15):
    """Return B with alpha = B @ coefficients, in the supplied trial order.

    Splines are clamped cubic B-splines with equally spaced interior knots.
    Short sequences use fewer bases and, if necessary, a lower degree.
    ``trial`` uses an identity basis, giving one attention vector per trial.
    """
    if not isinstance(n_trials, (int, np.integer)) or n_trials < 1:
        raise ValueError("n_trials must be a positive integer")
    if trajectory == "trial":
        return np.eye(n_trials, dtype=float)
    if trajectory != "spline":
        raise ValueError("trajectory must be 'spline' or 'trial'")
    if not isinstance(n_basis, (int, np.integer)) or n_basis < 1:
        raise ValueError("n_basis must be a positive integer")
    k = min(n_basis, n_trials)
    if k == 1:
        return np.ones((n_trials, 1))
    degree = min(3, k - 1)
    interior = np.linspace(0, 1, k - degree + 1)[1:-1]
    knots = np.r_[np.zeros(degree + 1), interior, np.ones(degree + 1)]
    return BSpline(knots, np.eye(k), degree)(np.linspace(0, 1, n_trials))
