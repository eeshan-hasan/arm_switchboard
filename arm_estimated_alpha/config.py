"""DataFrame interface using the same cleaned-data columns as arm."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .fitting import fit_subject, _jsonable


class EstimatedAlphaConfig:
    """Fit independent attention trajectories and behavioral parameters per subject.

    Pass fixed parameters and initial values in model_params; fit_params selects
    which numeric parameters are estimated together with the attention trajectory.
    Remaining keyword arguments are fit_subject options (e.g. n_basis).
    """

    _fit_subject = staticmethod(fit_subject)

    def __init__(self, model_params=None, fit_params=("gamma_w",), **fit_options):
        self.model_params = dict(model_params or {})
        self.fit_params = tuple(fit_params)
        self.fit_options = dict(fit_options)

    def fit(self, data, *, single_subject=False, n_runs=1, mask=None, verbose=True,
            feature_columns=("stim.Orientation", "stim.Frequency"), feature_scale=100.0,
            subject_column="subject_ID", feedback_column="truth", response_column="resp",
            **fit_options):
        """Fit a DataFrame; truth/resp labels are one-based, as in existing CSVs.

        Row order is preserved within subjects. No implicit sorting, phase resets,
        or dropped trials. NaN/0 responses are missing observations; truth is
        required on every trial because it drives the association updates.
        The optional mask is positional and only changes likelihood inclusion.
        """
        columns = [*feature_columns, feedback_column, response_column]
        if not single_subject:
            columns.append(subject_column)
        missing = set(columns) - set(data.columns)
        if missing:
            raise ValueError(f"Missing data columns: {sorted(missing)}")
        if data.empty:
            raise ValueError("Cannot fit an empty DataFrame")
        if not np.isfinite(feature_scale) or feature_scale <= 0:
            raise ValueError("feature_scale must be finite and positive")
        if subject_column in data and data[subject_column].isna().any():
            raise ValueError("Subject IDs may not be missing")
        if single_subject:
            if subject_column in data and data[subject_column].nunique() != 1:
                raise ValueError("single_subject=True requires exactly one subject")
            subject = data[subject_column].iloc[0] if subject_column in data else 0
            groups = [(subject, np.arange(len(data)))]
        else:
            codes, subjects = pd.factorize(data[subject_column], sort=False)
            groups = [(subject, np.flatnonzero(codes == i)) for i, subject in enumerate(subjects)]
        if mask is not None:
            mask = np.asarray(mask)
            if mask.shape != (len(data),) or mask.dtype != np.bool_:
                raise ValueError("mask must be a positional boolean array matching data length")
        options = {**self.fit_options, **fit_options}
        results = {}
        predictions = {}
        for subject, positions in groups:
            rows = data.iloc[positions]
            if verbose:
                print(f"Fitting subject {subject}: {len(rows)} trials", flush=True)
            result = self._fit_subject(
                rows[list(feature_columns)].to_numpy(dtype=float) / feature_scale,
                rows[feedback_column].to_numpy(dtype=float) - 1,
                rows[response_column].to_numpy(dtype=float) - 1,
                model_params=self.model_params, fit_params=self.fit_params, n_runs=n_runs,
                mask=None if mask is None else mask[positions], **options,
            )
            results[subject] = result
            for name, values in result.to_frame().items():
                if name not in predictions:
                    # A category may occur in only some subjects. Leave absent
                    # columns missing rather than exposing uninitialized values.
                    predictions[name] = (np.zeros(len(data), dtype=bool) if values.dtype == bool
                                         else np.full(len(data), np.nan))
                predictions[name][positions] = values.to_numpy()
            if verbose:
                print(f"  NLL={result.neg_LL:.6f}; objective={result.objective:.6f}; "
                      f"converged={result.success}; {result.message}", flush=True)
        self.results = results
        self.best_params = {subject: result.params for subject, result in results.items()}
        self.alpha_trajectories = {subject: result.alpha_trajectory for subject, result in results.items()}
        self.neg_LLs = {subject: result.neg_LL for subject, result in results.items()}
        self.neg_LL = sum(self.neg_LLs.values())
        self.objective = sum(result.objective for result in results.values())
        self.model_data = data.copy()
        for name, values in predictions.items():
            self.model_data[name] = values
        if single_subject:
            self.result = next(iter(results.values()))
        elif hasattr(self, "result"):
            del self.result
        return self

    def make_model_data(self):
        """Return input data plus trial-aligned fitted attention/probabilities."""
        if not hasattr(self, "model_data"):
            raise RuntimeError("Call fit() first")
        return self.model_data.copy()

    def save(self, folder):
        """Save per-subject JSON, a manifest, and a CSV in original row order."""
        if not hasattr(self, "results"):
            raise RuntimeError("Call fit() first")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        entries = []
        for i, (subject, result) in enumerate(self.results.items()):
            filename = f"fit_{i:04d}.json"
            result.save(folder / filename)
            entries.append({"subject_ID": subject, "file": filename,
                            "neg_LL": result.neg_LL, "objective": result.objective,
                            "success": result.success})
        (folder / "manifest.json").write_text(json.dumps(_jsonable(entries), indent=2, allow_nan=False))
        self.model_data.to_csv(folder / "model_data.csv", index=False)


# An explicit name for callers who keep multiple model config types together.
EstimatedAlphaModelConfig = EstimatedAlphaConfig
