"""Run with python -m arm_estimated_alpha from a notebook machine or cluster."""

import argparse

import pandas as pd
import torch

from .config import EstimatedAlphaConfig
from .model import NUMERIC_PARAMS


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fit latent attention trajectories to cleaned ARM data")
    parser.add_argument("csv", help="Cleaned CSV, with one-based truth and resp labels")
    parser.add_argument("--output", required=True, help="Directory for fitted data and parameters")
    parser.add_argument("--subject", help="Fit only this subject_ID (default: all subjects)")
    parser.add_argument("--trajectory", choices=("spline", "trial"), default="spline")
    parser.add_argument("--n-basis", type=int, default=15)
    parser.add_argument("--lambda-alpha", type=float, default=1.0)
    parser.add_argument("--alpha-prior-sd", type=float, default=2.0)
    parser.add_argument("--n-runs", type=int, default=1)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads per process")
    parser.add_argument("--n-points", type=int, default=30, help="Hidden-grid points per stimulus dimension")
    parser.add_argument("--delta", type=float, nargs="+", default=[50.0], help="One value or one per dimension (default: 50)")
    parser.add_argument("--gamma-w", type=float, default=0.1)
    parser.add_argument("--decision-rule", choices=("luce", "softmax"), default="luce")
    parser.add_argument("--w-update-type", choices=("hebbian", "prediction_error", "prediction_error_2"),
                        default="prediction_error")
    parser.add_argument("--fit-params", nargs="*", choices=NUMERIC_PARAMS, default=["gamma_w"],
                        help="Behavioral parameters to estimate jointly with attention")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    data = pd.read_csv(args.csv)
    if args.subject is not None:
        if "subject_ID" not in data:
            parser.error("CSV needs a subject_ID column")
        data = data[data["subject_ID"].astype(str) == args.subject]
        if data.empty:
            parser.error(f"No rows for subject {args.subject!r}")
    config = EstimatedAlphaConfig(
        model_params={"delta": args.delta, "gamma_w": args.gamma_w, "n_points": args.n_points,
                      "decision_rule": args.decision_rule, "w_update_type": args.w_update_type},
        fit_params=args.fit_params, trajectory=args.trajectory, n_basis=args.n_basis,
        lambda_alpha=args.lambda_alpha, alpha_prior_sd=args.alpha_prior_sd,
        maxiter=args.maxiter, seed=args.seed,
    )
    config.fit(data, n_runs=args.n_runs)
    config.save(args.output)
    print(f"Saved fits to {args.output}", flush=True)


if __name__ == "__main__":
    main()
