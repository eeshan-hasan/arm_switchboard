import numpy as np
from scipy.optimize import minimize, differential_evolution
from joblib import Parallel, delayed


def make_objective(model_config, data, mask=None):
    def obj(x):
        group_built_params_list = model_config.build_params_group(x)

        subject_IDs=data['subject_ID'].unique()

        neg_LL = np.sum([ 
        model_config.get_negLL(
                    built_params=group_built_params_list[i],
                    data=data[data['subject_ID']==subject_IDs[i]],
                    mask=mask,
                ) for i in range(model_config.n_subjects)
                ])

        # Protect optimizers from NaN / inf
        if not np.isfinite(neg_LL):
            return 1e100

        return neg_LL

    return obj

def objective(x, model_config, data, mask=None):
    group_built_params_list = model_config.build_params_group(x)

    subject_IDs=data['subject_ID'].unique()

    
    neg_LL = np.sum([ 
        model_config.get_negLL(
                    built_params=group_built_params_list[i],
                    data=data[data['subject_ID']==subject_IDs[i]],
                    mask=mask,
                ) for i in range(model_config.n_subjects)
                ])
    return neg_LL if np.isfinite(neg_LL) else 1e100


def find_best_box(
    model_config,
    data,
    initial_guess="random",
    seed=0,
    mask=None,
):
    np.random.seed(seed)

    obj = make_objective(
        model_config,
        data,
        mask=mask,
    )

    bounds_list = model_config.get_bounds_group()

    try:
        # -----------------------------------------------------
        # 1. GLOBAL SEARCH: Differential Evolution
        # -----------------------------------------------------
        de_res = differential_evolution(
            objective,
            bounds=bounds_list,
            args=(model_config, data, mask),
            seed=seed,

            # Population size is roughly:
            # popsize * number_of_parameters
            popsize=30,

            maxiter=300,

            # Convergence criterion for DE
            tol=1e-7,

            # We do our own L-BFGS-B polish below
            polish=False,

            # IMPORTANT:
            # Parallelization happens across n_runs using joblib,
            # so don't also parallelize inside DE.
            workers=25,
        )

        # -----------------------------------------------------
        # 2. LOCAL POLISH: L-BFGS-B
        # -----------------------------------------------------
        local_res = minimize(
            obj,
            de_res.x,
            method="L-BFGS-B",
            bounds=bounds_list,
            options={
                "maxiter": 500,
                "ftol": 1e-12,
                "gtol": 1e-6,
                "maxls": 50,
                "disp": False,
            },
        )

        # -----------------------------------------------------
        # 3. Keep whichever solution is actually better
        # -----------------------------------------------------
        if (
            np.isfinite(local_res.fun)
            and local_res.fun <= de_res.fun
        ):
            res = local_res
        else:
            res = de_res

        best_params = model_config.params_fit(res.x)

    except KeyError:
        # Nothing to estimate
        x0 = model_config.get_initial_guess(
            initial_guess=initial_guess
        )

        best_params = model_config.params_fit(x0)
        res = None

    return best_params, res, [res]#This is to have a consistent function signature with the parallel one.


def find_best_box_repeated(
    model_config,
    data,
    n_runs=10,
    mask=None,
):
    results = Parallel(
        n_jobs=-1,
        backend="loky",
    )(
        delayed(find_best_box)(
            model_config,
            data,
            initial_guess="random",
            seed=i,
            mask=mask,
        )
        for i in range(n_runs)
    )

    best_params_candidates = []
    res_candidates = []

    for best_params, res, _ in results:
        best_params_candidates.append(best_params)
        res_candidates.append(res)

    valid_idx = [
        i
        for i, res in enumerate(res_candidates)
        if res is not None and np.isfinite(res.fun)
    ]

    if len(valid_idx) == 0:
        return best_params_candidates[0], res_candidates[0]

    negll_candidates = [
        res_candidates[i].fun
        for i in valid_idx
    ]

    best_i = valid_idx[np.argmin(negll_candidates)]

    # Useful diagnostics
    print("\nOptimization results:")
    for i in valid_idx:
        res = res_candidates[i]

        grad_norm = np.nan

        if hasattr(res, "jac") and res.jac is not None:
            grad_norm = np.max(np.abs(res.jac))

        print(
            f"run={i:2d}  "
            f"NegLL={res.fun:.6f}  "
            f"success={res.success}  "
            f"nit={getattr(res, 'nit', None)}  "
            f"grad_inf={grad_norm:.3g}"
        )

    print(
        f"\nBest Neg_LL = "
        f"{res_candidates[best_i].fun}"
    )

    return (
        best_params_candidates[best_i],
        res_candidates[best_i],
        res_candidates
    )


def optimizer_group(
    model_config,
    data,
    n_runs=3,
    mask=None,
):
    if n_runs == 1:
        return find_best_box(
            model_config,
            data,
            mask=mask,
        )

    return find_best_box_repeated(
        model_config,
        data,
        n_runs=n_runs,
        mask=mask,
    )


__all__ = ["optimizer_group"]
