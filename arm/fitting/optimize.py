import numpy as np
from scipy.optimize import minimize
from joblib import Parallel, delayed


def make_objective(model_config, data, mask=None):
    def obj(x):
        model_config.build_params_x(x)
        neg_LL = model_config.get_negLL(data=data, mask=mask)
        return neg_LL
    return obj


def find_best_box(model_config, data, initial_guess="random", seed=0, mask=None):
    np.random.seed(seed)

    obj = make_objective(model_config, data, mask=mask)

    bounds_list = model_config.get_bounds()
    x0 = model_config.get_initial_guess(initial_guess=initial_guess)

    try:
        res = minimize(
            obj,
            x0,
            method="L-BFGS-B",
            bounds=bounds_list,
            options={
                "maxiter": 1000,
                "ftol": 1e-6,
                "disp": False,
            },
        )
        best_params = model_config.params_from_x(res.x)

    except KeyError:
        # Nothing to estimate
        best_params = model_config.params_from_x(x0)
        res = None

    return best_params, res


def find_best_box_repeated(model_config, data, n_runs=2, mask=None):
    results = Parallel(n_jobs=-1, backend="loky")(
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

    for best_params, res in results:
        best_params_candidates.append(best_params)
        res_candidates.append(res)

    valid_idx = [i for i, res in enumerate(res_candidates) if res is not None]

    if len(valid_idx) == 0:
        return best_params_candidates[0], res_candidates[0]

    ll_candidates = [res_candidates[i].fun for i in valid_idx]
    best_i = valid_idx[np.argmin(ll_candidates)]

    return best_params_candidates[best_i], res_candidates[best_i]


def optimizer(model_config, data, n_runs=10, mask=None):
    if n_runs == 1:
        return find_best_box(model_config, data, mask=mask)

    return find_best_box_repeated(
        model_config,
        data,
        n_runs=n_runs,
        mask=mask,
    )


__all__ = ["optimizer"]