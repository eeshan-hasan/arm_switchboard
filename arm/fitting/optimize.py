def make_objective(model_config,data):
    def obj(x):
        params = model_config.build_params_x(x)
        ll= model_config.get_LL(params=params,experimenta_data=data)
        return ll
    return obj 


def find_best_box(model_config,initial_guess='random',seed=0):
    np.random.seed(seed)
    
    obj= make_objective(model_config,data)

    bounds_list=model_config.get_bounds()  
    x0 = model_config.get_x0(initial_guess='random')
    try:
        res=minimize(
            obj, x0, method="L-BFGS-B", bounds=bounds_list,
            options={"maxiter": 1000, "ftol": 1e-6, "disp": True}
        )
        best_params=build_params_from_x(res.x, estimated_params, model_params)
    except ValueError:#Nothing to estimate
        best_params = set_default_params_exp(model_params) 
        res = 'Did not Converge'
    return best_params, res


def find_best_box_repeated(switches,data,n_runs=2):
    best_params_candidates= []
    res_candidates = []
    results = Parallel(n_jobs=-1, backend='loky')(
    delayed(find_best_box)(
        switches,
        data,
        initial_guess='random',
        seed=i
    )
    for i in range(n_runs)
    )

    for i in range(n_runs):
        best_params, res = results[i]
        best_params_candidates.append(best_params)
        res_candidates.append(res)
    ll_candidates = [res.fun for res in res_candidates]
    best_params = best_params_candidates[np.argmin(ll_candidates)]
    res = res_candidates[np.argmin(ll_candidates)]
    return best_params,res
