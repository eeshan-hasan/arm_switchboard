
def calc_AIC(estimatable_params,log_LL):
    k=count_params(estimatable_params,model_params)
    AIC = 2*k - 2*log_LL
    return AIC

def calc_BIC(estimatable_params,log_LL,data):
    n=len(data)
    k=count_params(estimatable_params)
    BIC=k * np.log(n) - 2 * log_LL
    return BIC

def count_params(estimatable_params):
    total_dims = sum(Params[param].dim for param in estimatable_params)
    return total_dims

def make_df(data):
    '''Individual df for participants'''
    for subject in data.subject_id.unique():
        df_subject = data


