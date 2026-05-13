def save_summary_params(best_params,switches,res,data,save_path):
    model_params = get_model_params(switches)
    estimatable_params = get_estimatable_params(model_params)
    neg_ll_best=calc_ll_all(best_params,experimenta_data=data)
    summary=best_params.copy()
    summary['neg_ll']=neg_ll_best
    summary['BIC'] = calc_BIC(estimatable_params,model_params,-neg_ll_best,data)
    summary['AIC'] = calc_AIC(estimatable_params,model_params,-neg_ll_best)
    summary['human_readable']=get_human_readable(switches)
    summary['converged'] = res.success
    summary['res'] = res
    with open(save_path+'/summary.txt', "w") as f:  # "w" = overwrite or create
        f.write(summary.__str__())
    pickle.dump(summary,open(save_path+'/summary_pickle.pickle','wb'))
    return summary

def save_summary_params_ind(results,switches,save_path='.'): 
    results['main'].to_csv(save_path+'/main.csv')
    summary = {}
    summary['AIC'] = results['main']['AIC'].sum()
    summary['BIC'] = results['main']['BIC'].sum()
    summary['LL'] = results['main']['LL'].sum()
    summary['human_readable'] = get_human_readable(switches)
    summary['converged'] = results['main']['converged'].value_counts()
    with open(save_path+'/summary.txt', "w") as f:  # "w" = overwrite or create
        f.write(summary.__str__())
    results['summary']=summary
    pickle.dump(results,open(save_path+'/results_pickle.pickle','wb'))

def apply_individuals(switches,data,n_runs=2):
    params_subject =[]
    res_subject = []
    simulations = []
    model_datas = []
    subjects = []
    for subject in data.subject_ID.unique():
        data_subject = data[data['subject_ID']==subject]
        best_params,res=find_best_box_repeated(switches,data_subject,n_runs)
        model_data = calc_model_data(best_params,data_subject)
        X=(data_subject[['stim.Orientation','stim.Frequency']].values)/100
        f = (data_subject['truth']-1).values
        model_output = models.run_learning_trials(X,f,best_params)

        task_structure = model_data['task_structure'].iloc[0]
        acc=model_data['acc'].mean()
        model_acc=model_data['model_acc_prob'].mean()
        LL = res.fun
        converged = res.success

        model_params = get_model_params(switches)
        estimatable_params = get_estimatable_params(get_model_params(switches))
        AIC = calc_AIC(estimatable_params=estimatable_params,model_params=model_params,log_LL=-LL)
        BIC =  calc_BIC(estimatable_params=estimatable_params,model_params=model_params,log_LL=-LL,data=data_subject)
        

        subjects.append({'subject_ID':subject,'task_structure':task_structure,'accuracy':acc,
        'model_accuracy':model_acc,'LL':LL,'AIC':AIC,'BIC':BIC,'converged':converged})
        params_subject.append(best_params)
        simulations.append(model_output)
        model_datas.append(model_data)
        res_subject.append(res)

    results_data = pd.concat([pd.DataFrame(subjects),pd.DataFrame(params_subject)],axis=1)

    results = {
            'main':results_data,
            'params_subject':pd.DataFrame(params_subject),
            'res_subject':res_subject,
            'simulations' :simulations,
            'model_data':pd.concat(model_datas)
    }
    return results