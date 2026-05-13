import os

def run_switchboard_ind(switch_index, n_runs=2, test=False,switchboard_number=8,animal='human'):
    if(type(switch_index) is int):
        switches = get_switch(switch_index)
    else:
        switches = switch_index
    print(switches)
    data = read_data.read_data(animal=animal,test=test)
    save_path = './switchboard/'+str(switchboard_number)+'/'+str(switch_index)
    os.makedirs(save_path, exist_ok=True)
    print('saving at', save_path)

    # If there are already files in the folder, stop the loop
    if any(entry.is_file() for entry in os.scandir(save_path)):
        print('Files already exist in', save_path, '-> breaking')
        return
    
    #Get Model Config based on Switches

    #Model Config + Data object

    #Results=Model Config.fit()

    results = apply_individuals(switches,data=data,n_runs=n_runs)#X_vector, Convergence Info, Model_Params

    #Post_Process

    #Save Results

    #Make Plots

def get_Xf(data):
    X=(data[['stim.Orientation','stim.Frequency']].values)/100
    f = (data['truth']-1).values
    return X,f

    
def post_processing(results,switches,save_path):    
    save_params(results,switches=switches,save_path=save_path)
    plot_params(results,switches=switches,save_path=save_path)
    plot_params_condition(results,switches=switches,save_path=save_path)
    make_plots(results['model_data'],save_path=save_path)
    trajectories_plot(results=results,save_path=save_path)

def apply_individuals(switches,data,n_runs=2):
    params_subject =[]
    res_subject = []
    simulations = []
    model_datas = []
    subjects = []
    for subject in data.subject_ID.unique():
        data_subject = data[data['subject_ID']==subject]
        X,f = get_Xf(data)

        best_params,res=find_best_box_repeated(switches,data_subject,n_runs)
        model_data = calc_model_data(best_params,data_subject)
        X,f = get_Xf(data)
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