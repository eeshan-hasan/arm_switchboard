from abc import ABC, abstractmethod
from copy import deepcopy
from .optimize import optimizer
import pickle
import pandas as pd
import os
import json
import numpy as np
import pandas as pd

import numpy as np
from tqdm import tqdm
import numpy as np

from .params import Params

class BaseModelConfig(ABC):
    def __init__(self, model_init=None, param_defs=Params):
        self.param_defs = param_defs
        self.model_init = model_init or {}

        self.init_params = {}
        self.fixed_params = {}
        self.estimated_params = []
        self.x_names = []

        self.use_inits(self.model_init)
        self.get_estimated_params()
        self.make_fixed_params()
        self.get_x_names()

    @abstractmethod 
    def use_inits(self, model_init):
        """Model-specific init parsing."""
        pass

    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(-x))

    @abstractmethod
    def get_negLL(self,data):
        """Get LL for a single participant"""
        pass

    @abstractmethod
    def get_estimated_params(self):
        """Model-specific rules for which params are fit."""
        pass

    def make_fixed_params(self):
        self.fixed_params = deepcopy(self.init_params)

        for name, value in list(self.fixed_params.items()):
            if value == "default":
                p = self.param_defs[name]
                if p.dim > 1:
                    self.fixed_params[name] = p.default 
                elif p.dim ==1:
                    self.fixed_params[name]= p.default[0]

        return self.fixed_params

    def get_x_names(self):
        ''' This opens up the arrays'''
        self.x_names = []

        for name in self.estimated_params:
            dim = self.param_defs[name].dim

            if dim > 1:
                self.x_names += [f"{name}_{i}" for i in range(dim)]
            else:
                self.x_names.append(name)

        return self.x_names
    
    def params_from_x(self,x):
        param_dict = {}
        cursor=0
        for name in self.estimated_params:
            p=self.param_defs[name]
            dim = p.dim
            if dim == 1:
                raw_x = x[cursor]
                param_dict[name] = p.transform(raw_x)
            if(dim>1):
                raw_x = np.asarray(x[cursor:cursor + dim], dtype=float)
                param_dict[name] = p.transform(raw_x)
            cursor += dim
        return param_dict

    def build_params_x(self, x):
        built_params = deepcopy(self.fixed_params)
        params_from_x=self.params_from_x(x)
        built_params.update(params_from_x)
        return built_params
    
    def get_x_from_params(self,params):
        x=[]
        for param in self.estimated_params:
            p = self.param_defs[param]
            dim=p.dim
            if(dim==1):
                x.append(p.inv_transform(params[param]))
            if(dim>1):
                x=list(x)+list(p.inv_transform(params[param]))
                
            print(param)
            print(dim)

        print(x)
        return x

    def get_bounds(self):

        bounds_list = []
        for name in self.estimated_params:
            for i in range(self.param_defs[name].dim):
                bounds_list.append((self.param_defs[name].bounds))
        return bounds_list

    def get_initial_guess(self,initial_guess='random'):
        x0 = []
        for param in self.estimated_params:
            dim = self.param_defs[param].dim
            bounds = self.param_defs[param].bounds
            if(initial_guess == 'random'):
                for i in range(dim):
                    x0_param = np.random.uniform(bounds[0],bounds[1])
                    x0.append(x0_param)
            if(initial_guess == 'default'):
                for i in range(dim):
                    x0_param = self.param_defs[param].init_value
                    x0.append(x0_param)
            if(initial_guess == 'median'):
                bounds_param = bounds[param]
                for i in range(dim):
                    x0_param = np.median(np.random.uniform(bounds[0],bounds[1]))
                    x0.append(x0_param)
        return x0
    
    def get_subject_IDs(self,data):
        return data.subject_ID.unique()
    

    def fit(self, data, single_subject=False, n_runs=10):
        """
        Fit model parameters.

        If single_subject=True, data should already contain one participant.
        Otherwise, data needs a subject_ID column.
        """

        if single_subject:
            best_p, res, res_all = optimizer(self, data, n_runs)

            self.best_params = best_p
            self.result = res
            self.neg_LL = res.fun
            self.detailed_fit = res_all

            print("Best Neg_LL =", self.neg_LL)
            return self.best_params

        best_params = {}
        results = {}
        neg_LLs = {}
        detailed_fits = {}
        optimizer_table = {}


        for subject in tqdm(data.subject_ID.unique()):
            data_subject = data[data["subject_ID"] == subject]

            best_p, res, res_all = optimizer(self, data_subject, n_runs)

            best_params[subject] = best_p
            results[subject] = res
            neg_LLs[subject] = res.fun
            detailed_fits[subject] = res_all
            optimizer_table[subject] = [res_i.fun for res_i in res_all]

        self.best_params = best_params
        self.results = results
        self.neg_LLs = neg_LLs
        self.neg_LL = np.sum(list(neg_LLs.values()))
        self.detailed_fits = res_all
        self.optimizer_table = optimizer_table

        print("Best Neg_LL =", self.neg_LL)
        return self

    
    def save(self, filename, foldername="./Switchboard/Test/"):
        import os
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        with open(f"{foldername}{filename}.pkl", "wb") as f:
            pickle.dump(self, f)

    def write_json(self, filename, foldername="./Switchboard/Test/"):

        if not os.path.exists(foldername):
            os.makedirs(foldername)

        data = {
            "model_init": self.model_init,
            "init_params": self.init_params,
            "fixed_params":self.fixed_params,
            "estimated_params": self.estimated_params,
            "summary": self.summary,
            "best_params": self.best_params,
            "optimizer_table" : self.optimizer_table
        }

        def convert_key(key):
            if isinstance(key, np.generic):
                key = key.item()

            if isinstance(key, (str, int, float, bool)) or key is None:
                return key

            return str(key)


        def json_converter(obj):

            if isinstance(obj, dict):
                return {
                    convert_key(key): json_converter(value)
                    for key, value in obj.items()
                }

            if isinstance(obj, pd.Series):
                return json_converter(obj.to_dict())

            if isinstance(obj, pd.DataFrame):
                return json_converter(obj.to_dict(orient="records"))

            if isinstance(obj, np.ndarray):
                return json_converter(obj.tolist())

            if isinstance(obj, np.generic):
                return obj.item()

            if isinstance(obj, (list, tuple)):
                return [json_converter(value) for value in obj]

            return obj


        with open(f"{foldername}{filename}.json", "w") as f:
            json.dump(json_converter(data), f, indent=4)

                
    @staticmethod
    def get_Xf(data):
        X=(data[['stim.Orientation','stim.Frequency']].values)/100
        f = (data['truth']-1).values
        return X,f
    
    @staticmethod
    def get_resp(data):
        return (data['resp']-1).values

    def make_model_data(self, data,params=None):
        model_data = []
        for subject_ID in data['subject_ID'].unique():
            subject_data = data[data['subject_ID']==subject_ID].copy()
            if(params == None):
                best_params_x=self.results[subject_ID].x
            else:
                best_params_x=self.get_x_from_params(params)
            built_params=self.build_params_x(best_params_x)
            simulation=self.run_model(data=subject_data,built_params=built_params)

            subject_data['model_prob_resp']=simulation.results[:,0]
            subject_data['model_prob_correct']=simulation.correct_prob
            subject_data['alpha_1']=simulation.alpha_trajectory[:,0]
            subject_data['alpha_2']=simulation.alpha_trajectory[:,1]

            resp = self.get_resp(subject_data)
            subject_data['model_prob_human_resp'] = simulation.results[np.arange(len(resp)), resp]

            subject_data['LL'] = subject_data['model_prob_resp'].sum()

            model_data.append(subject_data)
        self.model_data = pd.concat(model_data)
        return self.model_data

    def calculate_information_metrics(self):
        
        penalty_BIC=self.model_data.groupby(['subject_ID'])['resp'].count().apply(np.log)*len(self.get_x_names())
        penalty_AIC=2*len(self.get_x_names())
        self.AICs = pd.Series(self.neg_LLs)*2+penalty_AIC   
        self.BICs = pd.Series(self.neg_LLs)*2+penalty_BIC
        self.AIC = np.sum(self.AICs)
        self.BIC = np.sum(self.BICs)
        #RB AIC BIC
        model_data = self.model_data
        rb_ids=model_data[model_data['task_structure'] == 'RB']['subject_ID'].unique()
        ii_ids=model_data[model_data['task_structure'] == 'II']['subject_ID'].unique()

        self.AICs_rb = self.AICs.loc[rb_ids]
        self.AICs_ii = self.AICs.loc[ii_ids]

        self.BICs_rb = self.BICs.loc[rb_ids]
        self.BICs_ii = self.BICs.loc[ii_ids]

        self.AIC_rb = np.sum(self.AICs_rb)
        self.AIC_ii = np.sum(self.AICs_ii)

        self.BIC_rb = np.sum(self.BICs_rb)
        self.BIC_ii = np.sum(self.BICs_ii)

        self.summary = {
            'neg_LLs' : self.neg_LLs,
            'neg_LL': self.neg_LL,
            'AICs' : self.AICs,
            'BICs' :   self.BICs,
            'AIC': self.AIC,
            'BIC': self.BIC,
            'AIC_rb': self.AIC_rb,
            'BIC_rb': self.BIC_rb,
            'AIC_ii': self.AIC_ii,
            'BIC_ii': self.BIC_ii,
        }
            


