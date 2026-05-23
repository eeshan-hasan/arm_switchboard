from abc import ABC, abstractmethod
from copy import deepcopy
from .optimize import optimizer

import numpy as np
from tqdm import tqdm

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
        self.built_params = deepcopy(self.fixed_params)
        params_from_x=self.params_from_x(x)
        self.built_params.update(params_from_x)

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
    
    def make_cv_masks(
        self,
        data_subject,
        scheme="blocked",
        test_frac=0.2,
        block_size=5,
        seed=None,
        trial_col=None,
    ):
        """
        Returns train_mask, test_mask aligned to data_subject rows.
        """

        if trial_col is not None:
            data_subject = data_subject.sort_values(trial_col)

        n = len(data_subject)
        idx = np.arange(n)

        if scheme == "even_odd":
            train_mask = idx % 2 == 0
            test_mask = ~train_mask

        elif scheme == "odd_even":
            test_mask = idx % 2 == 0
            train_mask = ~test_mask

        elif scheme == "random":
            rng = np.random.default_rng(seed)
            n_test = int(np.round(test_frac * n))
            test_idx = rng.choice(idx, size=n_test, replace=False)
            test_mask = np.isin(idx, test_idx)
            train_mask = ~test_mask

        elif scheme == "blocked":
            block_id = idx // block_size
            test_mask = block_id % 2 == 1
            train_mask = ~test_mask

        else:
            raise ValueError(f"Unknown CV scheme: {scheme}")

        return train_mask, test_mask

    def fit(self, data, single_subject=False, n_runs=10):
        """
        Fit model parameters.

        If single_subject=True, data should already contain one participant.
        Otherwise, data needs a subject_ID column.
        """

        if single_subject:
            best_p, res = optimizer(self, data, n_runs)

            self.best_params = best_p
            self.result = res
            self.neg_LL = res.fun

            print("Best Neg_LL =", self.neg_LL)
            return self.best_params

        best_params = {}
        results = {}
        neg_LLs = {}

        for subject in tqdm(data.subject_ID.unique()):
            data_subject = data[data["subject_ID"] == subject]

            best_p, res = optimizer(self, data_subject, n_runs)

            best_params[subject] = best_p
            results[subject] = res
            neg_LLs[subject] = res.fun

        self.best_params = best_params
        self.results = results
        self.neg_LLs = neg_LLs
        self.neg_LL = np.sum(list(neg_LLs.values()))

        print("Best Neg_LL =", self.neg_LL)
        return self.best_params

    def cross_validate(
        self,
        data,
        scheme="blocked",
        test_frac=0.2,
        block_size=5,
        n_runs=10,
        seed=None,
        trial_col=None,
    ):
        """
        Fit each subject on training trials and evaluate on held-out trials.

        Requires get_negLL(data, mask=None) to support masking.
        """

        cv_results = {}
        total_train_negLL = 0.0
        total_test_negLL = 0.0
        total_n_train = 0
        total_n_test = 0

        for subject in tqdm(data.subject_ID.unique()):
            data_subject = data[data["subject_ID"] == subject].copy()

            if trial_col is not None:
                data_subject = data_subject.sort_values(trial_col)

            train_mask, test_mask = self.make_cv_masks(
                data_subject,
                scheme=scheme,
                test_frac=test_frac,
                block_size=block_size,
                seed=seed,
                trial_col=None,
            )

            best_p, res = optimizer(
                self,
                data_subject,
                n_runs=n_runs,
                mask=train_mask,
            )

            self.built_params = deepcopy(self.fixed_params)
            self.built_params.update(best_p)

            train_negLL = self.get_negLL(data_subject, mask=train_mask)
            test_negLL = self.get_negLL(data_subject, mask=test_mask)

            cv_results[subject] = {
                "params": best_p,
                "result": res,
                "train_negLL": train_negLL,
                "test_negLL": test_negLL,
                "n_train": int(np.sum(train_mask)),
                "n_test": int(np.sum(test_mask)),
                "train_negLL_per_trial": train_negLL / np.sum(train_mask),
                "test_negLL_per_trial": test_negLL / np.sum(test_mask),
                "train_mask": train_mask,
                "test_mask": test_mask,
            }

            total_train_negLL += train_negLL
            total_test_negLL += test_negLL
            total_n_train += np.sum(train_mask)
            total_n_test += np.sum(test_mask)

        self.cv_results = cv_results
        self.cv_train_negLL = total_train_negLL
        self.cv_test_negLL = total_test_negLL
        self.cv_train_negLL_per_trial = total_train_negLL / total_n_train
        self.cv_test_negLL_per_trial = total_test_negLL / total_n_test

        print("CV Train Neg_LL =", self.cv_train_negLL)
        print("CV Test Neg_LL =", self.cv_test_negLL)
        print("CV Train Neg_LL per trial =", self.cv_train_negLL_per_trial)
        print("CV Test Neg_LL per trial =", self.cv_test_negLL_per_trial)

        return cv_results