from .model_config import BaseModelConfig
from instance_learning import models
from .params_strength import Params_Strength
from arm.models import StrengthModel
import numpy as np


class StrengthModelConfig(BaseModelConfig):
    def use_inits(self, model_init,param_defs=Params_Strength):
        attention_update_type = model_init.get("attention_update_type", "p_regularization_1")
        if attention_update_type == 'p_regularization_1':
            self.init_params['attention_update_type'] = 'p_regularization'
            self.init_params['regularization_p'] = 1
        elif attention_update_type == 'p_regularization_2':
            self.init_params['attention_update_type'] = 'p_regularization'
            self.init_params['regularization_p'] = 2

        elif attention_update_type == 'p_regularization_0.5':
            self.init_params['attention_update_type'] = 'p_regularization'
            self.init_params['regularization_p'] = 0.5
        
        elif attention_update_type == 'p_regularization_0.75':
            self.init_params['attention_update_type'] = 'p_regularization'
            self.init_params['regularization_p'] = 0.75
        else:#none,loss,sum_to_constant
            self.init_params["attention_update_type"] = attention_update_type

        self.init_params["delta"] = model_init.get("delta", "fit_to_data")
        self.init_params["guessing"] = model_init.get("guessing", "default")
        self.init_params["decay"] = model_init.get("decay", "default")
        #self.init_params['response_bias'] = model_init.get('response_bias', 'default')

        self.init_params["loss_derivative"] = model_init.get("loss_derivative", "ce")
        self.init_params["attention_update_dims"] = model_init.get("attention_update_dims", "all")
        self.init_params["alpha_clip"] = model_init.get("alpha_clip", (-10000, 10000))

        self.init_params["initial_alpha"] = model_init.get("initial_alpha", "fit_to_data")
        if(self.init_params["attention_update_type"] == 'none'):
            self.init_params["initial_alpha"] = "default"

        self.init_params["w_update_type"] = model_init.get("w_update_type", "prediction_error")
        self.init_params["initialization_association"] = model_init.get("initialization_association","fit_to_data")

        self.init_params['decision_rule'] = model_init.get('decision_rule', 'luce')

        return self
        
    
    def get_negLL(self, built_params, data, mask=None):
        """Get negative LL for a single participant."""
        resp = self.get_resp(data)
        return self.run_model(data, built_params).neg_LL(resp, mask=mask)
   
    def run_model(self,data,built_params):
        X, f = self.get_Xf(data)
        return StrengthModel(X, f, built_params).predict_proba()

    def get_estimated_params(self):
        self.estimated_params = []

        for name in ["delta", "decay", "guessing"]:
            if self.init_params.get(name) == "fit_to_data":
                self.estimated_params.append(name)
        if(self.init_params.get('decision_rule') == 'softmax'):
            self.estimated_params.append('beta')

        if self.init_params.get("initial_alpha") == "fit_to_data":
            if self.init_params["attention_update_type"] == "sum_to_constant":
                self.estimated_params.append("initial_alpha_s")
                self.init_params["initial_alpha"]='default'
            if self.init_params["attention_update_type"] in ['loss','p_regularization']:
                if self.init_params["attention_update_dims"] == "all":
                    self.estimated_params.append("initial_alpha")
                elif self.init_params["attention_update_dims"] == "single":
                    self.estimated_params.append("initial_alpha_s")
            
        attention_update_type = self.init_params["attention_update_type"]

        if attention_update_type == "loss":
            self.estimated_params.append("lr")

        elif attention_update_type == "p_regularization":
            self.estimated_params += [
                "lr",
                "regularization_strength",
            ]
        elif attention_update_type == "sum_to_constant":
            self.estimated_params += [
                "lr"
            ]

        elif attention_update_type == "competition_regularization":
            self.estimated_params += [
                "lr",
                "competition_strength",
                "regularization_strength",
            ]

        if self.init_params.get("w_update_type") in {"hebbian","prediction_error"}:
            self.estimated_params.append("gamma_w")

        if self.init_params.get("initialization_association") == "fit_to_data":
            self.estimated_params.append("initialization_association")

        return self.estimated_params
