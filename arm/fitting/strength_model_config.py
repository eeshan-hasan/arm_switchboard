from .model_config import BaseModelConfig
from instance_learning import models
from .params_strength import Params_Strength

class StrengthModelConfig(BaseModelConfig):
    def use_inits(self, model_init,param_defs=Params_Strength):
        update_type = model_init.get("attention_update_type", "p_regularization_1")

        mapping = {
            "none": ("none", None),
            "loss": ("loss", None),
            "p_regularization_2": ("p_regularization", 2),
            "p_regularization_1": ("p_regularization", 1),
            "p_regularization_0.75": ("p_regularization", 0.75),
            "p_regularization_0.5": ("p_regularization", 0.5),
        }

        attention_update_type, p = mapping.get(
            update_type,
            ("p_regularization", 1)
        )

        self.init_params["attention_update_type"] = attention_update_type

        if p is not None:
            self.init_params["regularization_p"] = p

        self.init_params["delta"] = model_init.get("delta", "fit_to_data")
        self.init_params["guessing"] = model_init.get("guessing", "default")

        self.init_params["loss_derivative"] = model_init.get("loss_derivative", "ce")
        self.init_params["attention_update_dims"] = model_init.get("attention_update_dims", "all")
        self.init_params["alpha_clip"] = model_init.get("alpha_clip", (-10000, 10000))

        self.init_params["initial_alpha"] = model_init.get("initial_alpha", "fit_to_data")
        if(self.init_params["attention_update_type"] == 'none'):
            self.init_params["initial_alpha"] = "default"

        self.init_params["w_update_type"] = model_init.get("w_update_type", "prediction_error")
        self.init_params["initialization_association"] = model_init.get("initialization_association","fit_to_data")

        return self
        
    @staticmethod
    def get_Xf(data):
        X=(data[['stim.Orientation','stim.Frequency']].values)/100
        f = (data['truth']-1).values
        return X,f
    
    @staticmethod
    def get_resp(data):
        return (data['resp']-1).values

    def get_negLL(self, data, mask=None):
        """Get negative LL for a single participant."""
        from arm.models import StrengthModel

        X, f = self.get_Xf(data)
        resp = self.get_resp(data)

        return StrengthModel(X, f, self.built_params).fit().neg_LL(resp, mask=mask)

    def get_estimated_params(self):
        self.estimated_params = []

        for name in ["delta", "decay", "guessing"]:
            if self.init_params.get(name) == "fit_to_data":
                self.estimated_params.append(name)
        if self.init_params.get("initial_alpha") == "fit_to_data":
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

        elif attention_update_type == "competition_regularization":
            self.estimated_params += [
                "lr",
                "competition_strength",
                "regularization_strength",
            ]

        if self.init_params.get("w_update_type") in {"prediction_error"}:
            self.estimated_params.append("gamma_w")

        if self.init_params.get("initialization_association") == "fit_to_data":
            self.estimated_params.append("initialization_association")

        return self.estimated_params
