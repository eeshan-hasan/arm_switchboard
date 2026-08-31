from typing import Any
from copy import deepcopy
from .models import Model

import numpy as np

verbose = False
quick = False

class StrengthModel(Model):
    def __init__(self, X: np.ndarray, f: np.ndarray, params: dict[str, Any] | None):
        """
        X is the coordinates.
        f is the feedback/category labels in integer format, e.g. 0, 1, ...
        params is a dict with the model parameters.
        """
        self.X = np.asarray(X)

        self.f = np.asarray(f, dtype=int)
        self.initialize_feedback()

        self.set_defaults(params)
        self.initialize_trajectories()
        self.results= ''
    
    def predict_proba(self):
        '''Just runs the model'''
        self.results = self.run_learning_trials()
        return self

    def set_defaults(self, params: dict[str, Any] | None) -> None:
        params = {} if params is None else params

        self.alpha = np.asarray(params.get("initial_alpha",np.ones(self.X.shape[1])))
        self.delta = params.get("delta",np.ones(self.X.shape[1]))
        self.gamma_w = params.get("gamma_w", 1)#Learning Rate

        self.initialization_association = 10 ** params.get("initialization_association", 0)
        if(quick):
            self.n_points = params.get("n_points", 10)
        else:
            self.n_points = params.get("n_points", 100)

        self.lr = params.get("lr", 0.1)
        self.attention_update_type = params.get("attention_update_type", "none")
        self.alpha_clip = params.get("alpha_clip", (-4, 4))
       
        self.w_update_type = params.get("w_update_type", "prediction_error")
        self.decay = params.get('decay',0)

        self.regularization_p = params.get("regularization_p", 1)
        self.regularization_strength = params.get("regularization_strength", 0.01)
        
        self.guessing = params.get("guessing", 0.01)
        
        self.attention_parameterization = params.get("attention_parameterization", "sigmoid")
        self.loss_derivative = params.get("loss_derivative", "ce")
        self.attention_update_dims = params.get("attention_update_dims", "all")
        self.save_trajectories = params.get("save_trajectories", True)

        self.decision_rule = params.get("decision_rule", "luce")#or softmax
        self.beta = params.get("beta", 1.0)#default is 1

        if self.attention_update_dims == "single":
            self.initial_alpha = params.get("initial_alpha_s", np.mean(params.get('initial_alpha',0)))            
            self.initial_alpha = np.zeros(shape=self.delta.shape) + np.mean(self.initial_alpha)
            self.alpha = self.initial_alpha.copy()
        
        if self.attention_update_type == "sum_to_constant":
 
            self.initial_alpha_s=params.get("initial_alpha_s",np.mean(self.alpha))
            self.initial_alpha = np.array([self.initial_alpha_s,-self.initial_alpha_s])            
            self.alpha = self.initial_alpha.copy()

        
        mins = np.min(self.X, axis=0)
        maxs = np.max(self.X, axis=0)
        grid_axes = [
            np.linspace(lo, hi, self.n_points)
            for lo, hi in zip(mins, maxs)
        ]
        background_grid = np.stack(
            np.meshgrid(*grid_axes),
            axis=-1,
        ).reshape(-1, self.X.shape[1])

        self.hidden_units = background_grid

        self.n_categories = len(np.unique(self.f))

        if params.get("w") is None:
            self.grid_size = self.hidden_units.shape[0]
            self.w = np.zeros((self.grid_size, self.n_categories))
            self.w[: , :] = (
                self.initialization_association / max(1, self.grid_size)
            )
        else:
            self.w = np.asarray(params["w"], dtype=float)

    def initialize_trajectories(self) -> None:
        self.all_w: list[np.ndarray] = []
        self.decisions: list[np.ndarray] = []
        self.dloss: list[np.ndarray] = []
        self.dreg: list[np.ndarray] = []
        self.all_alpha = np.zeros((self.X.shape[0], self.alpha.shape[0]))

    def run_learning_trials(self) -> dict[str, Any]:
        for idx in range(len(self.X)):
            probe = self.X[idx]

            activations_mat = self.calc_activations(probe)
            activations = self.calc_activations_hidden_units(activations_mat)

            E = self.calc_evidence(activations)
            D = self.calc_evidence_decision(E)

            f_trial = self.feedback_mat[idx]

            prev_alpha = deepcopy(self.alpha)
            self.alpha, dloss_trial, dreg_trial = self.update_attention(
                activations=activations,
                x=probe,
                f=f_trial,
                D=D
            )
            self.update_weights(
                f=f_trial,
                activations=activations,
                D = D
            )
            #print(activations)
            #print(self.w)

            self.decisions.append(D)
            if self.save_trajectories:
                self.all_w.append(self.w.copy())
                self.all_alpha[idx, :] = prev_alpha
                self.dloss.append(dloss_trial)
                self.dreg.append(dreg_trial)
        self.decision_probs = np.array(self.decisions)
        self.alpha_trajectory = self.all_alpha[:,:]
        self.correct_prob = self.decision_probs[np.arange(len(self.f)), self.f]
        
        return self.decision_probs

    @staticmethod
    def sigmoid(x: np.ndarray | float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x)),
        )

    @staticmethod
    def distances(x: np.ndarray, delta: np.ndarray, hidden_units: np.ndarray) -> np.ndarray:
        return delta * np.abs(x - hidden_units)

    def calc_activations(
        self,
        x: np.ndarray) -> np.ndarray:
        d = self.distances(x, self.delta, self.hidden_units)

        if self.attention_parameterization == "none":
            transformed_alpha = 10 ** self.alpha
        elif self.attention_parameterization == "sigmoid":
            transformed_alpha = self.sigmoid(self.alpha)
        else:
            raise ValueError(
                f"Unknown attention_parameterization: {self.attention_parameterization}"
            )
        activations = np.exp((-transformed_alpha) * d)

        return activations

    @staticmethod
    def calc_activations_hidden_units(activations: np.ndarray) -> np.ndarray:
        activations = np.prod(activations, axis=1)
        return activations.reshape(len(activations), -1)

    def calc_evidence(self,activations: np.ndarray) -> np.ndarray:
        if verbose:
            print("Activations:", activations)
            print("Weights:", self.w)
        return (activations * self.w).sum(axis=0) + 1e-100

    def calc_evidence_decision(self, E: np.ndarray) -> np.ndarray:
        if self.decision_rule == "luce":
            return (E / np.sum(E)) * (1 - self.guessing) + (self.guessing / len(E))
        if self.decision_rule == "softmax":
            exp_E = np.exp(self.beta*(E - np.max(E)))
            return (exp_E / np.sum(exp_E)) * (1 - self.guessing) + (self.guessing / len(E))
        else:
            raise ValueError(f"Unknown decision_rule: {self.decision_rule}")

    def update_weights(
        self,
        f: np.ndarray,
        activations: np.ndarray,
        D :np.ndarray
    ):
        self.w = self.w*(1-self.decay)
        
        if self.w_update_type == "hebbian":
            self.w += (self.gamma_w)*f*activations
        
        elif self.w_update_type == "prediction_error":
            self.w += self.gamma_w*(f-self.w)*activations
            self.w = np.maximum(self.w,0)

        elif self.w_update_type == "prediction_error_2":
            self.w += (f-D)*activations*self.gamma_w
            self.w = np.maximum(self.w,0)

        else:

            raise ValueError(f"Unknown w_update_type: {self.w_update_type}")
        
        

    def calc_attention_loss_gradient(
        self,
        w: np.ndarray,
        activations: np.ndarray,
        probe: np.ndarray,
        y_true: int,
    ) -> np.ndarray:
        activations = np.asarray(activations).reshape(-1, 1)
        hidden_units = np.asarray(self.hidden_units)
        delta = np.asarray(self.delta).reshape(1, -1)

        if(self.decision_rule == 'luce'):

            cat_ev = self.calc_evidence(activations)
            tot_ev = np.sum(cat_ev)
            den = tot_ev**2
            decision_prob = self.calc_evidence_decision(cat_ev)

            dervact = -activations * (delta * np.abs(hidden_units - probe))

            evi_deriv = dervact.T @ w
            sum_evi_deriv = evi_deriv.sum(axis=1, keepdims=True)
            dP = (tot_ev * evi_deriv - sum_evi_deriv * cat_ev[None, :]) / den
        
        if(self.decision_rule == 'softmax'):
            #cat_ev = self.calc_evidence(activations)
            #exp_E = np.exp(self.beta*cat_ev)
            #decision_prob = exp_E / np.sum(exp_E)
            #dervact = -activations * (delta * np.abs(hidden_units - probe))
            #dE = dervact.T @ w
            #derv_exp_E = self.beta * dE * exp_E
            #den = np.sum(exp_E)**2
            #dP = (np.sum(exp_E) * derv_exp_E - np.sum(derv_exp_E, axis=0) * exp_E) / den

            cat_ev = self.calc_evidence(activations)

            # Stable softmax
            z = self.beta * cat_ev
            z -= np.max(z)
            exp_z = np.exp(z)
            decision_prob = exp_z / np.sum(exp_z)

            dervact = -activations * (delta * np.abs(hidden_units - probe))
            dE = dervact.T @ w

            # Stable derivative of softmax
            dP = self.beta * decision_prob * (
                dE - np.sum(decision_prob * dE)
            )

        p_true = decision_prob[y_true]
        dP = (1-self.guessing)*dP

        if self.loss_derivative == "ce":
            grad = -dP[:, y_true] / p_true
        elif self.loss_derivative == "sse":
            grad = -2 * (1 - p_true) * dP[:, y_true]
        else:
            raise ValueError(f"Unknown loss_derivative: {self.loss_derivative}")

        if verbose:
            print("grad:", grad)

        return grad

    def calc_regularization(self,alpha: np.ndarray) -> np.ndarray:
        return self.regularization_strength * self.regularization_p * (alpha ** (self.regularization_p - 1))

    def update_attention(
        self,
        activations: np.ndarray,
        x: np.ndarray,
        f: np.ndarray,
        D: np.ndarray    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        if self.attention_update_type == "none":
            dloss = np.zeros(shape=self.alpha.shape)
            dreg = np.zeros(shape=self.alpha.shape)

        elif self.attention_update_type == "loss":
            grad_loss = self.calc_attention_loss_gradient(w=self.w,activations=activations,probe=x,y_true=int(np.argmax(f)))
            if self.attention_parameterization == "none":
                dloss = grad_loss * (10 ** self.alpha)
            elif self.attention_parameterization == "sigmoid":
                sigmoid_alpha = self.sigmoid(self.alpha)
                dloss = grad_loss * sigmoid_alpha * (1 - sigmoid_alpha)
            else:
                raise ValueError(
                    f"Unknown attention_parameterization: {self.attention_parameterization}")
            dreg = np.zeros(shape=self.alpha.shape)

        elif self.attention_update_type == "p_regularization":
            grad_loss = self.calc_attention_loss_gradient(
                w=self.w,
                activations=activations,
                probe=x,
                y_true=int(np.argmax(f))            )

            if self.attention_parameterization == "sigmoid":
                sigmoid_alpha = self.sigmoid(self.alpha)
                dloss = grad_loss * sigmoid_alpha * (1 - sigmoid_alpha)
                dreg = (
                    self.calc_regularization(
                        sigmoid_alpha)
                    * sigmoid_alpha
                    * (1 - sigmoid_alpha)
                )

            elif self.attention_parameterization == "none":
                dloss = grad_loss * (10 ** self.alpha)
                dreg = self.calc_regularization(
                    self.alpha + 1)
            else:
                raise ValueError(
                    f"Unknown attention_parameterization: {self.attention_parameterization}"
                )

        if(self.attention_update_type == 'none'):
            new_alpha = self.alpha
        
        if(self.attention_update_type in ['loss','p_regularization']):
            if self.attention_update_dims == "all":
                new_alpha = self.alpha - self.lr * (dloss + dreg)
            elif self.attention_update_dims == "single":
                new_alpha = self.alpha - self.lr * (dloss + dreg).sum()
        
        if self.attention_update_type == "sum_to_constant":
            grad_loss = self.calc_attention_loss_gradient(w=self.w,activations=activations,probe=x,y_true=int(np.argmax(f)))
            if self.attention_parameterization == "none":
                dloss = grad_loss * (10 ** self.alpha)
            elif self.attention_parameterization == "sigmoid":
                sigmoid_alpha = self.sigmoid(self.alpha)
                dloss = grad_loss * sigmoid_alpha * (1 - sigmoid_alpha)
            else:
                raise ValueError(
                    f"Unknown attention_parameterization: {self.attention_parameterization}"
                )

            dreg = np.zeros(shape=self.alpha.shape)

            new_alpha = np.zeros(self.alpha.shape)
            new_alpha[0] = self.alpha[0] - self.lr * (dloss[0] + dreg[0]) + self.lr * (dloss[1] + dreg[1])
            new_alpha[1] = self.alpha[1] - self.lr * (dloss[1] + dreg[1]) + self.lr * (dloss[0] + dreg[0])
        if(self.attention_update_type not in ['none','p_regularization','loss','sum_to_constant']):
            raise ValueError(f"Unknown attention_update_type: {self.attention_update_type}")

        new_alpha = new_alpha.clip(self.alpha_clip[0], self.alpha_clip[1])

        if verbose:
            print("Updated alpha:", new_alpha)

        return new_alpha, dloss, dreg

__all__ = [
    "StrengthModel",
]
