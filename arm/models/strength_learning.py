from typing import Any
from copy import deepcopy
from .models import Model

import numpy as np

verbose = False

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
    
    def fit(self):
        self.results = self.run_learning_trials()
        return self

    def post_processing():
        if np.isnan(decision_prob.sum()):
            correct_prob = np.zeros(len(self.f))
            decisions_realized = np.random.binomial(1, 0.5, len(self.f))
            accuracy = decisions_realized == self.f
        else:
            correct_prob = decision_prob[np.arange(self.f.size), self.f]
            if decision_prob.shape[1] == 2:
                decisions_realized = np.random.binomial(1, decision_prob[:, 1])
            else:
                decisions_realized = decision_prob.argmax(axis=1)
            accuracy = decisions_realized == self.f

        accuracy_max = decision_prob.argmax(axis=1) == self.f
        correct_prob = np.clip(correct_prob, 1e-12, 1.0)
        loss = -np.log(correct_prob)

    def set_defaults(self, params: dict[str, Any] | None) -> None:
        params = {} if params is None else params

        self.alpha = np.asarray(params.get("initial_alpha",np.ones(self.X.shape[1])))
        self.delta = params.get("delta",np.ones(self.X.shape[1]))
        self.gamma_w = params.get("gamma_w", 1)#Learning Rate

        self.initialization_association = 10 ** params.get("initialization_association", 0)
        self.n_points = params.get("n_points", 10)

        self.lr = params.get("lr", 0.1)
        self.attention_update_type = params.get("attention_update_type", "none")
        self.alpha_clip = params.get("alpha_clip", (-4, 4))
       
        self.w_update_type = params.get("w_update_type", "prediction_error")

        self.regularization_p = params.get("regularization_p", 1)
        self.regularization_strength = params.get("regularization_strength", 0.01)
        
        self.guessing = params.get("guessing", 0.01)
        
        self.partial_encoding = params.get("partial_encoding", True)
        self.attention_parameterization = params.get("attention_parameterization", "sigmoid")
        self.loss_derivative = params.get("loss_derivative", "sse")
        self.attention_update_dims = params.get("attention_update_dims", "all")
        self.save_trajectories = params.get("save_trajectories", True)

        if self.attention_update_dims == "single":
            self.initial_alpha = params.get("initial_alpha_s", np.mean(self.alpha))
            self.alpha = np.zeros(shape=self.delta.shape) + self.initial_alpha

        
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

            feedback_mat_active = self.feedback_mat[: idx + 1, :]#Why is this idx+1
            f_trial = feedback_mat_active[-1]

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
        return (E / np.sum(E)) * (1 - self.guessing) + (self.guessing / len(E))

    def update_weights(
        self,
        f: np.ndarray,
        activations: np.ndarray,
        D :np.ndarray
    ):
        #print('f',f)
        #print('D',D)
        #print(f-D)
        #print(activations.shape)
        #print(((f-D)*activations).shape)
        if self.w_update_type == "prediction_error":
            #print('Inside TD')
            #print('_________')
            self.w += (f-self.w)*activations*self.gamma_w
            #print(self.w)
        else:
            raise ValueError(f"Unknown w_update_type: {self.w_update_type}")

        if verbose:
            print(feedback_mat_active[-1], "Feedback for current exemplar")


    def calc_attention_loss_gradient(
        self,
        w: np.ndarray,
        activations: np.ndarray,
        probe: np.ndarray,
        y_true: int,
        alpha_trajectory: np.ndarray | list[Any] | None = None,
    ) -> np.ndarray:
        activations = np.asarray(activations).reshape(-1, 1)
        hidden_units = np.asarray(self.hidden_units)
        delta = np.asarray(self.delta).reshape(1, -1)
        alpha_trajectory = [] if alpha_trajectory is None else alpha_trajectory

        cat_ev = self.calc_evidence(activations)
        tot_ev = np.sum(cat_ev)
        den = tot_ev**2
        decision_prob = self.calc_evidence_decision(cat_ev)

        dervact = -activations * (delta * np.abs(hidden_units - probe))

        evi_deriv = dervact.T @ w
        sum_evi_deriv = evi_deriv.sum(axis=1, keepdims=True)
        dP = (tot_ev * evi_deriv - sum_evi_deriv * cat_ev[None, :]) / den

        p_true = decision_prob[y_true]

        if self.loss_derivative == "ce":
            grad = -dP[:, y_true] / p_true
        elif self.loss_derivative == "sse":
            grad = -2 * (1 - p_true) * dP[:, y_true]
        else:
            raise ValueError(f"Unknown loss_derivative: {self.loss_derivative}")

        if verbose:
            print("grad:", grad)

        return grad

    @staticmethod
    def calc_regularization(
        alpha: np.ndarray,
        p: float = 2,
        regularization_strength: float = 0.1,
    ) -> np.ndarray:
        return regularization_strength * p * (alpha ** (p - 1))

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
            grad_loss = self.calc_attention_loss_gradient(
                w=self.w,
                activations=activations,
                probe=x,
                y_true=int(np.argmax(f))            )

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

        elif self.attention_update_type == "p_regularization":
            grad_loss = self.calc_attention_loss_gradient(
                w=self.w,
                activations=activations,
                probe=x,
                y_true=int(np.argmax(f)),
                alpha_trajectory=alpha_trajectory,
            )

            if self.attention_parameterization == "sigmoid":
                sigmoid_alpha = self.sigmoid(self.alpha)
                dloss = grad_loss * sigmoid_alpha * (1 - sigmoid_alpha)
                dreg = (
                    self.calc_regularization(
                        sigmoid_alpha,
                        p=self.regularization_p,
                        regularization_strength=self.regularization_strength,
                    )
                    * sigmoid_alpha
                    * (1 - sigmoid_alpha)
                )

            elif self.attention_parameterization == "none":
                dloss = grad_loss * (10 ** self.alpha)
                dreg = self.calc_regularization(
                    self.alpha + 1,
                    p=self.regularization_p,
                    regularization_strength=self.regularization_strength,
                )
            else:
                raise ValueError(
                    f"Unknown attention_parameterization: {self.attention_parameterization}"
                )

        else:
            raise ValueError(f"Unknown attention_update_type: {self.attention_update_type}")

        if self.attention_update_dims == "all":
            new_alpha = self.alpha - self.lr * (dloss + dreg)
        elif self.attention_update_dims == "single":
            new_alpha = self.alpha - self.lr * (dloss + dreg).sum()
        else:
            raise ValueError(f"Unknown attention_update_dims: {self.attention_update_dims}")

        new_alpha = new_alpha.clip(self.alpha_clip[0], self.alpha_clip[1])

        if verbose:
            print("Updated alpha:", new_alpha)

        return new_alpha, dloss, dreg

    @classmethod
    def make_predictions(cls, x: np.ndarray, params: dict[str, Any]) -> dict[str, Any]:
        x = np.asarray(x)

        alpha = np.asarray(params["initial_alpha"], dtype=float)
        delta = np.asarray(params["delta"], dtype=float)
        exemplars = params.get("exemplars")
        w = params.get("w")
        guessing = params.get("guessing", 0.05)
        attention_parameterization = params.get("attention_parameterization", "sigmoid")
        partial_encoding = params.get("partial_encoding", True)

        if exemplars is None or w is None:
            raise ValueError("params must include fitted `exemplars` and `w` for prediction")

        exemplars = np.asarray(exemplars, dtype=float)
        w = np.asarray(w, dtype=float)

        decisions = []
        for point in x:
            d = cls.distances(point, delta, exemplars)

            if attention_parameterization == "none":
                transformed_alpha = 10 ** alpha
            elif attention_parameterization == "sigmoid":
                transformed_alpha = cls.sigmoid(alpha)
            else:
                raise ValueError(
                    f"Unknown attention_parameterization: {attention_parameterization}"
                )

            activations_mat = np.exp((-transformed_alpha) * d)
            activations = cls.calc_activations_hidden_units(activations_mat)
            E = cls.calc_evidence(activations, w)
            D = (E / np.sum(E)) * (1 - guessing) + (guessing / len(E))
            decisions.append(D)

        decisions = np.array(decisions)

        return {
            "trajectories": {
                "decision_prob": decisions,
                "x": x,
            },
            "state": {
                "w": w,
                "exemplars": exemplars,
                "initial_alpha": alpha,
                "delta": delta,
                "partial_encoding": partial_encoding,
                "attention_parameterization": attention_parameterization,
            },
        }


__all__ = [
    "StrengthModel",
]
