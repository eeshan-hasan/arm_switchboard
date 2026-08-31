from typing import Any
from copy import deepcopy
from .models import Model

import numpy as np

verbose = False

class InstanceModel(Model):
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
    
    def predict_proba(self):
        self.results = self.run_learning_trials()
        return self
    
    def LL(self,responses):
        if self.results is None:
            raise RuntimeError("Call fit() before LL().")
        return (self.decision_probs[np.arange(len(responses)),responses])
    
    def neg_LL(self, responses, mask=None):
        probs = self.LL(responses)

        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            probs = probs[mask]

        return -np.sum(np.log(probs + 1e-12))

    def post_processing(self):
        if np.isnan(self.decision_probs.sum()):
            correct_prob = np.zeros(len(self.f))
            decisions_realized = np.random.binomial(1, 0.5, len(self.f))
            accuracy = self.decisions_realized == self.f
        else:
            correct_prob = self.decision_probs[np.arange(self.f.size), self.f]
            if self.decision_probs.shape[1] == 2:
                self.decisions_realized = np.random.binomial(1, self.decision_probs[:, 1])
            else:
                self.decisions_realized = self.decision_probs.argmax(axis=1)
            accuracy = self.decisions_realized == self.f

        accuracy_max = self.decision_probs.argmax(axis=1) == self.f
        correct_prob = np.clip(correct_prob, 1e-12, 1.0)
        loss = -np.log(correct_prob)

    def set_defaults(self, params: dict[str, Any] | None) -> None:
        if params is None:
            params = {}

        self.initialization = params.get("initialization", "grid")
        self.initialization_association = 10 ** params.get("initialization_association", 0)
        self.n_points = params.get("n_points", 10)

        self.alpha = np.asarray(params["initial_alpha"], dtype=float)
        self.delta = np.asarray(params["delta"], dtype=float)

        self.lr = params.get("lr", 0.1)
        self.attention_update_type = params.get("attention_update_type", "none")
        self.alpha_clip = params.get("alpha_clip", (-3, 4))
        self.gamma_w = params.get("gamma_w", 1)
        self.w_update_type = params.get("w_update_type", "perfect_instances")
        self.regularization_p = params.get("regularization_p", 1)
        self.regularization_strength = params.get("regularization_strength", 0.01)
        self.reg_growth = params.get("reg_growth", 0)

        self.decay = params.get("decay", 0.05)
        self.guessing = params.get("guessing", 0.05)
        self.partial_encoding = params.get("partial_encoding", True)
        self.attention_parameterization = params.get("attention_parameterization", "sigmoid")
        self.loss_derivative = params.get("loss_derivative", "sse")
        self.attention_update_dims = params.get("attention_update_dims", "all")
        self.save_trajectories = params.get("save_trajectories", True)

        if self.attention_update_dims == "single":
            self.initial_alpha = params.get("initial_alpha_s", self.alpha)
            self.alpha = np.zeros(shape=self.delta.shape) + np.mean(self.initial_alpha)

        self.exemplars_override = params.get("exemplars")

        self.decision_rule = params.get("decision_rule", "luce")#or softmax
        self.beta = params.get("beta", 1.0)#default is 1

        if self.initialization == "point":
            self.exemplars = (
                np.asarray(self.exemplars_override, dtype=float)
                if self.exemplars_override is not None
                else np.array([np.mean(self.X, axis=0)])
            )

        elif self.initialization == "grid":
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

            self.exemplars = (
                np.asarray(self.exemplars_override, dtype=float)
                if self.exemplars_override is not None
                else background_grid
            )

        else:
            raise ValueError(f"Unknown initialization: {self.initialization}")

        self.n_categories = len(np.unique(self.f))

        if params.get("w") is None:
            if self.initialization == "point":
                self.w = np.zeros((len(self.X) + 1, self.n_categories))
                self.w[0, :] = self.initialization_association / max(1, self.n_categories)
                self.counter = 1
            else:
                self.grid_size = self.exemplars.shape[0]
                self.w = np.zeros((len(self.X) + self.grid_size, self.n_categories))
                self.w[: self.grid_size, :] = (
                    self.initialization_association / max(1, self.grid_size)
                )
                self.counter = self.grid_size
        else:
            self.w = np.asarray(params["w"], dtype=float)
            self.counter = len(self.w)
            self.w = np.vstack([
                self.w,
                np.zeros((len(self.X), self.n_categories)),
            ])

    def initialize_trajectories(self) -> None:
        self.all_w: list[np.ndarray] = []
        self.decisions: list[np.ndarray] = []
        self.dloss: list[np.ndarray] = []
        self.dreg: list[np.ndarray] = []

        self.all_alpha = np.zeros((self.w.shape[0], self.alpha.shape[0]))

        if self.initialization == "point":
            self.all_alpha[0, :] = self.alpha
        else:
            self.all_alpha[: self.exemplars.shape[0], :] = self.alpha

    def run_learning_trials(self) -> dict[str, Any]:
        for idx in range(len(self.X)):
            current_idx = len(self.exemplars)
            probe = self.X[idx]

            w_active = self.w[:current_idx, :]
            all_alpha_active = self.all_alpha[:current_idx, :]

            activations_mat = self.calc_activations(
                probe,
                alpha_trajectory=all_alpha_active,
            )
            activations = self.calc_activations_exemplars(activations_mat)

            E = self.calc_evidence(activations, w_active)
            D = self.calc_evidence_decision(E)

            feedback_mat_active = self.feedback_mat[: idx + 1, :]
            f_trial = feedback_mat_active[-1]

            if current_idx == 1:
                prev_alpha = deepcopy(self.alpha)
                dloss_trial = np.zeros(shape=self.alpha.shape)
                dreg_trial = np.zeros(shape=self.alpha.shape)
            else:
                prev_alpha = deepcopy(self.alpha)
                self.alpha, dloss_trial, dreg_trial = self.update_attention(
                    activations=activations,
                    w=w_active,
                    x=probe,
                    f=f_trial,
                    D=D,
                    alpha_trajectory=all_alpha_active,
                )

            self.w[: current_idx + 1, :] = self.update_weights(
                w=self.w[: current_idx + 1, :],
                feedback_mat_active=feedback_mat_active,
                activations=activations,
            )

            self.update_exemplars(probe)
            self.decisions.append(D)

            if self.save_trajectories:
                self.all_w.append(self.w.copy())
                self.all_alpha[current_idx, :] = prev_alpha
                self.dloss.append(dloss_trial)
                self.dreg.append(dreg_trial)
        self.decision_probs = np.array(self.decisions)
        self.alpha_trajectory = self.all_alpha[self.grid_size:,:]
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
    def distances(x: np.ndarray, delta: np.ndarray, exemplars: np.ndarray) -> np.ndarray:
        return delta * np.abs(x - exemplars)

    def calc_activations(
        self,
        x: np.ndarray,
        alpha_trajectory: np.ndarray | list[Any] | None = None,
    ) -> np.ndarray:
        d = self.distances(x, self.delta, self.exemplars)
        alpha_trajectory = [] if alpha_trajectory is None else alpha_trajectory

        if self.attention_parameterization == "none":
            transformed_alpha = 10 ** self.alpha
            transformed_alpha_history = 10 ** np.asarray(alpha_trajectory)
        elif self.attention_parameterization == "sigmoid":
            transformed_alpha = self.sigmoid(self.alpha)
            transformed_alpha_history = self.sigmoid(np.asarray(alpha_trajectory))
        else:
            raise ValueError(
                f"Unknown attention_parameterization: {self.attention_parameterization}"
            )

        if self.partial_encoding and len(alpha_trajectory):
            activations = np.exp((-transformed_alpha) * d * transformed_alpha_history)
        else:
            activations = np.exp((-transformed_alpha) * d)

        return activations

    @staticmethod
    def calc_activations_exemplars(activations: np.ndarray) -> np.ndarray:
        activations = np.prod(activations, axis=1)
        return activations.reshape(len(activations), -1)

    @staticmethod
    def calc_evidence(activations: np.ndarray, w: np.ndarray) -> np.ndarray:
        if verbose:
            print("Activations:", activations)
            print("Weights:", w)
        return (activations * w).sum(axis=0) + 1e-100

    def calc_evidence_decision(self, E: np.ndarray) -> np.ndarray:
        if self.decision_rule == "luce":
            return (E / np.sum(E)) * (1 - self.guessing) + (self.guessing / len(E))
        if self.decision_rule == "softmax":
            exp_E = np.exp(self.beta*E - np.max(E))
            return (exp_E / np.sum(exp_E)) * (1 - self.guessing) + (self.guessing / len(E))
        else:
            raise ValueError(f"Unknown decision_rule: {self.decision_rule}")

    def update_exemplars(self, x: np.ndarray) -> None:
        if len(self.exemplars) == 0:
            self.exemplars = np.array([x])
        else:
            self.exemplars = np.vstack([self.exemplars, x])

    def update_weights_RW(self,
        w_i: np.ndarray,
        f_i: np.ndarray,
        act: np.ndarray,
        gamma_w: float = 1,
    ) -> np.ndarray:
        error = f_i - w_i
        return w_i + self.gamma_w * error * act

    @staticmethod
    def update_weights_noisy_feedback(
        w_i: np.ndarray,
        f_i: np.ndarray,
        gamma_w: float = 1,
    ) -> np.ndarray:
        chance = 1 / len(f_i)
        f_assoc_feedback = gamma_w * (1 - chance) + chance
        not_f_assoc_feedback = (1 - f_assoc_feedback) / (len(f_i) - 1)
        return f_i * f_assoc_feedback + (1 - f_i) * not_f_assoc_feedback

    def update_weights(
        self,
        w: np.ndarray,
        feedback_mat_active: np.ndarray,
        activations: np.ndarray,
    ) -> np.ndarray:
        w = w * (1 - self.decay)

        if self.w_update_type == "prediction_error":
            for i in range(w.shape[0]):
                w[i] = self.update_weights_RW(
                    w[i],
                    feedback_mat_active[-1],
                    activations[i]
                )
        elif self.w_update_type == "noisy_feedback":
            w[-1] = self.update_weights_noisy_feedback(
                w[-1],
                feedback_mat_active[-1],
                gamma_w,
            )
        elif self.w_update_type == "perfect_instances":
            w[-1] = feedback_mat_active[-1]
        else:
            raise ValueError(f"Unknown w_update_type: {self.w_update_type}")

        if verbose:
            print(feedback_mat_active[-1], "Feedback for current exemplar")

        return w

    def calc_attention_loss_gradient(
        self,
        w: np.ndarray,
        activations: np.ndarray,
        probe: np.ndarray,
        y_true: int,
        alpha_trajectory: np.ndarray | list[Any] | None = None,
    ) -> np.ndarray:
        activations = np.asarray(activations).reshape(-1, 1)
        exemplars = np.asarray(self.exemplars)
        delta = np.asarray(self.delta).reshape(1, -1)
        alpha_trajectory = [] if alpha_trajectory is None else alpha_trajectory

        if(self.decision_rule == "luce"):
            cat_ev = self.calc_evidence(activations, w)
            tot_ev = np.sum(cat_ev)
            den = tot_ev**2
            
            decision_prob = self.calc_evidence_decision(cat_ev)

            if self.partial_encoding and len(alpha_trajectory):
                if self.attention_parameterization == "none":
                    transformed_alpha_history = 10 ** np.asarray(alpha_trajectory)
                elif self.attention_parameterization == "sigmoid":
                    transformed_alpha_history = self.sigmoid(np.asarray(alpha_trajectory))
                else:
                    raise ValueError(
                        f"Unknown attention_parameterization: {self.attention_parameterization}"
                    )

                dervact = -activations * (
                    delta * np.abs(exemplars - probe) * transformed_alpha_history
                )
            else:
                dervact = -activations * (delta * np.abs(exemplars - probe))

            evi_deriv = dervact.T @ w
            sum_evi_deriv = evi_deriv.sum(axis=1, keepdims=True)
            dP = (tot_ev * evi_deriv - sum_evi_deriv * cat_ev[None, :]) / den
        
        if(self.decision_rule == "softmax"):
            cat_ev = self.calc_evidence(activations, w)
            exp_E = np.exp(self.beta*cat_ev)
            decision_prob = exp_E / np.sum(exp_E)
            dervact = -activations * (delta * np.abs(exemplars - probe))
            dE = dervact.T @ w
            derv_exp_E = self.beta * dE * exp_E
            den = (np.sum(exp_E))**2
            dP = (np.sum(exp_E) * derv_exp_E - np.sum(derv_exp_E, axis=0) * exp_E) / den
            
        
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
        w: np.ndarray,
        x: np.ndarray,
        f: np.ndarray,
        D: np.ndarray,
        alpha_trajectory: np.ndarray | list[Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        del D
        alpha_trajectory = [] if alpha_trajectory is None else alpha_trajectory

        if self.attention_update_type == "none":
            dloss = np.zeros(shape=self.alpha.shape)
            dreg = np.zeros(shape=self.alpha.shape)

        elif self.attention_update_type == "loss":
            grad_loss = self.calc_attention_loss_gradient(
                w=w,
                activations=activations,
                probe=x,
                y_true=int(np.argmax(f)),
                alpha_trajectory=alpha_trajectory,
            )

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
                w=w,
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
        elif self.attention_update_dims == "sum_to_constant":
            new_alpha = np.zeros(self.alpha.shape)
            new_alpha[0] = self.alpha[0] - self.lr * (dloss[0] + dreg[0]) + self.lr * (dloss[1] + dreg[1])
            new_alpha[1] = self.alpha[1] - self.lr * (dloss[1] + dreg[1]) + self.lr * (dloss[0] + dreg[0])
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
            activations = cls.calc_activations_exemplars(activations_mat)
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
    "InstanceModel",
]
