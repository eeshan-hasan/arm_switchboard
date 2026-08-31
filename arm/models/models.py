from typing import Any
from copy import deepcopy
from abc import abstractmethod

import numpy as np


verbose = False


class Model:
    def __init__(self, X: np.ndarray, f: np.ndarray, params: dict[str, Any] | None):
        """
        X is the coordinates.
        f is the feedback/category labels in integer format, e.g. 0, 1, ...
        params is a dict with the model parameters.
        """
        self.X = np.asarray(X)
        self.f = np.asarray(f, dtype=int)
        self.initialize_feedback()

    @staticmethod
    def create(
        model_type: str,
        X: np.ndarray,
        f: np.ndarray,
        params: dict[str, Any] | None = None,
    ) -> "Model":
        if model_type == "strength":
            from .strength_learning import StrengthModel

            return StrengthModel(X, f, params)
        elif model_type == "instance":
            from .instance_learning import InstanceModel

            return InstanceModel(X, f, params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def fit(self):
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
   
    @abstractmethod
    def set_defaults(self, params: dict[str, Any] | None) -> None:
        pass

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

    def initialize_feedback(self) -> None:
        self.n_categories = len(np.unique(self.f))
        self.feedback_mat = np.zeros((len(self.X), self.n_categories))
        self.feedback_mat[np.arange(len(self.X)), self.f] = 1
    @abstractmethod
    def run_learning_trials(self) -> dict[str, Any]:
        pass

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

    


__all__ = [
    "Model",
]
