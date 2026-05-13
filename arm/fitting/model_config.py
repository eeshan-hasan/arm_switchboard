from abc import ABC, abstractmethod
from copy import deepcopy

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

    @abstractmethod
    def run_model(self,data):
        """Run Model for a Single Participant"""
        pass

    @abstractmethod
    def get_LL(self,data):
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
                p = self._get_param_def(name)
                self.fixed_params[name] = p.default if p.dim > 1 else p.default[0]

        return self.fixed_params

    def _estimated_param_names(self):
        names = getattr(self, "estimated_params", None)
        if names:
            return names
        return getattr(self, "estimatable_params", [])

    def _resolve_param_name(self, name):
        if name == "initial_alpha_s":
            return "initial_alpha"
        return name

    def _get_param_def(self, name):
        resolved_name = self._resolve_param_name(name)
        if resolved_name not in self.param_defs:
            raise KeyError(f"No parameter definition found for '{name}'")
        return self.param_defs[resolved_name]

    def _get_param_dim(self, name):
        if name == "initial_alpha_s":
            return 1
        return self._get_param_def(name).dim

    def get_x_names(self):
        self.x_names = []

        for name in self._estimated_param_names():
            dim = self._get_param_dim(name)

            if dim > 1:
                self.x_names += [f"{name}_{i}" for i in range(dim)]
            else:
                self.x_names.append(name)

        return self.x_names

    def build_params_x(self, x):
        built_params = deepcopy(self.fixed_params)

        cursor = 0
        for name in self._estimated_param_names():
            p = self._get_param_def(name)
            dim = self._get_param_dim(name)

            if dim == 1:
                raw_x = x[cursor]
            else:
                raw_x = np.asarray(x[cursor:cursor + dim], dtype=float)
            cursor += dim

            built_params[self._resolve_param_name(name)] = p.transform(raw_x)

        return built_params

    def get_bounds(self, return_as_list=False):
        bounds = {}

        for name in self._estimated_param_names():
            p = self._get_param_def(name)
            dim = self._get_param_dim(name)
            bounds[name] = [p.range] * dim

        if return_as_list:
            bounds_list = []
            for name in self._estimated_param_names():
                bounds_list.extend(bounds[name])
            return bounds_list

        return bounds
