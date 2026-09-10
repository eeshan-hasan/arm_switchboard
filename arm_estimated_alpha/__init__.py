"""Infer latent attention trajectories from behavior using a StrengthModel."""

from .basis import attention_basis
from .model import EstimatedAlphaModel
from .fitting import FitResult, fit_subject
from .config import EstimatedAlphaConfig, EstimatedAlphaModelConfig
from .rule_centered import RuleCenteredModel, RuleCenteredConfig, RuleCenteredFitResult, fit_rule_centered_subject

__all__ = [
    "attention_basis", "EstimatedAlphaModel", "FitResult", "fit_subject",
    "EstimatedAlphaConfig", "EstimatedAlphaModelConfig",
    "RuleCenteredModel", "RuleCenteredConfig", "RuleCenteredFitResult", "fit_rule_centered_subject",
]
