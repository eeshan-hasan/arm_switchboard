from __future__ import annotations

from .model_config import ModelConfig


def build_default_variants() -> dict[str, ModelConfig]:
    return {
        "baseline": ModelConfig(
            name="baseline",
            description="Static attention with perfect instance updates.",
            params={},
        ),
        "rw": ModelConfig(
            name="rw",
            description="Rescorla-Wagner style association updates.",
            params={"w_update_type": "RW"},
        ),
        "regularized_attention": ModelConfig(
            name="regularized_attention",
            description="Learns attention with p-regularization.",
            params={"attention_update_type": "p_regularization"},
        ),
        "shared_attention": ModelConfig(
            name="shared_attention",
            description="Uses one shared attention value across dimensions.",
            params={"attention_update_dims": "single"},
        ),
    }


__all__ = ["build_default_variants"]
