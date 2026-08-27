# SPDX-License-Identifier: Apache-2.0
"""Common model-shape extraction for compression methods."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .methods.base import ModelShape


def model_shape_from_config(model_config: Any) -> ModelShape:
    """Extract the stable method-facing shape from a vLLM model config."""
    text_config = getattr(model_config, "hf_text_config", None)
    if text_config is None:
        text_config = getattr(model_config, "hf_config", None)
    if text_config is None:
        raise ValueError("model has no Hugging Face text configuration")
    num_attention_heads = int(text_config.num_attention_heads)
    num_kv_heads = int(getattr(text_config, "num_key_value_heads", num_attention_heads))
    head_dim = getattr(text_config, "head_dim", None)
    if head_dim is None:
        head_dim = int(text_config.hidden_size) // num_attention_heads
    num_layers = int(
        getattr(
            text_config,
            "num_hidden_layers",
            getattr(text_config, "num_layers", 0),
        )
    )
    if num_layers <= 0:
        raise ValueError("model text configuration has no positive layer count")
    if num_attention_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("model attention head counts must be positive")
    if num_attention_heads % num_kv_heads != 0:
        raise ValueError("model query heads are not divisible by KV heads")
    if int(head_dim) <= 0 or int(head_dim) % 2:
        raise ValueError("model attention head dimension must be positive and even")
    rope_scaling = getattr(text_config, "rope_scaling", None)
    rope_parameters = getattr(text_config, "rope_parameters", None)
    rope_theta = getattr(text_config, "rope_theta", None)
    if rope_theta is None and isinstance(rope_parameters, Mapping):
        rope_theta = rope_parameters.get("rope_theta")
    if rope_theta is None:
        rope_theta = 10000.0

    has_rope_scaling = _has_rope_scaling(rope_scaling)
    has_rope_scaling = has_rope_scaling or _has_rope_scaling(rope_parameters)
    return ModelShape(
        model_type=str(getattr(text_config, "model_type", "")),
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=int(head_dim),
        rope_theta=float(rope_theta),
        has_rope_scaling=has_rope_scaling,
    )


def _has_rope_scaling(parameters: Any) -> bool:
    if not parameters:
        return False
    if not isinstance(parameters, Mapping):
        return True
    rope_type = (
        parameters.get("rope_type", parameters.get("type", "default")) or "default"
    )
    if str(rope_type) != "default":
        return True
    unscaled_keys = {"rope_theta", "rope_type", "type"}
    return bool(set(parameters) - unscaled_keys)
