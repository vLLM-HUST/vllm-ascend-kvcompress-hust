# SPDX-License-Identifier: Apache-2.0
"""Built-in V@O compression method."""

from collections.abc import Mapping
from typing import Any

from ...config import JsonScalar
from ..base import ModelShape
from .config import VATOConfig, vato_runtime_spec
from .method import VATOMethod


def create_vato_method(
    options: Mapping[str, JsonScalar], vllm_config: Any, model_shape: ModelShape
) -> VATOMethod:
    return VATOMethod(options, vllm_config, model_shape)


__all__ = ["VATOConfig", "VATOMethod", "create_vato_method", "vato_runtime_spec"]
