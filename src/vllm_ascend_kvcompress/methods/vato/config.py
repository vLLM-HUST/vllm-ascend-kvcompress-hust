# SPDX-License-Identifier: Apache-2.0
"""Calibration-free options for the V@O method."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields

from ...config import ASCEND_BLOCK_SIZE, JsonScalar, require_choice, require_int
from ..base import MethodRuntimeSpec


@dataclass(frozen=True)
class VATOConfig:
    kv_budget: int = 2048
    recompute_window: int = ASCEND_BLOCK_SIZE
    window_size: int = 32
    sink_size: int = 4
    variant: str = "dot"
    kernel_size: int = 1
    score_chunk_size: int = 512
    min_output_tokens_for_compression: int = 0

    @property
    def runtime_spec(self) -> MethodRuntimeSpec:
        return MethodRuntimeSpec(
            requires_private_destination=True,
            compression_threshold_tokens=self.kv_budget + self.recompute_window,
            required_recompute_tokens=self.recompute_window,
            max_physical_num_tokens=self.kv_budget,
            min_output_tokens_for_compression=self.min_output_tokens_for_compression,
        )

    @classmethod
    def from_method_config(cls, options: Mapping[str, JsonScalar]) -> VATOConfig:
        unknown = sorted(set(options) - {field.name for field in fields(cls)})
        if unknown:
            raise ValueError("unknown V@O method options: " + ", ".join(unknown))
        defaults = cls()
        values = {
            field.name: require_int(options, field.name, getattr(defaults, field.name))
            for field in fields(cls)
            if field.name != "variant"
        }
        config = cls(
            **values,
            variant=require_choice(
                options,
                "variant",
                "dot",
                {"dot", "abs", "cosine", "centered", "centered_norm"},
            ),
        )
        for name in ("kv_budget", "recompute_window", "score_chunk_size"):
            value = getattr(config, name)
            if value <= 0 or value % ASCEND_BLOCK_SIZE:
                raise ValueError(
                    f"method option {name!r} must be a positive multiple of 128"
                )
        if config.window_size <= 0:
            raise ValueError("method option 'window_size' must be positive")
        if config.sink_size < 0:
            raise ValueError("method option 'sink_size' must be non-negative")
        if config.window_size + config.sink_size >= config.kv_budget:
            raise ValueError(
                "V@O sink and recent windows must leave room within kv_budget"
            )
        if config.kernel_size <= 0 or config.kernel_size % 2 != 1:
            raise ValueError(
                "method option 'kernel_size' must be a positive odd integer"
            )
        if config.min_output_tokens_for_compression < 0:
            raise ValueError(
                "method option 'min_output_tokens_for_compression' must be non-negative"
            )
        return config


def vato_runtime_spec(options: Mapping[str, JsonScalar]) -> MethodRuntimeSpec:
    return VATOConfig.from_method_config(options).runtime_spec
