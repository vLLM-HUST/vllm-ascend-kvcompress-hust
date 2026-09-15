# SPDX-License-Identifier: Apache-2.0
"""Configuration owned by the built-in TriAttention method."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from ...config import ASCEND_BLOCK_SIZE, JsonScalar, require_choice, require_int

ScoreAggregation = Literal["mean", "max"]
LayerAggregation = Literal["mean", "max"]


@dataclass(frozen=True)
class TriAttentionConfig:
    """Validated options owned exclusively by the TriAttention method."""

    stats_path: Path
    auto_calibrate: bool = True
    calibration_input_path: Path | None = None
    calibration_max_length: int = 4096
    calibration_device: str = "auto"
    calibration_attn_implementation: str = "eager"
    calibration_local_files_only: bool = False
    kv_budget: int = 2048
    recompute_window: int = ASCEND_BLOCK_SIZE
    protected_recent_window: int = ASCEND_BLOCK_SIZE
    score_aggregation: ScoreAggregation = "mean"
    layer_aggregation: LayerAggregation = "mean"
    score_chunk_size: int = 512
    score_layer_stride: int = 8
    min_output_tokens_for_compression: int = 0

    @property
    def compression_threshold_tokens(self) -> int:
        return self.kv_budget + self.recompute_window

    @classmethod
    def from_method_config(
        cls, method_config: Mapping[str, JsonScalar]
    ) -> TriAttentionConfig:
        known_options = {
            "stats_path",
            "auto_calibrate",
            "calibration_input_path",
            "calibration_max_length",
            "calibration_device",
            "calibration_attn_implementation",
            "calibration_local_files_only",
            "kv_budget",
            "recompute_window",
            "protected_recent_window",
            "score_aggregation",
            "layer_aggregation",
            "score_chunk_size",
            "score_layer_stride",
            "min_output_tokens_for_compression",
        }
        unknown = sorted(set(method_config) - known_options)
        if unknown:
            raise ValueError(
                "unknown TriAttention method options: " + ", ".join(unknown)
            )

        stats_path_raw = method_config.get("stats_path")
        if not isinstance(stats_path_raw, str) or not stats_path_raw.strip():
            raise ValueError("method option 'stats_path' must be a non-empty string")

        input_path_raw = method_config.get("calibration_input_path")
        if input_path_raw is not None and (
            not isinstance(input_path_raw, str) or not input_path_raw.strip()
        ):
            raise ValueError(
                "method option 'calibration_input_path' must be a non-empty string"
            )
        calibration_device = method_config.get("calibration_device", "auto")
        if not isinstance(calibration_device, str) or not calibration_device.strip():
            raise ValueError(
                "method option 'calibration_device' must be a non-empty string"
            )

        config = cls(
            stats_path=Path(stats_path_raw).expanduser(),
            auto_calibrate=_require_bool(method_config, "auto_calibrate", True),
            calibration_input_path=(
                Path(input_path_raw).expanduser()
                if isinstance(input_path_raw, str)
                else None
            ),
            calibration_max_length=require_int(
                method_config, "calibration_max_length", 4096
            ),
            calibration_device=calibration_device.strip(),
            calibration_attn_implementation=cast(
                str,
                require_choice(
                    method_config,
                    "calibration_attn_implementation",
                    "eager",
                    {"eager", "sdpa", "flash_attention_2"},
                ),
            ),
            calibration_local_files_only=_require_bool(
                method_config, "calibration_local_files_only", False
            ),
            kv_budget=require_int(method_config, "kv_budget", 2048),
            recompute_window=require_int(
                method_config, "recompute_window", ASCEND_BLOCK_SIZE
            ),
            protected_recent_window=require_int(
                method_config, "protected_recent_window", ASCEND_BLOCK_SIZE
            ),
            score_aggregation=cast(
                ScoreAggregation,
                require_choice(
                    method_config,
                    "score_aggregation",
                    "mean",
                    {"mean", "max"},
                ),
            ),
            layer_aggregation=cast(
                LayerAggregation,
                require_choice(
                    method_config,
                    "layer_aggregation",
                    "mean",
                    {"mean", "max"},
                ),
            ),
            score_chunk_size=require_int(method_config, "score_chunk_size", 512),
            score_layer_stride=require_int(method_config, "score_layer_stride", 8),
            min_output_tokens_for_compression=require_int(
                method_config, "min_output_tokens_for_compression", 0
            ),
        )
        config._validate()
        return config

    def _validate(self) -> None:
        aligned_options = {
            "kv_budget": self.kv_budget,
            "recompute_window": self.recompute_window,
            "score_chunk_size": self.score_chunk_size,
        }
        for name, value in aligned_options.items():
            if value <= 0 or value % ASCEND_BLOCK_SIZE != 0:
                raise ValueError(
                    f"method option {name!r} must be a positive multiple of "
                    f"{ASCEND_BLOCK_SIZE}"
                )
        if self.protected_recent_window < 0:
            raise ValueError(
                "method option 'protected_recent_window' must be non-negative"
            )
        if self.protected_recent_window > self.kv_budget:
            raise ValueError(
                "method option 'protected_recent_window' cannot exceed 'kv_budget'"
            )
        if self.score_layer_stride <= 0:
            raise ValueError("method option 'score_layer_stride' must be positive")
        if self.min_output_tokens_for_compression < 0:
            raise ValueError(
                "method option 'min_output_tokens_for_compression' must be non-negative"
            )
        if self.calibration_max_length < 128:
            raise ValueError(
                "method option 'calibration_max_length' must be at least 128"
            )


def _require_bool(options: Mapping[str, JsonScalar], name: str, default: bool) -> bool:
    value = options.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"method option {name!r} must be a boolean")
    return value
