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
    kv_budget: int = 2048
    recompute_window: int = ASCEND_BLOCK_SIZE
    protected_recent_window: int = ASCEND_BLOCK_SIZE
    score_aggregation: ScoreAggregation = "mean"
    layer_aggregation: LayerAggregation = "mean"
    score_chunk_size: int = 512
    score_layer_stride: int = 4

    @property
    def compression_threshold_tokens(self) -> int:
        return self.kv_budget + self.recompute_window

    @classmethod
    def from_method_config(
        cls, method_config: Mapping[str, JsonScalar]
    ) -> TriAttentionConfig:
        known_options = {
            "stats_path",
            "kv_budget",
            "recompute_window",
            "protected_recent_window",
            "score_aggregation",
            "layer_aggregation",
            "score_chunk_size",
            "score_layer_stride",
        }
        unknown = sorted(set(method_config) - known_options)
        if unknown:
            raise ValueError(
                "unknown TriAttention method options: " + ", ".join(unknown)
            )

        stats_path_raw = method_config.get("stats_path")
        if not isinstance(stats_path_raw, str) or not stats_path_raw.strip():
            raise ValueError("method option 'stats_path' must be a non-empty string")

        config = cls(
            stats_path=Path(stats_path_raw).expanduser(),
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
            score_layer_stride=require_int(
                method_config, "score_layer_stride", 4
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
            raise ValueError(
                "method option 'score_layer_stride' must be positive"
            )
