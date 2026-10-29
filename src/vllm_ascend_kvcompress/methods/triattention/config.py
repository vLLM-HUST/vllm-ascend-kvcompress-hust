# SPDX-License-Identifier: Apache-2.0
"""内置 TriAttention 方法自行持有的配置。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from ...core.constants import KERNEL_CACHE_BLOCK_SIZE
from ...core.validation import require_bool, require_choice, require_int

ScoreAggregation = Literal["mean", "max"]
LayerAggregation = Literal["mean", "max"]
PositionPolicy = Literal["global", "v3"]


@dataclass(frozen=True)
class TriAttentionConfig:
    stats_path: Path  # 模型匹配的校准统计文件路径。
    auto_calibrate: bool = True  # 统计文件缺失时自动生成。
    calibration_input_path: Path | None = None  # 校准文本路径；None 使用内置文本。
    calibration_max_length: int = 4096  # 校准输入的最大 token 数。
    calibration_device: str = "auto"  # 校准设备；auto 自动选择。
    # 临时校准模型使用的注意力实现。
    calibration_attn_implementation: str = "eager"
    calibration_local_files_only: bool = False  # 校准加载仅使用本地模型文件。

    kv_budget: int = 2048  # 压缩后保留的 KV token 总数，包含保护区。
    # 重算窗口决定触发间隔，同时为 APC 保留前向后缀；不表示压缩后重跑历史。
    recompute_window: int = KERNEL_CACHE_BLOCK_SIZE
    position_policy: PositionPolicy = "global"  # 位置选择策略：global 或分段 v3。
    protected_prefix_window: int = 0  # 强制保留的起始 token 数。
    # 强制保留的末尾 token 数。
    protected_recent_window: int = KERNEL_CACHE_BLOCK_SIZE
    position_segments: int = 8  # v3 策略的中间区间分段数。

    score_aggregation: ScoreAggregation = "mean"  # 未来 query 采样位置间的聚合方式。
    layer_aggregation: LayerAggregation = "mean"  # 不同评分层之间的聚合方式。
    score_chunk_size: int = 512  # 分块评分时每块处理的 token 数。
    score_layer_stride: int = 8  # 评分层采样步长；所有缓存层仍搬运所选 KV。
    min_output_tokens_for_compression: int = 0  # 请求最大输出 token 数门槛；0 不限制。

    @property
    def compression_threshold_tokens(self) -> int:
        return self.kv_budget + self.recompute_window

    @classmethod
    def from_method_config(cls, method_config: Mapping[str, Any]) -> TriAttentionConfig:
        # 只读取本方法拥有的字段，额外字段不参与解析或校验。
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
            auto_calibrate=require_bool(method_config, "auto_calibrate", True),
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
            calibration_local_files_only=require_bool(
                method_config, "calibration_local_files_only", False
            ),
            kv_budget=require_int(method_config, "kv_budget", 2048),
            recompute_window=require_int(
                method_config, "recompute_window", KERNEL_CACHE_BLOCK_SIZE
            ),
            position_policy=cast(
                PositionPolicy,
                require_choice(
                    method_config,
                    "position_policy",
                    "global",
                    {"global", "v3"},
                ),
            ),
            protected_prefix_window=require_int(
                method_config, "protected_prefix_window", 0
            ),
            protected_recent_window=require_int(
                method_config, "protected_recent_window", KERNEL_CACHE_BLOCK_SIZE
            ),
            position_segments=require_int(method_config, "position_segments", 8),
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
        # 这里检查内核的 128-token 对齐；若 hybrid manager 页更大，
        # 预算还需通过 provider/stateful 的逻辑页对齐校验，
        # 方法配置单独通过并不保证启动成功。
        aligned_options = {
            "kv_budget": self.kv_budget,
            "recompute_window": self.recompute_window,
            "score_chunk_size": self.score_chunk_size,
        }
        for name, value in aligned_options.items():
            if value <= 0 or value % KERNEL_CACHE_BLOCK_SIZE != 0:
                raise ValueError(
                    f"method option {name!r} must be a positive multiple of "
                    f"{KERNEL_CACHE_BLOCK_SIZE}"
                )
        for name, value in (
            ("protected_prefix_window", self.protected_prefix_window),
            ("protected_recent_window", self.protected_recent_window),
        ):
            if value < 0:
                raise ValueError(f"method option {name!r} must be non-negative")
        if self.protected_prefix_window + self.protected_recent_window > self.kv_budget:
            raise ValueError(
                "protected prefix and recent windows cannot exceed 'kv_budget'"
            )
        if self.position_segments <= 0:
            raise ValueError("method option 'position_segments' must be positive")
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
