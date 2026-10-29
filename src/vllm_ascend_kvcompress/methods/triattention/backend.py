# SPDX-License-Identifier: Apache-2.0
"""独立于校准产物与设备的 TriAttention 静态配置。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..backend import CompressionBackend
from ..base import KVCompressionMethod, MethodRuntimeSpec, ModelShape
from .config import TriAttentionConfig


class TriAttentionBackend(CompressionBackend):
    name = "triattention"

    def __init__(
        self,
        options: Mapping[str, Any],
        vllm_config: Any,
        model_shape: ModelShape,
    ) -> None:
        super().__init__(vllm_config, model_shape)
        # 构造阶段只解析参数；stats_path 即使尚不存在，也不在此读取它。
        # 因此 scheduler 可以在自动校准发生前得到准确的触发阈值和目的块容量。
        self.config = TriAttentionConfig.from_method_config(options)
        self.runtime_spec = MethodRuntimeSpec(
            compression_threshold_tokens=self.config.compression_threshold_tokens,
            required_recompute_tokens=self.config.recompute_window,
            max_physical_num_tokens=self.config.kv_budget,
            min_output_tokens_for_compression=(
                self.config.min_output_tokens_for_compression
            ),
        )

    def prepare_model_load(self, runner: Any) -> None:
        from .calibration import ensure_calibration_for_runner

        ensure_calibration_for_runner(runner, self.config)

    # 真正读取统计的动作在 TriAttentionMethod 构造中发生。
    # provider 在校准准备完成后调用这里，避免加载前能力探测造成重复构造。
    def create_method(self) -> KVCompressionMethod:
        from .method import TriAttentionMethod

        return TriAttentionMethod(self.config, self.vllm_config, self.model_shape)


def create_triattention_backend(
    options: Mapping[str, Any], vllm_config: Any, model_shape: ModelShape
) -> CompressionBackend:
    return TriAttentionBackend(options, vllm_config, model_shape)
