# SPDX-License-Identifier: Apache-2.0
"""事务与生命周期测试共用的轻量 CPU 模拟对象。"""

from types import SimpleNamespace

import torch

from vllm_ascend_kvcompress.methods.backend import CompressionBackend
from vllm_ascend_kvcompress.methods.base import (
    CompressionResult,
    KVCompressionMethod,
    LayerCache,
    MethodRuntimeSpec,
    ModelShape,
)
from vllm_ascend_kvcompress.runtime.provider import AscendKVCompressionProvider
from vllm_ascend_kvcompress.runtime.transaction import PLAN_ATTRIBUTE, CompressionPlan


def host_config():
    text = SimpleNamespace(
        model_type="qwen2",
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        hidden_size=4,
        head_dim=4,
        rope_theta=10000.0,
    )
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=128, enable_prefix_caching=False),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        model_config=SimpleNamespace(hf_config=text, hf_text_config=text),
        speculative_config=None,
    )


class RecordingMethod(KVCompressionMethod):
    def __init__(self):
        self.last_request = None

    def compatibility_reasons(self, runner):
        return ()

    def bind_model_runner(self, runner, layer_caches):
        self.layer_caches = layer_caches

    def compress(self, request):
        self.last_request = request
        return CompressionResult(128)


# 只记录算法创建次数，不提前创建算法，便于区分静态配置和 worker 阶段。
# 这里的配置描述小型 CPU 场景，不用于声明真实 Ascend 服务兼容性。
class FakeBackend(CompressionBackend):
    name = "recording"

    def __init__(self, config=None, spec=None):
        super().__init__(
            config or host_config(), ModelShape("qwen2", 1, 1, 1, 4, 10000, False)
        )
        self.runtime_spec = spec or MethodRuntimeSpec(256, 128, 128)
        self.method_creations = 0

    def create_method(self):
        self.method_creations += 1
        return RecordingMethod()


class RecordingBlockTable:
    def __init__(self):
        self.added = None

    def add_row(self, block_ids, row_idx):
        self.added = block_ids, row_idx


def make_provider(backend=None, provider_cls=AscendKVCompressionProvider):
    """按正常流程构造核心状态，再提供模拟的已绑定 CPU 批次。"""
    backend = backend or FakeBackend()
    provider = provider_cls(backend)
    # 事务测试直接构造“已经绑定”的模拟状态，跳过真实宿主和设备初始化。
    # 下面空 cache 只标记绑定层身份，数值搬运由单独的 cache 测试覆盖。
    provider.method = backend.create_method()
    provider.layer_caches = (
        LayerCache("model.layers.0.self_attn", 0, torch.empty(0), torch.empty(0)),
    )
    provider.runner = SimpleNamespace(
        device=torch.device("cpu"),
        requests={},
        input_batch=SimpleNamespace(num_reqs=0, req_ids=[], req_id_to_index={}),
    )
    provider._request_offsets_cpu = torch.zeros(8, dtype=torch.int64)
    provider._request_offsets_device = torch.zeros(8, dtype=torch.int64)
    provider._physical_positions = torch.empty(8, dtype=torch.int64)
    provider._semantic_seq_lens_device = torch.empty(8, dtype=torch.int64)
    provider._semantic_seq_lens_cpu = torch.empty(8, dtype=torch.int64)
    return provider


def output_with_plan(semantic, source, destination, scheduled):
    output = SimpleNamespace(num_scheduled_tokens={"r": scheduled})
    setattr(
        output,
        PLAN_ATTRIBUTE,
        {
            "r": CompressionPlan(semantic, 128, source, destination),
        },
    )
    return output
