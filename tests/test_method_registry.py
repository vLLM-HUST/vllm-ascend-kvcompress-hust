# SPDX-License-Identifier: Apache-2.0
"""验证注册契约及静态配置与 worker 初始化之间的边界。"""

import pytest
from runtime_fakes import FakeBackend, host_config

from vllm_ascend_kvcompress.methods.base import ModelShape
from vllm_ascend_kvcompress.methods.registry import MethodRegistry
from vllm_ascend_kvcompress.methods.triattention.backend import TriAttentionBackend


def test_registry_normalizes_names_and_rejects_conflicts():
    registry = MethodRegistry()
    registry.register(" Recording ", lambda *args: FakeBackend())
    # 正常创建同时保护名称归一化；实际 worker 创建顺序由生命周期测试验证。
    backend = registry.create("RECORDING", {}, host_config(), FakeBackend().model_shape)
    assert isinstance(backend, FakeBackend)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("recording", lambda *args: FakeBackend())
    with pytest.raises(ValueError, match="unknown compression method"):
        registry.create("missing", {}, host_config(), FakeBackend().model_shape)


def test_triattention_backend_parses_budget_without_reading_artifact(tmp_path):
    backend = TriAttentionBackend(
        {"stats_path": str(tmp_path / "not-generated.pt"), "kv_budget": 2048},
        host_config(),
        ModelShape("qwen2", 1, 1, 1, 4, 10000, False),
    )
    assert backend.runtime_spec.max_physical_num_tokens == 2048
    assert backend.runtime_spec.compression_threshold_tokens == 2176
    assert not backend.config.stats_path.exists()


def test_registry_rejects_a_factory_returning_a_worker_method():
    registry = MethodRegistry()
    registry.register("v1", lambda *args: FakeBackend().create_method())
    with pytest.raises(TypeError, match="expected CompressionBackend"):
        registry.create("v1", {}, host_config(), FakeBackend().model_shape)
