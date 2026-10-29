# SPDX-License-Identifier: Apache-2.0
"""验证后端工厂替换、初始化顺序与生命周期分发。"""

from types import SimpleNamespace

import pytest
from runtime_fakes import FakeBackend, RecordingMethod, host_config

from vllm_ascend_kvcompress.host import compat as host_compat
from vllm_ascend_kvcompress.host import plugin
from vllm_ascend_kvcompress.host.plugin import _install_runner_hooks
from vllm_ascend_kvcompress.methods.triattention.backend import TriAttentionBackend
from vllm_ascend_kvcompress.runtime import scheduler as stateful
from vllm_ascend_kvcompress.runtime.provider import (
    RUNNER_PROVIDER_ATTRIBUTE,
    AscendKVCompressionProvider,
)


def _runtime(monkeypatch, failure=None):
    events = []

    # 事件列表记录宿主原方法与 provider 回调的交错顺序。
    # 按名称注入异常，用同一条调用链检查各阶段是否清理并透传原异常。
    def event(name):
        events.append(name)
        if name == failure:
            raise ValueError(name)

    # 只替换依赖真实宿主的绑定与校验，保留通用 provider 的状态初始化。
    # 生命周期回调及方法创建顺序仍走真实包装器。
    class Provider(AscendKVCompressionProvider):
        def validate_host(self, runner):
            event("validate")

        def after_model_load(self, runner):
            event("after_load")

        def bind_model_runner(self, runner, kv_cache_config):
            events.append(("bind", kv_cache_config))
            self.method.bind_model_runner(runner, ())
            self.after_cache_bind()

        def after_cache_bind(self):
            event("cache_bound")

        def before_execute_model(self, output):
            event("before_forward")

        def after_execute_model(self, output, result):
            event("after_forward")
            return "wrapped:" + result

        def after_attention_metadata(self, result, num_tokens, num_reqs):
            events.append(("metadata", num_tokens, num_reqs))
            return "specialized", result

        def compress_scheduled_requests(self, output):
            event("compress")

        def on_step_error(self, output):
            event("cleanup")

    class Backend(FakeBackend):
        def prepare_model_load(self, runner):
            event("prepare")

        def create_provider(self):
            event("provider")
            return Provider(self)

        def create_method(self):
            event("method")
            return super().create_method()

    class Runner:
        vllm_config = host_config()

        def load_model(self):
            event("weights")
            return "loaded"

        def initialize_kv_cache(self, incoming):
            event("allocate")
            self.kv_cache_config = "normalized"
            return incoming

        def _update_states(self, output):
            return output

        def _build_attention_metadata(self, num_tokens, num_reqs):
            return "native"

        def execute_model(self, scheduler_output):
            self.execute_model_state = (scheduler_output,)
            event("forward")
            return "output"

        def sample_tokens(self, grammar):
            event("sample")
            return grammar

    backend = Backend(Runner.vllm_config)
    monkeypatch.setattr(plugin, "create_backend", lambda *args: backend)
    monkeypatch.setattr(host_compat, "prepare_model_load", lambda config: None)
    monkeypatch.setattr(
        host_compat, "install_removed_model_routing_flag", lambda config: None
    )
    monkeypatch.setattr(
        plugin, "_install_runtime_slot_mapping_hooks", lambda *args: None
    )
    _install_runner_hooks(Runner, SimpleNamespace(method="recording", method_config={}))
    return Runner(), backend, events


def test_backend_preparation_precedes_weights_and_single_worker_creation(monkeypatch):
    runner, backend, events = _runtime(monkeypatch)
    assert runner.load_model() == "loaded"
    provider = getattr(runner, RUNNER_PROVIDER_ATTRIBUTE)
    assert provider.method is None
    assert backend.method_creations == 0

    assert runner.initialize_kv_cache("incoming") == "incoming"

    # 权重加载前只做准备；到了 KV 初始化前才创建一次算法实例。
    # 绑定时使用宿主规范化后的 cache 配置，而不是调用者传入的原始计划。
    assert backend.method_creations == 1
    assert isinstance(provider.method, RecordingMethod)
    assert events == [
        "provider",
        "prepare",
        "weights",
        "after_load",
        "method",
        "validate",
        "allocate",
        ("bind", "normalized"),
        "cache_bound",
    ]


def test_execution_and_metadata_use_selected_provider_overrides(monkeypatch):
    runner, _, events = _runtime(monkeypatch)
    runner.load_model()
    runner.initialize_kv_cache("incoming")
    assert runner._build_attention_metadata(12, 3) == ("specialized", "native")
    assert runner.execute_model(SimpleNamespace()) == "wrapped:output"
    assert runner.sample_tokens("tokens") == "tokens"
    assert events[-6:] == [
        ("metadata", 12, 3),
        "before_forward",
        "forward",
        "after_forward",
        "sample",
        "compress",
    ]


@pytest.mark.parametrize(
    "failure", ["before_forward", "forward", "after_forward", "sample", "compress"]
)
def test_execution_errors_cleanup_and_propagate_original_error(monkeypatch, failure):
    runner, _, events = _runtime(monkeypatch, failure)
    runner.load_model()
    runner.initialize_kv_cache("incoming")
    output = SimpleNamespace()
    with pytest.raises(ValueError, match=failure):
        runner.execute_model(output)
        runner.sample_tokens("tokens")
    assert events[-1] == "cleanup"


def test_triattention_prepare_passes_parsed_config_to_calibration(monkeypatch):
    from vllm_ascend_kvcompress.methods.triattention import calibration

    backend = TriAttentionBackend(
        {"stats_path": "missing.pt"}, host_config(), FakeBackend().model_shape
    )
    calls = []
    monkeypatch.setattr(
        calibration,
        "ensure_calibration_for_runner",
        lambda runner, config: calls.append((runner, config)),
    )
    runner = object()
    backend.prepare_model_load(runner)
    assert calls == [(runner, backend.config)]


def test_scheduler_hook_uses_backend_scheduler_factory(monkeypatch):
    events = []

    class State:
        def before_schedule(self):
            events.append("before")

        def after_schedule(self, output):
            events.append(("after", output))

        def after_output(self, output):
            events.append(("ack", output))

    class Backend(FakeBackend):
        def create_scheduler(self, scheduler):
            events.append(("factory", scheduler))
            return State()

    class Scheduler:
        def __init__(self):
            self.vllm_config = host_config()

        def schedule(self):
            events.append("schedule")
            return "batch"

        def update_from_output(self, output):
            events.append(("accepted", output))
            return "accepted"

    class Manager:
        def free(self, request):
            return None

    backend = Backend()
    monkeypatch.setattr(stateful, "create_backend", lambda *args: backend)
    stateful.install_stateful_compression_hooks(
        Scheduler, Manager, SimpleNamespace(method="recording", method_config={})
    )
    scheduler = Scheduler()
    assert scheduler.schedule() == "batch"
    assert scheduler.update_from_output("batch") == "accepted"
    assert backend.method_creations == 0
    assert events == [
        ("factory", scheduler),
        "before",
        "schedule",
        ("after", "batch"),
        ("accepted", "batch"),
        ("ack", "batch"),
    ]
