# SPDX-License-Identifier: Apache-2.0
"""验证轻量宿主包装层的安装、延迟加载与实际槽位表。"""

from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace

import pytest

import vllm_ascend_kvcompress.host.plugin as plugin
from vllm_ascend_kvcompress.host.plugin import (
    _install_runner_hooks,
    _install_runtime_slot_mapping_hooks,
    _RunnerPatchLoader,
)


class _Runner:
    def load_model(self):
        return "loaded"

    def initialize_kv_cache(self, config):
        return config

    def _update_states(self, output):
        return output

    def _build_attention_metadata(self, *args, **kwargs):
        return args, kwargs

    def execute_model(self, scheduler_output):
        return scheduler_output

    def sample_tokens(self, grammar):
        return grammar


class _WrappedLoader:
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.NPUModelRunner = _Runner


def test_runner_hook_installation_is_idempotent() -> None:
    selection = object()
    _install_runner_hooks(_Runner, selection)
    first_update = _Runner._update_states
    first_sample = _Runner.sample_tokens

    _install_runner_hooks(_Runner, selection)

    assert _Runner._update_states is first_update
    assert _Runner.sample_tokens is first_sample


def test_concrete_slot_mapping_hook_applies_provider_once() -> None:
    class ConcreteBlockTable:
        def compute_slot_mapping(self, num_reqs, query_start_loc, positions):
            self.arguments = num_reqs, query_start_loc, positions

    concrete = ConcreteBlockTable()
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            block_table=SimpleNamespace(block_tables=[concrete])
        )
    )
    _install_runtime_slot_mapping_hooks(runner)
    _install_runtime_slot_mapping_hooks(runner)
    setattr(
        concrete,
        plugin.RUNNER_PROVIDER_ATTRIBUTE,
        SimpleNamespace(
            physical_positions_for_slot_mapping=lambda position: position - 172
        ),
    )
    concrete.compute_slot_mapping(1, [0, 1], 300)
    assert concrete.arguments == (1, [0, 1], 128)
    # 同一块表类型中的未绑定实例继续使用原始语义位置。
    plain = ConcreteBlockTable()
    plain.compute_slot_mapping(1, [0, 1], 300)
    assert plain.arguments == (1, [0, 1], 300)


def test_runtime_slot_mapping_hook_rejects_missing_slot_mapping_method() -> None:
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            block_table=SimpleNamespace(block_tables=[object()])
        )
    )

    with pytest.raises(RuntimeError, match="does not expose compute_slot_mapping"):
        _install_runtime_slot_mapping_hooks(runner)


def test_lazy_loader_patches_runner_after_module_execution(monkeypatch) -> None:
    calls = []
    selection = object()
    monkeypatch.setattr(
        plugin,
        "_install_runner_hooks",
        lambda runner, config: calls.append((runner, config)),
    )
    loader = _RunnerPatchLoader(_WrappedLoader(), selection)
    module = ModuleType("fake_runner")
    spec = ModuleSpec(module.__name__, loader)

    assert loader.create_module(spec) is None
    loader.exec_module(module)

    assert calls == [(_Runner, selection)]
