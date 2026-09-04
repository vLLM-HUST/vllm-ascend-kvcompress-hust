# SPDX-License-Identifier: Apache-2.0

from importlib.machinery import ModuleSpec
from types import ModuleType

import vllm_ascend_kvcompress.plugin as plugin
from vllm_ascend_kvcompress.plugin import (
    _install_runner_hooks,
    _install_slot_mapping_hook,
    _RunnerPatchLoader,
)


class _Runner:
    def initialize_kv_cache(self, config):
        return config

    def _update_states(self, output):
        return output

    def _build_attention_metadata(self, *args, **kwargs):
        return args, kwargs

    def sample_tokens(self, grammar):
        return grammar


class _BlockTable:
    def compute_slot_mapping(self, num_reqs, query_start_loc, positions):
        self.arguments = num_reqs, query_start_loc, positions


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


def test_slot_mapping_hook_is_idempotent() -> None:
    _install_slot_mapping_hook(_BlockTable)
    first = _BlockTable.compute_slot_mapping
    _install_slot_mapping_hook(_BlockTable)
    assert _BlockTable.compute_slot_mapping is first


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


def test_runner_binds_the_normalized_ascend_cache_plan(monkeypatch) -> None:
    events = []

    class Provider:
        def __init__(self, vllm_config, selection):
            events.append(("create", vllm_config, selection))

        def validate_host(self, runner):
            events.append(("validate", runner))

        def bind_model_runner(self, runner, cache_config):
            events.append(("bind", runner, cache_config))

    class Runner:
        vllm_config = object()

        def initialize_kv_cache(self, incoming):
            self.kv_cache_config = "normalized-cache-plan"
            return incoming

        def _update_states(self, output):
            return output

        def _build_attention_metadata(self, *args, **kwargs):
            return args, kwargs

        def sample_tokens(self, grammar):
            return grammar

    monkeypatch.setattr(plugin, "AscendKVCompressionProvider", Provider)
    selection = object()
    _install_runner_hooks(Runner, selection)

    assert Runner().initialize_kv_cache("incoming-plan") == "incoming-plan"
    assert events[-1][0] == "bind"
    assert events[-1][2] == "normalized-cache-plan"
