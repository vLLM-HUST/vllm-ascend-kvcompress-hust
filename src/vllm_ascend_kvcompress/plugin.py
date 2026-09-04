# SPDX-License-Identifier: Apache-2.0
"""Opt-in ``vllm.general_plugins`` registration for current vLLM Ascend."""

from __future__ import annotations

import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec, PathFinder
from types import ModuleType
from typing import Any

from .config import extension_enabled, load_runtime_selection
from .provider import RUNNER_PROVIDER_ATTRIBUTE, AscendKVCompressionProvider

_PATCH_MARKER = "_ascend_kvcompress_patch_v3"
_RUNNER_MODULE = "vllm_ascend.worker.model_runner_v1"
_runner_patch_finder: _RunnerPatchFinder | None = None


def register() -> None:
    """Install hooks only when direct or Extension Manager activation is explicit."""
    if not extension_enabled():
        return
    selection = load_runtime_selection()

    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.worker.block_table import BlockTable

    _install_lazy_runner_hook(selection)
    _install_slot_mapping_hook(BlockTable)

    from .stateful import install_stateful_compression_hooks

    install_stateful_compression_hooks(Scheduler, KVCacheManager, selection)


class _RunnerPatchLoader(Loader):
    def __init__(self, wrapped: Loader, selection: Any) -> None:
        self.wrapped = wrapped
        self.selection = selection

    def create_module(self, spec: ModuleSpec) -> ModuleType | None:
        create = getattr(self.wrapped, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module: ModuleType) -> None:
        self.wrapped.exec_module(module)
        _install_runner_hooks(module.NPUModelRunner, self.selection)


class _RunnerPatchFinder(MetaPathFinder):
    """Wrap only the Ascend runner loader without importing it in API processes."""

    def __init__(self, selection: Any) -> None:
        self.selection = selection

    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        del target
        if fullname != _RUNNER_MODULE:
            return None
        spec = PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        spec.loader = _RunnerPatchLoader(spec.loader, self.selection)
        return spec


def _install_lazy_runner_hook(selection: Any) -> None:
    global _runner_patch_finder
    loaded = sys.modules.get(_RUNNER_MODULE)
    if loaded is not None and hasattr(loaded, "NPUModelRunner"):
        _install_runner_hooks(loaded.NPUModelRunner, selection)
        return
    if _runner_patch_finder is None:
        _runner_patch_finder = _RunnerPatchFinder(selection)
        sys.meta_path.insert(0, _runner_patch_finder)


def _install_runner_hooks(runner_cls: type[Any], selection: Any) -> None:
    if runner_cls.__dict__.get(_PATCH_MARKER, False):
        return
    original_initialize = runner_cls.initialize_kv_cache
    original_update = runner_cls._update_states
    original_metadata = runner_cls._build_attention_metadata
    original_sample = runner_cls.sample_tokens

    def initialize_kv_cache(runner: Any, kv_cache_config: Any) -> Any:
        provider = AscendKVCompressionProvider(runner.vllm_config, selection)
        provider.validate_host(runner)
        setattr(runner, RUNNER_PROVIDER_ATTRIBUTE, provider)
        result = original_initialize(runner, kv_cache_config)
        # Ascend deep-copies and normalizes the incoming plan before storing
        # the cache config that actually owns the allocated tensors.
        provider.bind_model_runner(runner, runner.kv_cache_config)
        return result

    def update_states(runner: Any, scheduler_output: Any) -> Any:
        provider = getattr(runner, RUNNER_PROVIDER_ATTRIBUTE, None)
        if provider is not None:
            provider.before_update_states(scheduler_output)
        result = original_update(runner, scheduler_output)
        if provider is not None:
            provider.after_update_states(scheduler_output)
        return result

    def build_attention_metadata(runner: Any, *args: Any, **kwargs: Any) -> Any:
        provider = getattr(runner, RUNNER_PROVIDER_ATTRIBUTE, None)
        if provider is not None:
            provider.apply_physical_attention_lengths()
        return original_metadata(runner, *args, **kwargs)

    def sample_tokens(runner: Any, grammar_output: Any) -> Any:
        state = runner.execute_model_state
        scheduler_output = state[0] if state is not None else None
        output = original_sample(runner, grammar_output)
        provider = getattr(runner, RUNNER_PROVIDER_ATTRIBUTE, None)
        if provider is not None and scheduler_output is not None:
            provider.compress_scheduled_requests(scheduler_output)
        return output

    setattr(runner_cls, f"{_PATCH_MARKER}_original_initialize", original_initialize)
    setattr(runner_cls, f"{_PATCH_MARKER}_original_update", original_update)
    setattr(runner_cls, f"{_PATCH_MARKER}_original_metadata", original_metadata)
    setattr(runner_cls, f"{_PATCH_MARKER}_original_sample", original_sample)
    runner_cls.initialize_kv_cache = initialize_kv_cache
    runner_cls._update_states = update_states
    runner_cls._build_attention_metadata = build_attention_metadata
    runner_cls.sample_tokens = sample_tokens
    setattr(runner_cls, _PATCH_MARKER, True)


def _install_slot_mapping_hook(block_table_cls: type[Any]) -> None:
    marker = f"{_PATCH_MARKER}_slot_mapping"
    if block_table_cls.__dict__.get(marker, False):
        return
    original = block_table_cls.compute_slot_mapping

    def compute_slot_mapping(
        block_table: Any,
        num_reqs: int,
        query_start_loc: Any,
        positions: Any,
    ) -> None:
        provider = getattr(block_table, RUNNER_PROVIDER_ATTRIBUTE, None)
        if provider is not None:
            positions = provider.physical_positions_for_slot_mapping(positions)
        original(block_table, num_reqs, query_start_loc, positions)

    setattr(block_table_cls, f"{marker}_original", original)
    block_table_cls.compute_slot_mapping = compute_slot_mapping
    setattr(block_table_cls, marker, True)
