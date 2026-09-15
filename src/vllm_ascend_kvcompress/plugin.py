# SPDX-License-Identifier: Apache-2.0
"""Opt-in ``vllm.general_plugins`` registration for current vLLM Ascend."""

from __future__ import annotations

import sys
from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec, PathFinder
from types import ModuleType
from typing import Any

from vllm.logger import logger

from .config import extension_enabled, load_runtime_selection
from .provider import RUNNER_PROVIDER_ATTRIBUTE, AscendKVCompressionProvider

_PATCH_MARKER = "_ascend_kvcompress_patch_v3"
_RUNNER_MODULE = "vllm_ascend.worker.model_runner_v1"
_runner_patch_finder: _RunnerPatchFinder | None = None
_TRITON_GLUON_MODULES = (
    "triton.experimental",
    "triton.experimental.gluon",
    "triton.experimental.gluon.language",
    "triton.experimental.gluon.nvidia",
)


def register() -> None:
    """Install hooks only when direct or Extension Manager activation is explicit."""
    if not extension_enabled():
        return
    selection = load_runtime_selection()
    logger.info(
        "Ascend KV compression plugin activated provider=%s method=%s",
        selection.provider_name,
        selection.method,
    )
    _prepare_current_triton_runtime()

    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.scheduler import Scheduler

    _install_lazy_runner_hook(selection)

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
    # Current vLLM folded the legacy ``xdrope_section`` check into
    # ``uses_mrope`` while the aligned Ascend runner still reads this old
    # attribute in three paths.  Supply the neutral legacy value only when the
    # host no longer defines it; models using xdrope are already represented by
    # ``uses_mrope`` on this validated vLLM line.
    if not hasattr(runner_cls, "uses_xdrope_dim"):
        runner_cls.uses_xdrope_dim = 0
    original_initialize = runner_cls.initialize_kv_cache
    original_load_model = runner_cls.load_model
    original_update = runner_cls._update_states
    original_metadata = runner_cls._build_attention_metadata
    original_sample = runner_cls.sample_tokens

    def load_model(runner: Any) -> Any:
        from .calibration import ensure_calibration_for_runner

        generated = ensure_calibration_for_runner(runner, selection)
        if generated:
            logger.info(
                "Generated model-matched TriAttention calibration artifact "
                "before loading serving weights"
            )
        return original_load_model(runner)

    def initialize_kv_cache(runner: Any, kv_cache_config: Any) -> Any:
        provider = AscendKVCompressionProvider(runner.vllm_config, selection)
        provider.validate_host(runner)
        setattr(runner, RUNNER_PROVIDER_ATTRIBUTE, provider)
        result = original_initialize(runner, kv_cache_config)
        # Ascend deep-copies and normalizes the incoming plan before storing
        # the cache config that actually owns the allocated tensors.
        _install_runtime_slot_mapping_hooks(runner)
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
    setattr(runner_cls, f"{_PATCH_MARKER}_original_load_model", original_load_model)
    setattr(runner_cls, f"{_PATCH_MARKER}_original_update", original_update)
    setattr(runner_cls, f"{_PATCH_MARKER}_original_metadata", original_metadata)
    setattr(runner_cls, f"{_PATCH_MARKER}_original_sample", original_sample)
    runner_cls.load_model = load_model
    runner_cls.initialize_kv_cache = initialize_kv_cache
    runner_cls._update_states = update_states
    runner_cls._build_attention_metadata = build_attention_metadata
    runner_cls.sample_tokens = sample_tokens
    setattr(runner_cls, _PATCH_MARKER, True)


def _prepare_current_triton_runtime() -> None:
    """Preload a module required by the current Ascend JIT specializer.

    Triton Ascend 3.6 resolves Gluon descriptor types from its native argument
    specializer.  Its first RoPE launch fails unless the NVIDIA descriptor
    namespace has already been materialized, even though the package is present
    in the wheel.  Keep the workaround local to enabled plugin processes and
    fail with a useful installation error if the expected module is absent.
    """
    # vLLM-Ascend's legacy compatibility path creates parent modules with an
    # empty ``__path__``.  Triton Ascend is distributed as ``triton-ascend``,
    # so a host check for distribution ``triton`` can select that path even on
    # 3.6.  Remove only those unmistakable empty stubs before importing the
    # real packages from the wheel.
    legacy_stub_present = any(
        getattr(sys.modules.get(name), "__path__", None) == []
        for name in _TRITON_GLUON_MODULES[:-1]
    )
    if legacy_stub_present:
        for name in reversed(_TRITON_GLUON_MODULES):
            sys.modules.pop(name, None)

    try:
        for name in _TRITON_GLUON_MODULES:
            import_module(name)
    except ImportError as error:
        raise RuntimeError(
            f"the validated Triton Ascend runtime is incomplete: cannot import {name}"
        ) from error


def _install_runtime_slot_mapping_hooks(runner: Any) -> None:
    """Patch the concrete tables allocated by the current Ascend runner.

    vLLM-Ascend owns a device-specific BlockTable implementation.  Patching
    vLLM's generic table is therefore neither sufficient nor a stable way to
    locate the active slot-mapping seam.  Resolve it from the initialized
    runner and fail closed if the expected single-group structure changes.
    """
    group_table = getattr(getattr(runner, "input_batch", None), "block_table", None)
    block_tables = getattr(group_table, "block_tables", None)
    if not isinstance(block_tables, list) or len(block_tables) != 1:
        raise RuntimeError(
            "Ascend KV compression requires exactly one concrete block table"
        )
    block_table = block_tables[0]
    if not callable(getattr(block_table, "compute_slot_mapping", None)):
        raise RuntimeError("Ascend block table does not expose compute_slot_mapping")
    _install_slot_mapping_hook(type(block_table))


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
        return original(block_table, num_reqs, query_start_loc, positions)

    setattr(block_table_cls, f"{marker}_original", original)
    block_table_cls.compute_slot_mapping = compute_slot_mapping
    setattr(block_table_cls, marker, True)
