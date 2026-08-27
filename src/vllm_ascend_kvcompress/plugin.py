# SPDX-License-Identifier: Apache-2.0
"""Idempotent vLLM general-plugin registration for the Ascend provider."""

from __future__ import annotations

from typing import Any

from .provider import (
    PROVIDER_FACTORY_QUALNAME,
    RUNNER_PROVIDER_ATTRIBUTE,
    WORKER_PROVIDER_ATTRIBUTE,
    create_provider,
    unsupported_report,
)

_PATCH_MARKER = "_ascend_kvcompress_patch_v2"


def _provider_factory_qualname(cls: type[Any]) -> str:
    del cls
    return PROVIDER_FACTORY_QUALNAME


def _validate_kv_cache_compression(worker: Any) -> Any:
    original = getattr(type(worker), f"{_PATCH_MARKER}_validate")
    core_config = worker.vllm_config.kv_cache_compression_config
    if core_config is None:
        return original(worker)
    factory = worker.current_platform.get_kv_cache_compression_provider_factory()
    if factory != PROVIDER_FACTORY_QUALNAME:
        return unsupported_report(
            worker,
            f"Ascend platform provider factory is {factory!r}, expected "
            f"{PROVIDER_FACTORY_QUALNAME!r}",
        )
    try:
        return create_provider(worker).compatibility_report(worker)
    except Exception as error:
        return unsupported_report(
            worker,
            f"provider initialization failed: {type(error).__name__}: {error}",
        )


def _initialize_from_config(worker: Any, kv_cache_config: Any) -> Any:
    original = getattr(type(worker), f"{_PATCH_MARKER}_initialize")
    result = original(worker, kv_cache_config)
    provider = getattr(worker, WORKER_PROVIDER_ATTRIBUTE, None)
    if provider is not None:
        provider.bind_model_runner(worker.model_runner, kv_cache_config)
        setattr(worker.model_runner, RUNNER_PROVIDER_ATTRIBUTE, provider)
    return result


def _update_states(model_runner: Any, scheduler_output: Any) -> Any:
    original = getattr(type(model_runner), f"{_PATCH_MARKER}_update_states")
    result = original(model_runner, scheduler_output)
    provider = getattr(model_runner, RUNNER_PROVIDER_ATTRIBUTE, None)
    if provider is not None:
        provider.consume_block_table_updates(scheduler_output)
    return result


def _prepare_inputs(
    model_runner: Any, scheduler_output: Any, num_scheduled_tokens: Any
) -> Any:
    original = getattr(type(model_runner), f"{_PATCH_MARKER}_prepare_inputs")
    result = original(model_runner, scheduler_output, num_scheduled_tokens)
    provider = getattr(model_runner, RUNNER_PROVIDER_ATTRIBUTE, None)
    if provider is not None:
        provider.apply_physical_decode_state(scheduler_output, num_scheduled_tokens)
    return result


def _sample_tokens(model_runner: Any, grammar_output: Any) -> Any:
    state = model_runner.execute_model_state
    scheduler_output = state.scheduler_output if state is not None else None
    original = getattr(type(model_runner), f"{_PATCH_MARKER}_sample_tokens")
    output = original(model_runner, grammar_output)
    provider = getattr(model_runner, RUNNER_PROVIDER_ATTRIBUTE, None)
    if provider is None or scheduler_output is None:
        return output
    plans = provider.compress_transactions(scheduler_output)
    if not plans:
        return output
    from vllm.v1.outputs import ModelRunnerOutput

    if not isinstance(output, ModelRunnerOutput):
        raise RuntimeError(
            "Ascend KV compression requires synchronous ModelRunnerOutput"
        )
    if output.kv_cache_compression_plans is not None:
        raise RuntimeError("model runner output already contains compression plans")
    output.kv_cache_compression_plans = plans
    return output


def register() -> None:
    """Install the minimal platform, worker, and v1 model-runner hooks."""
    from vllm_ascend.platform import NPUPlatform
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    from vllm_ascend.worker.worker import NPUWorker

    _install_hooks(NPUPlatform, NPUWorker, NPUModelRunner)
    _install_ascend_balance_scheduler_hook()


def _install_ascend_balance_scheduler_hook() -> None:
    from vllm_ascend.core.victim_selector import UnifiedVictimSelector
    from vllm_ascend.patch.platform.patch_balance_schedule import (
        BalanceScheduler,
        _balance_scheduling_enabled,
    )

    _install_balance_scheduler_hook(
        BalanceScheduler,
        UnifiedVictimSelector,
        _balance_scheduling_enabled,
    )


def _install_balance_scheduler_hook(
    scheduler_cls: type[Any],
    victim_selector_cls: type[Any],
    balance_scheduling_enabled: Any,
) -> None:
    marker = f"{_PATCH_MARKER}_balance_scheduler"
    if scheduler_cls.__dict__.get(marker, False):
        return
    original_init = scheduler_cls.__init__
    base_scheduler_cls = scheduler_cls.__mro__[1]

    def _initialize(
        self: Any,
        vllm_config: Any,
        kv_cache_config: Any,
        structured_output_manager: Any,
        block_size: int,
        hash_block_size: int | None = None,
        mm_registry: Any = None,
        include_finished_set: bool = False,
        log_stats: bool = False,
        kv_cache_compression_runtime_spec: Any = None,
    ) -> None:
        if kv_cache_compression_runtime_spec is None:
            original_kwargs = {
                "vllm_config": vllm_config,
                "kv_cache_config": kv_cache_config,
                "structured_output_manager": structured_output_manager,
                "block_size": block_size,
                "hash_block_size": hash_block_size,
                "include_finished_set": include_finished_set,
                "log_stats": log_stats,
            }
            if mm_registry is not None:
                original_kwargs["mm_registry"] = mm_registry
            original_init(self, **original_kwargs)
            return

        if mm_registry is None:
            from vllm.multimodal import MULTIMODAL_REGISTRY

            mm_registry = MULTIMODAL_REGISTRY
        base_scheduler_cls.__init__(
            self,
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=structured_output_manager,
            block_size=block_size,
            hash_block_size=hash_block_size,
            mm_registry=mm_registry,
            include_finished_set=include_finished_set,
            log_stats=log_stats,
            kv_cache_compression_runtime_spec=(kv_cache_compression_runtime_spec),
        )
        self._balance_enabled = balance_scheduling_enabled(vllm_config)
        if self._balance_enabled:
            import torch

            data_parallel_size = vllm_config.parallel_config.data_parallel_size
            self.balance_queue = [
                torch.tensor([0], dtype=torch.int, device="cpu")
                for _ in range(data_parallel_size)
            ]
        self.victim_selector = victim_selector_cls.from_vllm_config(vllm_config)

    setattr(scheduler_cls, f"{marker}_original_init", original_init)
    scheduler_cls.__init__ = _initialize
    setattr(scheduler_cls, marker, True)


def _install_hooks(
    platform_cls: type[Any], worker_cls: type[Any], runner_cls: type[Any]
) -> None:
    """Install hooks on explicit classes so registration can be unit tested."""
    if platform_cls.__dict__.get(_PATCH_MARKER, False):
        return
    previous_factory = platform_cls.get_kv_cache_compression_provider_factory()
    if previous_factory not in {None, PROVIDER_FACTORY_QUALNAME}:
        raise RuntimeError(
            "vLLM-Ascend already declares a different KV compression provider: "
            f"{previous_factory}"
        )

    setattr(
        worker_cls,
        f"{_PATCH_MARKER}_validate",
        worker_cls.validate_kv_cache_compression,
    )
    setattr(
        worker_cls,
        f"{_PATCH_MARKER}_initialize",
        worker_cls.initialize_from_config,
    )
    setattr(
        runner_cls,
        f"{_PATCH_MARKER}_update_states",
        runner_cls._update_states,
    )
    setattr(
        runner_cls,
        f"{_PATCH_MARKER}_prepare_inputs",
        runner_cls._prepare_inputs,
    )
    setattr(
        runner_cls,
        f"{_PATCH_MARKER}_sample_tokens",
        runner_cls.sample_tokens,
    )

    platform_cls.get_kv_cache_compression_provider_factory = classmethod(
        _provider_factory_qualname
    )
    worker_cls.validate_kv_cache_compression = _validate_kv_cache_compression
    worker_cls.initialize_from_config = _initialize_from_config
    runner_cls._update_states = _update_states
    runner_cls._prepare_inputs = _prepare_inputs
    runner_cls.sample_tokens = _sample_tokens
    setattr(platform_cls, _PATCH_MARKER, True)
