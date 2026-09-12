# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side state for the upstream-aligned plugin adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vllm.logger import logger

from .config import ASCEND_BLOCK_SIZE, ProviderSelection
from .methods.triattention.config import TriAttentionConfig

_STATE_ATTRIBUTE = "_ascend_kvcompress_scheduler_state_v3"
_OFFSETS_ATTRIBUTE = "_ascend_kvcompress_active_offsets_v3"
_PATCH_MARKER = "_ascend_kvcompress_manager_patch_v3"
_SUPPORTED_SCHEDULER_TYPES = frozenset(
    {
        ("vllm.v1.core.sched.scheduler", "Scheduler"),
        (
            "vllm_ascend.patch.platform.patch_balance_schedule",
            "BalanceScheduler",
        ),
    }
)


@dataclass(frozen=True)
class SchedulerPendingCompression:
    semantic_anchor: int
    physical_anchor: int
    block_ids: tuple[int, ...]


@dataclass(frozen=True)
class SchedulerActiveCompression:
    semantic_anchor: int
    physical_anchor: int

    @property
    def removed_tokens(self) -> int:
        return self.semantic_anchor - self.physical_anchor


class SchedulerCompressionState:
    """Mirror worker transactions across the synchronous execution barrier."""

    def __init__(self, scheduler: Any, selection: ProviderSelection) -> None:
        if selection.method != "triattention":
            raise ValueError(
                f"scheduler adapter does not support method {selection.method!r}"
            )
        config = TriAttentionConfig.from_method_config(selection.method_config)
        self.scheduler = scheduler
        self.threshold = config.compression_threshold_tokens
        self.budget = config.kv_budget
        self.min_output_tokens = config.min_output_tokens_for_compression
        self.pending: dict[str, SchedulerPendingCompression] = {}
        self.active: dict[str, SchedulerActiveCompression] = {}
        self._validate_host()
        setattr(scheduler.kv_cache_manager, _OFFSETS_ATTRIBUTE, self.active)
        logger.info(
            "Ascend KV compression scheduler bound threshold_tokens=%d "
            "target_tokens=%d min_output_tokens=%d",
            self.threshold,
            self.budget,
            self.min_output_tokens,
        )

    def _validate_host(self) -> None:
        scheduler = self.scheduler
        config = scheduler.vllm_config
        reasons: list[str] = []
        scheduler_type = type(scheduler)
        scheduler_identity = (scheduler_type.__module__, scheduler_type.__name__)
        if scheduler_identity not in _SUPPORTED_SCHEDULER_TYPES:
            reasons.append(
                "the upstream v1 Scheduler or the current Ascend "
                "BalanceScheduler wrapper is required"
            )
        if bool(getattr(scheduler, "_balance_enabled", False)):
            reasons.append("Ascend balance scheduling must be disabled")
        if bool(config.cache_config.enable_prefix_caching):
            reasons.append("prefix caching must be disabled")
        if int(scheduler.block_size) != ASCEND_BLOCK_SIZE:
            reasons.append(
                f"scheduler block size must be {ASCEND_BLOCK_SIZE}, "
                f"got {scheduler.block_size}"
            )
        if config.speculative_config is not None:
            reasons.append("speculative decoding is unsupported")
        if config.kv_transfer_config is not None:
            reasons.append("KV transfer is unsupported")
        if bool(getattr(config.scheduler_config, "async_scheduling", False)):
            reasons.append("asynchronous scheduling is unsupported")
        parallel = config.parallel_config
        for name, attr in (
            ("tensor parallel", "tensor_parallel_size"),
            ("pipeline parallel", "pipeline_parallel_size"),
            ("data parallel", "data_parallel_size"),
            ("prefill context parallel", "prefill_context_parallel_size"),
            ("decode context parallel", "decode_context_parallel_size"),
        ):
            value = int(getattr(parallel, attr, 1))
            if value != 1:
                reasons.append(f"{name} size must be one, got {value}")
        if len(scheduler.kv_cache_config.kv_cache_groups) != 1:
            reasons.append("exactly one full-attention KV group is required")
        if reasons:
            raise RuntimeError(
                "Ascend KV compression is incompatible with this scheduler:\n- "
                + "\n- ".join(reasons)
            )

    def before_schedule(self) -> None:
        """Free the compacted tail only after the prior worker step completed."""
        scheduler = self.scheduler
        for request_id in tuple(scheduler.finished_req_ids):
            self.pending.pop(request_id, None)
            self.active.pop(request_id, None)
        manager = scheduler.kv_cache_manager
        single_managers = manager.coordinator.single_type_managers
        if len(single_managers) != 1:
            raise RuntimeError("compression requires exactly one cache manager")
        cache_manager = single_managers[0]
        for request_id, pending in tuple(self.pending.items()):
            if request_id not in scheduler.requests:
                self.pending.pop(request_id, None)
                continue
            blocks = cache_manager.req_to_blocks.get(request_id)
            if blocks is None:
                self.pending.pop(request_id, None)
                continue
            keep = len(pending.block_ids)
            actual_prefix = tuple(block.block_id for block in blocks[:keep])
            if actual_prefix != pending.block_ids:
                raise RuntimeError(
                    f"request {request_id!r} block table changed before "
                    "compression commit"
                )
            tail = blocks[keep:]
            source_block_count = len(blocks)
            del blocks[keep:]
            manager.block_pool.free_blocks(reversed(tail))
            if hasattr(cache_manager, "num_cached_block"):
                cache_manager.num_cached_block[request_id] = min(
                    int(cache_manager.num_cached_block.get(request_id, 0)), keep
                )
            self.active[request_id] = SchedulerActiveCompression(
                pending.semantic_anchor, pending.physical_anchor
            )
            logger.info(
                "KV compression scheduler commit request_id=%s "
                "semantic_tokens=%d physical_tokens=%d source_blocks=%d "
                "destination_blocks=%d released_blocks=%d",
                request_id,
                pending.semantic_anchor,
                pending.physical_anchor,
                source_block_count,
                keep,
                len(tail),
            )
            self.pending.pop(request_id, None)

    def after_schedule(self, output: Any) -> None:
        """Arm the same deterministic transaction the worker performs."""
        reset_ids = set(getattr(output, "preempted_req_ids", ()))
        reset_ids.update(output.finished_req_ids)
        for request_id in reset_ids:
            self.pending.pop(request_id, None)
            self.active.pop(request_id, None)

        single_manager = (
            self.scheduler.kv_cache_manager.coordinator.single_type_managers[0]
        )
        for request_id in output.num_scheduled_tokens:
            if request_id in self.pending:
                continue
            request = self.scheduler.requests.get(request_id)
            if request is None:
                continue
            if _request_max_tokens(request) < self.min_output_tokens:
                continue
            # The current host advances this value in ``_update_after_schedule``
            # before ``Scheduler.schedule`` returns. The worker receives the
            # pre-step value and separately adds its scheduled-token count.
            semantic = int(request.num_computed_tokens)
            active = self.active.get(request_id)
            if active is None:
                physical = semantic
                if semantic != int(request.num_prompt_tokens):
                    continue
            else:
                physical = active.physical_anchor + semantic - active.semantic_anchor
            if physical < self.threshold:
                continue
            blocks = single_manager.req_to_blocks.get(request_id, ())
            keep = _blocks_for_tokens(self.budget)
            if len(blocks) < keep:
                raise RuntimeError(
                    f"request {request_id!r} has too few scheduler blocks"
                )
            self.pending[request_id] = SchedulerPendingCompression(
                semantic_anchor=semantic,
                physical_anchor=self.budget,
                block_ids=tuple(block.block_id for block in blocks[:keep]),
            )


def _request_max_tokens(request: Any) -> int:
    value = getattr(request, "max_tokens", None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(
            "current vLLM Request.max_tokens must be a non-negative integer"
        )
    return value


def install_stateful_compression_hooks(
    scheduler_cls: type[Any],
    manager_cls: type[Any],
    selection: ProviderSelection,
) -> None:
    """Install idempotent scheduler and allocation wrappers."""
    _install_manager_hooks(manager_cls)
    marker = f"{_PATCH_MARKER}_scheduler"
    if scheduler_cls.__dict__.get(marker, False):
        return
    original_init = scheduler_cls.__init__
    original_schedule = scheduler_cls.schedule

    def initialize(scheduler: Any, *args: Any, **kwargs: Any) -> None:
        original_init(scheduler, *args, **kwargs)
        setattr(
            scheduler, _STATE_ATTRIBUTE, SchedulerCompressionState(scheduler, selection)
        )

    def schedule(scheduler: Any, *args: Any, **kwargs: Any) -> Any:
        state = getattr(scheduler, _STATE_ATTRIBUTE)
        state.before_schedule()
        output = original_schedule(scheduler, *args, **kwargs)
        state.after_schedule(output)
        return output

    setattr(scheduler_cls, f"{marker}_original_init", original_init)
    setattr(scheduler_cls, f"{marker}_original_schedule", original_schedule)
    scheduler_cls.__init__ = initialize
    scheduler_cls.schedule = schedule
    setattr(scheduler_cls, marker, True)


def _install_manager_hooks(manager_cls: type[Any]) -> None:
    if manager_cls.__dict__.get(_PATCH_MARKER, False):
        return
    original_allocate = manager_cls.allocate_slots
    original_free = manager_cls.free

    def allocate_slots(manager: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        active = getattr(manager, _OFFSETS_ATTRIBUTE, {}).get(request.request_id)
        if active is None:
            return original_allocate(manager, request, *args, **kwargs)
        semantic = int(request.num_computed_tokens)
        request.num_computed_tokens = semantic - active.removed_tokens
        try:
            return original_allocate(manager, request, *args, **kwargs)
        finally:
            request.num_computed_tokens = semantic

    def free(manager: Any, request: Any) -> Any:
        getattr(manager, _OFFSETS_ATTRIBUTE, {}).pop(request.request_id, None)
        return original_free(manager, request)

    setattr(manager_cls, f"{_PATCH_MARKER}_original_allocate", original_allocate)
    setattr(manager_cls, f"{_PATCH_MARKER}_original_free", original_free)
    manager_cls.allocate_slots = allocate_slots
    manager_cls.free = free
    setattr(manager_cls, _PATCH_MARKER, True)


def _blocks_for_tokens(num_tokens: int) -> int:
    return (num_tokens + ASCEND_BLOCK_SIZE - 1) // ASCEND_BLOCK_SIZE
