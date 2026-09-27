# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side state for the upstream-aligned plugin adapter."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from vllm.logger import logger

from .config import ProviderSelection
from .methods.triattention.config import TriAttentionConfig
from .provider import (
    SCHEDULER_TRANSACTIONS_ATTRIBUTE,
    _allowed_scheduler_block_sizes,
    _find_full_attention_group,
)

_STATE_ATTRIBUTE = "_ascend_kvcompress_scheduler_state_v3"
_OFFSETS_ATTRIBUTE = "_ascend_kvcompress_active_offsets_v3"
_DEFERRED_RELEASES_ATTRIBUTE = "_ascend_kvcompress_deferred_releases_v3"
_PENDING_ATTRIBUTE = "_ascend_kvcompress_pending_transactions_v3"
_GROUP_LENGTH_STATE_ATTRIBUTE = "_ascend_kvcompress_group_length_state_v3"
_PATCH_MARKER = "_ascend_kvcompress_manager_patch_v3"
_SUPPORTED_SCHEDULER_TYPES = frozenset(
    {
        ("vllm.v1.core.sched.scheduler", "Scheduler"),
        ("vllm.v1.core.sched.async_scheduler", "AsyncScheduler"),
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
    source_block_ids: tuple[int, ...]
    destination_block_ids: tuple[int, ...]
    private_destination_blocks: tuple[Any, ...] = ()


@dataclass(frozen=True)
class SchedulerActiveCompression:
    semantic_anchor: int
    physical_anchor: int

    @property
    def removed_tokens(self) -> int:
        return self.semantic_anchor - self.physical_anchor


@dataclass
class SchedulerDeferredRelease:
    request_id: str
    blocks: tuple[Any, ...]
    remaining_outputs: int


class SchedulerCompressionState:
    """Mirror worker transactions across acknowledged model-output barriers."""

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
        # Scheduler.block_size is the LCM alignment across every hybrid cache
        # group; it can be larger than the full-attention manager page.
        self.scheduler_alignment_size = int(scheduler.block_size)
        self.scheduler_block_size = self.scheduler_alignment_size
        self.pending: dict[str, SchedulerPendingCompression] = {}
        self.active: dict[str, SchedulerActiveCompression] = {}
        self.deferred_releases: dict[str, list[SchedulerDeferredRelease]] = {}
        host_config = scheduler.vllm_config
        self.async_scheduling = bool(
            getattr(host_config.scheduler_config, "async_scheduling", False)
        )
        self.max_concurrent_batches = int(
            getattr(
                host_config,
                "max_concurrent_batches",
                2 if self.async_scheduling else 1,
            )
        )
        if self.max_concurrent_batches < 1:
            raise RuntimeError("max_concurrent_batches must be positive")
        self.attention_group_index = 0
        self.prefix_caching = bool(host_config.cache_config.enable_prefix_caching)
        self.group_specific_lengths = False
        self._validate_host()
        coordinator = scheduler.kv_cache_manager.coordinator
        if self.group_specific_lengths:
            _install_coordinator_hooks(type(coordinator))
            setattr(coordinator, _GROUP_LENGTH_STATE_ATTRIBUTE, self)
            setattr(
                scheduler.kv_cache_manager,
                _GROUP_LENGTH_STATE_ATTRIBUTE,
                self,
            )
        setattr(scheduler.kv_cache_manager, _OFFSETS_ATTRIBUTE, self.active)
        setattr(
            scheduler.kv_cache_manager,
            _DEFERRED_RELEASES_ATTRIBUTE,
            self.deferred_releases,
        )
        setattr(scheduler.kv_cache_manager, _PENDING_ATTRIBUTE, self.pending)
        logger.info(
            "Ascend KV compression scheduler bound threshold_tokens=%d "
            "target_tokens=%d min_output_tokens=%d scheduler_alignment=%d "
            "attention_logical_block_size=%d",
            self.threshold,
            self.budget,
            self.min_output_tokens,
            self.scheduler_alignment_size,
            self.scheduler_block_size,
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
        if config.kv_transfer_config is not None:
            reasons.append("KV transfer is unsupported")
        parallel = config.parallel_config
        for name, attr in (
            ("pipeline parallel", "pipeline_parallel_size"),
            ("data parallel", "data_parallel_size"),
            ("prefill context parallel", "prefill_context_parallel_size"),
            ("decode context parallel", "decode_context_parallel_size"),
        ):
            value = int(getattr(parallel, attr, 1))
            if value != 1:
                reasons.append(f"{name} size must be one, got {value}")
        try:
            self.attention_group_index = _find_full_attention_group(
                scheduler.kv_cache_config.kv_cache_groups
            )
        except RuntimeError as error:
            reasons.append(str(error))
        single_managers = scheduler.kv_cache_manager.coordinator.single_type_managers
        if self.attention_group_index >= len(single_managers):
            reasons.append("full-attention cache manager is unavailable")
        else:
            manager_block_sizes = tuple(
                int(getattr(manager, "block_size", 0)) for manager in single_managers
            )
            if any(block_size <= 0 for block_size in manager_block_sizes):
                reasons.append("KV cache managers must expose positive block sizes")
            else:
                expected_alignment = math.lcm(*manager_block_sizes)
                if self.scheduler_alignment_size != expected_alignment:
                    reasons.append(
                        "scheduler alignment must equal the LCM of cache-group "
                        f"block sizes {manager_block_sizes}; got "
                        f"{self.scheduler_alignment_size}"
                    )
                self.scheduler_block_size = manager_block_sizes[
                    self.attention_group_index
                ]
                allowed_block_sizes = _allowed_scheduler_block_sizes(config)
                if self.scheduler_block_size not in allowed_block_sizes:
                    reasons.append(
                        "full-attention logical block size must be one of "
                        f"{sorted(allowed_block_sizes)}, got "
                        f"{self.scheduler_block_size}"
                    )
                if self.budget % self.scheduler_block_size:
                    reasons.append(
                        "compression target must be divisible by full-attention "
                        f"logical block size {self.scheduler_block_size}"
                    )
        mamba_cache_mode = getattr(config.cache_config, "mamba_cache_mode", "none")
        if len(scheduler.kv_cache_config.kv_cache_groups) > 1:
            if mamba_cache_mode not in {"none", "align"}:
                reasons.append(
                    "hybrid compression supports only mamba_cache_mode "
                    f"'none' or 'align', got {mamba_cache_mode!r}"
                )
            self.group_specific_lengths = mamba_cache_mode == "align"
        if reasons:
            raise RuntimeError(
                "Ascend KV compression is incompatible with this scheduler:\n- "
                + "\n- ".join(reasons)
            )

    def before_schedule(self) -> None:
        """Free the compacted tail only after the prior worker step completed."""
        scheduler = self.scheduler
        for request_id in tuple(scheduler.finished_req_ids):
            self._discard_pending(request_id)
            self.active.pop(request_id, None)
        if self.async_scheduling:
            return
        self._commit_pending(defer_release_outputs=0)

    def after_model_output(self) -> None:
        """Commit async transactions after their model output is acknowledged."""
        if not self.async_scheduling:
            return
        self._drain_deferred_releases()
        scheduler = self.scheduler
        for request_id in tuple(scheduler.finished_req_ids):
            self._discard_pending(request_id)
            self.active.pop(request_id, None)
        self._commit_pending(
            defer_release_outputs=max(0, self.max_concurrent_batches - 1)
        )

    def _commit_pending(self, *, defer_release_outputs: int) -> None:
        scheduler = self.scheduler
        manager = scheduler.kv_cache_manager
        single_managers = manager.coordinator.single_type_managers
        if self.attention_group_index >= len(single_managers):
            raise RuntimeError("full-attention cache manager is unavailable")
        cache_manager = single_managers[self.attention_group_index]
        for request_id, pending in tuple(self.pending.items()):
            if request_id not in scheduler.requests:
                self._discard_pending(request_id)
                continue
            blocks = cache_manager.req_to_blocks.get(request_id)
            if blocks is None:
                self._discard_pending(request_id)
                continue
            source_count = len(pending.source_block_ids)
            actual_source = tuple(block.block_id for block in blocks[:source_count])
            if actual_source != pending.source_block_ids:
                raise RuntimeError(
                    f"request {request_id!r} block table changed before "
                    "compression commit"
                )
            source_block_count = len(blocks)
            if pending.private_destination_blocks:
                released = tuple(reversed(blocks))
                blocks[:] = pending.private_destination_blocks
            else:
                keep = len(pending.destination_block_ids)
                released = tuple(reversed(blocks[keep:]))
                del blocks[keep:]
            if defer_release_outputs and released:
                self.deferred_releases.setdefault(request_id, []).append(
                    SchedulerDeferredRelease(
                        request_id=request_id,
                        blocks=released,
                        remaining_outputs=defer_release_outputs,
                    )
                )
            else:
                manager.block_pool.free_blocks(released)
            if hasattr(cache_manager, "num_cached_block"):
                if pending.private_destination_blocks:
                    cache_manager.num_cached_block[request_id] = 0
                else:
                    cache_manager.num_cached_block[request_id] = min(
                        int(cache_manager.num_cached_block.get(request_id, 0)),
                        len(pending.destination_block_ids),
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
                len(pending.destination_block_ids),
                len(released),
            )
            self.pending.pop(request_id, None)

    def _discard_pending(self, request_id: str) -> None:
        pending = self.pending.pop(request_id, None)
        if pending is not None and pending.private_destination_blocks:
            self.scheduler.kv_cache_manager.block_pool.free_blocks(
                pending.private_destination_blocks
            )

    def _drain_deferred_releases(self) -> None:
        for request_id, releases in tuple(self.deferred_releases.items()):
            retained: list[SchedulerDeferredRelease] = []
            for release in releases:
                remaining = release.remaining_outputs - 1
                if remaining <= 0:
                    self.scheduler.kv_cache_manager.block_pool.free_blocks(
                        release.blocks
                    )
                else:
                    retained.append(
                        SchedulerDeferredRelease(
                            request_id=release.request_id,
                            blocks=release.blocks,
                            remaining_outputs=remaining,
                        )
                    )
            if retained:
                self.deferred_releases[request_id] = retained
            else:
                self.deferred_releases.pop(request_id, None)

    def after_schedule(self, output: Any) -> None:
        """Arm the same deterministic transaction the worker performs."""
        reset_ids = set(getattr(output, "preempted_req_ids", ()))
        reset_ids.update(output.finished_req_ids)
        for request_id in reset_ids:
            self._discard_pending(request_id)
            self.active.pop(request_id, None)

        coordinator = self.scheduler.kv_cache_manager.coordinator
        single_manager = coordinator.single_type_managers[self.attention_group_index]
        transactions: dict[str, tuple[int, ...]] = {}
        speculative_requests = getattr(output, "scheduled_spec_decode_tokens", {})
        for request_id in output.num_scheduled_tokens:
            if speculative_requests.get(request_id):
                continue
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
            keep = _blocks_for_tokens(self.budget, self.scheduler_block_size)
            if len(blocks) < keep:
                raise RuntimeError(
                    f"request {request_id!r} has too few scheduler blocks"
                )
            source_ids = tuple(block.block_id for block in blocks)
            private_blocks: tuple[Any, ...] = ()
            if self.prefix_caching:
                try:
                    allocated = (
                        self.scheduler.kv_cache_manager.block_pool.get_new_blocks(keep)
                    )
                except ValueError:
                    continue
                if len(allocated) != keep:
                    if allocated:
                        self.scheduler.kv_cache_manager.block_pool.free_blocks(
                            allocated
                        )
                    raise RuntimeError("block pool returned a partial allocation")
                if any(
                    getattr(block, "ref_cnt", 1) != 1
                    or getattr(block, "block_hash", None) is not None
                    for block in allocated
                ):
                    self.scheduler.kv_cache_manager.block_pool.free_blocks(allocated)
                    raise RuntimeError(
                        "block pool returned a shared or hashed compression destination"
                    )
                private_blocks = tuple(allocated)
                destination_ids = tuple(block.block_id for block in private_blocks)
            else:
                destination_ids = source_ids[:keep]
            self.pending[request_id] = SchedulerPendingCompression(
                semantic_anchor=semantic,
                physical_anchor=self.budget,
                source_block_ids=source_ids,
                destination_block_ids=destination_ids,
                private_destination_blocks=private_blocks,
            )
            transactions[request_id] = destination_ids
        setattr(output, SCHEDULER_TRANSACTIONS_ATTRIBUTE, transactions)


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
    original_update_from_output = scheduler_cls.update_from_output

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

    def update_from_output(
        scheduler: Any, scheduler_output: Any, model_runner_output: Any
    ) -> Any:
        result = original_update_from_output(
            scheduler, scheduler_output, model_runner_output
        )
        getattr(scheduler, _STATE_ATTRIBUTE).after_model_output()
        return result

    setattr(scheduler_cls, f"{marker}_original_init", original_init)
    setattr(scheduler_cls, f"{marker}_original_schedule", original_schedule)
    setattr(
        scheduler_cls,
        f"{marker}_original_update_from_output",
        original_update_from_output,
    )
    scheduler_cls.__init__ = initialize
    scheduler_cls.schedule = schedule
    scheduler_cls.update_from_output = update_from_output
    setattr(scheduler_cls, marker, True)


def _install_manager_hooks(manager_cls: type[Any]) -> None:
    if manager_cls.__dict__.get(_PATCH_MARKER, False):
        return
    original_allocate = manager_cls.allocate_slots
    original_free = manager_cls.free

    def allocate_slots(manager: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        active = getattr(manager, _OFFSETS_ATTRIBUTE, {}).get(request.request_id)
        if (
            active is None
            or getattr(manager, _GROUP_LENGTH_STATE_ATTRIBUTE, None) is not None
        ):
            return original_allocate(manager, request, *args, **kwargs)
        semantic = int(request.num_computed_tokens)
        request.num_computed_tokens = semantic - active.removed_tokens
        try:
            return original_allocate(manager, request, *args, **kwargs)
        finally:
            request.num_computed_tokens = semantic

    def free(manager: Any, request: Any) -> Any:
        getattr(manager, _OFFSETS_ATTRIBUTE, {}).pop(request.request_id, None)
        pending = getattr(manager, _PENDING_ATTRIBUTE, {}).pop(request.request_id, None)
        if pending is not None and pending.private_destination_blocks:
            manager.block_pool.free_blocks(pending.private_destination_blocks)
        releases = getattr(manager, _DEFERRED_RELEASES_ATTRIBUTE, {}).pop(
            request.request_id, ()
        )
        for release in releases:
            manager.block_pool.free_blocks(release.blocks)
        return original_free(manager, request)

    setattr(manager_cls, f"{_PATCH_MARKER}_original_allocate", original_allocate)
    setattr(manager_cls, f"{_PATCH_MARKER}_original_free", original_free)
    manager_cls.allocate_slots = allocate_slots
    manager_cls.free = free
    setattr(manager_cls, _PATCH_MARKER, True)


def _install_coordinator_hooks(coordinator_cls: type[Any]) -> None:
    """Translate only the compacted full-attention group's token lengths."""
    marker = f"{_PATCH_MARKER}_coordinator"
    if coordinator_cls.__dict__.get(marker, False):
        return
    original_count = coordinator_cls.get_num_blocks_to_allocate
    original_allocate = coordinator_cls.allocate_new_blocks
    original_remove = coordinator_cls.remove_skipped_blocks
    original_cache = coordinator_cls.cache_blocks

    def get_num_blocks_to_allocate(
        coordinator: Any,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Any,
        num_encoder_tokens: int,
        total_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        state = getattr(coordinator, _GROUP_LENGTH_STATE_ATTRIBUTE, None)
        active = None if state is None else state.active.get(request_id)
        if active is None:
            return original_count(
                coordinator,
                request_id,
                num_tokens,
                new_computed_blocks,
                num_encoder_tokens,
                total_computed_tokens,
                num_tokens_main_model,
                apply_admission_cap=apply_admission_cap,
            )
        result = 0
        for index, manager in enumerate(coordinator.single_type_managers):
            if index == state.attention_group_index:
                group_tokens = _physical_tokens(num_tokens, active)
                group_computed = _physical_tokens(total_computed_tokens, active)
                group_main = _physical_tokens(num_tokens_main_model, active)
            else:
                group_tokens = num_tokens
                group_computed = total_computed_tokens
                group_main = num_tokens_main_model
            result += manager.get_num_blocks_to_allocate(
                request_id,
                group_tokens,
                new_computed_blocks[index],
                group_computed,
                group_main,
                apply_admission_cap=apply_admission_cap,
            )
        return result

    def allocate_new_blocks(
        coordinator: Any,
        request_id: str,
        num_tokens: int,
        num_tokens_main_model: int,
        num_encoder_tokens: int = 0,
    ) -> tuple[list[Any], ...]:
        state = getattr(coordinator, _GROUP_LENGTH_STATE_ATTRIBUTE, None)
        active = None if state is None else state.active.get(request_id)
        if active is None:
            return original_allocate(
                coordinator,
                request_id,
                num_tokens,
                num_tokens_main_model,
                num_encoder_tokens,
            )
        allocated = []
        for index, manager in enumerate(coordinator.single_type_managers):
            if index == state.attention_group_index:
                group_tokens = _physical_tokens(num_tokens, active)
                group_main = _physical_tokens(num_tokens_main_model, active)
            else:
                group_tokens = num_tokens
                group_main = num_tokens_main_model
            allocated.append(
                manager.allocate_new_blocks(request_id, group_tokens, group_main)
            )
        return tuple(allocated)

    def remove_skipped_blocks(
        coordinator: Any,
        request_id: str,
        total_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        state = getattr(coordinator, _GROUP_LENGTH_STATE_ATTRIBUTE, None)
        active = None if state is None else state.active.get(request_id)
        if active is None:
            return original_remove(
                coordinator,
                request_id,
                total_computed_tokens,
                num_prompt_tokens,
            )
        for index, manager in enumerate(coordinator.single_type_managers):
            group_tokens = (
                _physical_tokens(total_computed_tokens, active)
                if index == state.attention_group_index
                else total_computed_tokens
            )
            manager.remove_skipped_blocks(request_id, group_tokens, num_prompt_tokens)

    def cache_blocks(coordinator: Any, request: Any, num_computed_tokens: int) -> None:
        state = getattr(coordinator, _GROUP_LENGTH_STATE_ATTRIBUTE, None)
        active = None if state is None else state.active.get(request.request_id)
        if active is None:
            return original_cache(coordinator, request, num_computed_tokens)
        for index, manager in enumerate(coordinator.single_type_managers):
            if index == state.attention_group_index:
                # Compressed blocks do not represent contiguous token hashes and
                # must never be published into the automatic prefix cache.
                continue
            manager.cache_blocks(
                request,
                num_computed_tokens,
                retention_interval=coordinator.retention_interval,
            )

    setattr(coordinator_cls, f"{marker}_original_count", original_count)
    setattr(coordinator_cls, f"{marker}_original_allocate", original_allocate)
    setattr(coordinator_cls, f"{marker}_original_remove", original_remove)
    setattr(coordinator_cls, f"{marker}_original_cache", original_cache)
    coordinator_cls.get_num_blocks_to_allocate = get_num_blocks_to_allocate
    coordinator_cls.allocate_new_blocks = allocate_new_blocks
    coordinator_cls.remove_skipped_blocks = remove_skipped_blocks
    coordinator_cls.cache_blocks = cache_blocks
    setattr(coordinator_cls, marker, True)


def _physical_tokens(semantic_tokens: int, active: SchedulerActiveCompression) -> int:
    return max(0, int(semantic_tokens) - active.removed_tokens)


def _blocks_for_tokens(num_tokens: int, block_size: int) -> int:
    return (num_tokens + block_size - 1) // block_size
