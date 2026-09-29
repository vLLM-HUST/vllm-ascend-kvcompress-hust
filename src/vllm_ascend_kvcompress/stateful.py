# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side state for the upstream-aligned plugin adapter."""

from __future__ import annotations

import inspect
import math
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from vllm.logger import logger

from .config import ProviderSelection
from .methods import create_method
from .methods.triattention.config import TriAttentionConfig
from .model import model_shape_from_config
from .provider import (
    _allowed_scheduler_block_sizes,
    _find_full_attention_group,
    _supports_mtp2,
    _validate_method_runtime_spec,
)
from .transaction import PLAN_ATTRIBUTE, CompressionPlan

_STATE_ATTRIBUTE = "_ascend_kvcompress_scheduler_state_v3"
_OFFSETS_ATTRIBUTE = "_ascend_kvcompress_active_offsets_v3"
_DEFERRED_RELEASES_ATTRIBUTE = "_ascend_kvcompress_deferred_releases_v3"
_PENDING_ATTRIBUTE = "_ascend_kvcompress_pending_transactions_v3"
_GROUP_LENGTH_STATE_ATTRIBUTE = "_ascend_kvcompress_group_length_state_v3"
_PREFIX_ADMISSION_STATE_ATTRIBUTE = "_ascend_kvcompress_prefix_admission_state_v3"
_PATCH_MARKER = "_ascend_kvcompress_manager_patch_v3"
_PREFIX_LOOKUP_PATCH_MARKER = "_ascend_kvcompress_prefix_lookup_patch_v3"
_PREFIX_CACHE_HIT_CAP: ContextVar[tuple[int, int] | None] = ContextVar(
    "ascend_kvcompress_prefix_cache_hit_cap", default=None
)
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


@dataclass
class SchedulerPendingCompression:
    plan: CompressionPlan
    reserved_blocks: tuple[Any, ...]
    ready: bool = False
    previous_eligible_step: int | None = None


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
        if selection.method == "triattention":
            # Keep the existing scheduler path independent of worker calibration.
            config = TriAttentionConfig.from_method_config(selection.method_config)
            threshold = config.compression_threshold_tokens
            budget = config.kv_budget
            required_recompute_tokens = config.recompute_window
            min_output_tokens = config.min_output_tokens_for_compression
        else:
            method = create_method(
                selection.method,
                selection.method_config,
                scheduler.vllm_config,
                model_shape_from_config(scheduler.vllm_config.model_config),
            )
            spec = method.runtime_spec
            _validate_method_runtime_spec(method.name, spec)
            threshold = spec.compression_threshold_tokens
            budget = spec.max_physical_num_tokens
            required_recompute_tokens = spec.required_recompute_tokens
            min_output_tokens = spec.min_output_tokens_for_compression
        self.scheduler = scheduler
        self.threshold = threshold
        self.budget = budget
        self.required_recompute_tokens = required_recompute_tokens
        self.min_output_tokens = min_output_tokens
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
        self.group_specific_lengths = False
        self._validate_host()
        coordinator = scheduler.kv_cache_manager.coordinator
        coordinator_hooks = self.group_specific_lengths and all(
            hasattr(type(coordinator), name)
            for name in (
                "get_num_blocks_to_allocate",
                "allocate_new_blocks",
                "remove_skipped_blocks",
                "cache_blocks",
            )
        )
        if coordinator_hooks:
            _install_coordinator_hooks(type(coordinator))
            setattr(coordinator, _GROUP_LENGTH_STATE_ATTRIBUTE, self)
        setattr(scheduler.kv_cache_manager, _OFFSETS_ATTRIBUTE, self.active)
        setattr(
            scheduler.kv_cache_manager,
            _PREFIX_ADMISSION_STATE_ATTRIBUTE,
            self,
        )
        setattr(scheduler.kv_cache_manager, _PENDING_ATTRIBUTE, self.pending)
        setattr(
            scheduler.kv_cache_manager,
            _DEFERRED_RELEASES_ATTRIBUTE,
            self.deferred_releases,
        )
        self._install_attention_cache_hook()
        if not coordinator_hooks:
            self._install_attention_length_hooks()
        logger.info(
            "Ascend KV compression scheduler bound threshold_tokens=%d "
            "target_tokens=%d min_output_tokens=%d scheduler_alignment=%d "
            "attention_logical_block_size=%d required_recompute_tokens=%d",
            self.threshold,
            self.budget,
            self.min_output_tokens,
            self.scheduler_alignment_size,
            self.scheduler_block_size,
            self.required_recompute_tokens,
        )

    def prefix_cache_hit_cap(self, request: Any) -> int | None:
        """Limit APC so an eligible method receives its required query suffix."""
        prompt_tokens = int(request.num_prompt_tokens)
        if (
            prompt_tokens < self.threshold
            or _request_max_tokens(request) < self.min_output_tokens
        ):
            return None
        return max(
            0,
            min(
                prompt_tokens - 1,
                prompt_tokens - self.required_recompute_tokens,
            ),
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
        if config.speculative_config is not None and not _supports_mtp2(config):
            reasons.append(
                "speculative decoding is unqualified; experimental Qwen3.5 "
                "MTP2 requires mamba_cache_mode='align' and explicit opt-in"
            )
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
        if len(scheduler.kv_cache_config.kv_cache_groups) > 1:
            mamba_cache_mode = getattr(config.cache_config, "mamba_cache_mode", "none")
            if mamba_cache_mode not in {"none", "align"}:
                reasons.append(
                    "hybrid compression requires mamba_cache_mode='none' or 'align'"
                )
            self.group_specific_lengths = mamba_cache_mode == "align"
        if reasons:
            raise RuntimeError(
                "Ascend KV compression is incompatible with this scheduler:\n- "
                + "\n- ".join(reasons)
            )

    def before_schedule(self) -> None:
        """Commit only plans whose model output has crossed the worker barrier."""
        scheduler = self.scheduler
        for request_id in tuple(scheduler.finished_req_ids):
            self._discard_pending(request_id)
            self.active.pop(request_id, None)
        if self.async_scheduling:
            return
        self._commit_pending(defer_release_outputs=0)

    def after_model_output(self) -> None:
        """Commit acknowledged async transactions after the worker barrier."""
        if not self.async_scheduling:
            return
        self._drain_deferred_releases()
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
            if not pending.ready:
                continue
            if request_id not in scheduler.requests:
                self._discard_pending(request_id)
                continue
            blocks = cache_manager.req_to_blocks.get(request_id)
            if blocks is None:
                self._discard_pending(request_id)
                continue
            plan = pending.plan
            actual_source = tuple(block.block_id for block in blocks)
            if actual_source != plan.source_block_ids:
                self._discard_pending(request_id)
                raise RuntimeError(
                    f"request {request_id!r} block table changed before "
                    "compression commit"
                )
            source_blocks = tuple(blocks)
            blocks[:] = pending.reserved_blocks
            released = tuple(reversed(source_blocks))
            if defer_release_outputs:
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
                # Private compressed KV has no valid prefix hash for the
                # request's original semantic token sequence.
                cache_manager.num_cached_block[request_id] = 0
            self.active[request_id] = SchedulerActiveCompression(
                plan.semantic_anchor, plan.physical_anchor
            )
            request = scheduler.requests[request_id]
            if pending.previous_eligible_step is not None:
                request.next_decode_eligible_step = pending.previous_eligible_step
            logger.info(
                "KV compression scheduler commit request_id=%s "
                "semantic_tokens=%d physical_tokens=%d source_blocks=%d "
                "destination_blocks=%d released_blocks=%d",
                request_id,
                plan.semantic_anchor,
                plan.physical_anchor,
                len(plan.source_block_ids),
                len(plan.destination_block_ids),
                len(plan.source_block_ids),
            )
            self.pending.pop(request_id, None)

    def after_output(self, output: Any) -> None:
        """Acknowledge writes only after vLLM has accepted worker output."""
        for request_id, plan in getattr(output, PLAN_ATTRIBUTE, {}).items():
            pending = self.pending.get(request_id)
            if pending is not None and pending.plan == plan:
                pending.ready = True
        self.after_model_output()

    def _drain_deferred_releases(self) -> None:
        pool = self.scheduler.kv_cache_manager.block_pool
        for request_id, releases in tuple(self.deferred_releases.items()):
            retained: list[SchedulerDeferredRelease] = []
            for release in releases:
                release.remaining_outputs -= 1
                if release.remaining_outputs <= 0:
                    pool.free_blocks(release.blocks)
                else:
                    retained.append(release)
            if retained:
                self.deferred_releases[request_id] = retained
            else:
                self.deferred_releases.pop(request_id, None)

    def _discard_pending(self, request_id: str) -> None:
        pending = self.pending.pop(request_id, None)
        if pending is None:
            return
        self.scheduler.kv_cache_manager.block_pool.free_blocks(
            reversed(pending.reserved_blocks)
        )
        request = self.scheduler.requests.get(request_id)
        if request is not None and pending.previous_eligible_step is not None:
            request.next_decode_eligible_step = pending.previous_eligible_step

    def _install_attention_cache_hook(self) -> None:
        """Do not register private compressed KV under original token hashes."""
        cache_manager = (
            self.scheduler.kv_cache_manager.coordinator.single_type_managers[
                self.attention_group_index
            ]
        )
        original = getattr(cache_manager, "cache_blocks", None)
        if original is None:
            return

        def cache_blocks(request: Any, *args: Any, **kwargs: Any) -> Any:
            if request.request_id in self.active:
                return None
            return original(request, *args, **kwargs)

        cache_manager.cache_blocks = cache_blocks

    def _install_attention_length_hooks(self) -> None:
        """Translate lengths only for full attention, never for Mamba state.

        KVCacheManager passes the same token counts to all cache groups. After
        compression, that is correct for Mamba/GDN but full attention needs
        the compacted count. Keep Request's semantic count untouched.
        """
        cache_manager = (
            self.scheduler.kv_cache_manager.coordinator.single_type_managers[
                self.attention_group_index
            ]
        )

        def physical(request_id: str, tokens: int) -> int:
            active = self.active.get(request_id)
            return max(0, tokens - active.removed_tokens) if active else tokens

        original_count = getattr(cache_manager, "get_num_blocks_to_allocate", None)
        if original_count is not None:

            def get_num_blocks_to_allocate(
                request_id: str,
                num_tokens: int,
                new_computed_blocks: Any,
                total_computed_tokens: int,
                num_local_computed_tokens: int,
                num_tokens_main_model: int,
                apply_admission_cap: bool = False,
            ) -> Any:
                return original_count(
                    request_id,
                    physical(request_id, num_tokens),
                    new_computed_blocks,
                    physical(request_id, total_computed_tokens),
                    physical(request_id, num_local_computed_tokens),
                    physical(request_id, num_tokens_main_model),
                    apply_admission_cap=apply_admission_cap,
                )

            cache_manager.get_num_blocks_to_allocate = get_num_blocks_to_allocate

        original_allocate = getattr(cache_manager, "allocate_new_blocks", None)
        if original_allocate is not None:

            def allocate_new_blocks(
                request_id: str, num_tokens: int, num_tokens_main_model: int
            ) -> Any:
                return original_allocate(
                    request_id,
                    physical(request_id, num_tokens),
                    physical(request_id, num_tokens_main_model),
                )

            cache_manager.allocate_new_blocks = allocate_new_blocks

        original_skip = getattr(cache_manager, "remove_skipped_blocks", None)
        if original_skip is not None:

            def remove_skipped_blocks(
                request_id: str,
                processed_computed_tokens: int,
                num_prompt_tokens: int | None = None,
            ) -> Any:
                return original_skip(
                    request_id,
                    physical(request_id, processed_computed_tokens),
                    num_prompt_tokens,
                )

            cache_manager.remove_skipped_blocks = remove_skipped_blocks

    def after_schedule(self, output: Any) -> None:
        """Arm the same deterministic transaction the worker performs."""
        reset_ids = set(getattr(output, "preempted_req_ids", ()))
        reset_ids.update(output.finished_req_ids)
        for request_id in reset_ids:
            self._discard_pending(request_id)
            self.active.pop(request_id, None)

        coordinator = self.scheduler.kv_cache_manager.coordinator
        single_manager = coordinator.single_type_managers[self.attention_group_index]
        plans: dict[str, CompressionPlan] = {}
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
            if self.scheduler.vllm_config.speculative_config is not None and active:
                # A decode step's optimistic MTP length can roll back after
                # acceptance. Initial prompt compression has no draft KV.
                continue
            blocks = single_manager.req_to_blocks.get(request_id, ())
            keep = _blocks_for_tokens(self.budget, self.scheduler_block_size)
            if len(blocks) < keep:
                raise RuntimeError(
                    f"request {request_id!r} has too few scheduler blocks"
                )
            pool = self.scheduler.kv_cache_manager.block_pool
            if pool.get_num_free_blocks() < keep:
                logger.warning(
                    "Skipping KV compression for request %s: %d private "
                    "destination blocks unavailable",
                    request_id,
                    keep,
                )
                continue
            destination = tuple(pool.get_new_blocks(keep))
            plan = CompressionPlan(
                semantic_anchor=semantic,
                physical_anchor=self.budget,
                source_block_ids=tuple(block.block_id for block in blocks),
                destination_block_ids=tuple(block.block_id for block in destination),
            )
            previous_step = None
            if self.async_scheduling:
                previous_step = int(request.next_decode_eligible_step)
                request.next_decode_eligible_step = 1 << 62
            self.pending[request_id] = SchedulerPendingCompression(
                plan=plan,
                reserved_blocks=destination,
                previous_eligible_step=previous_step,
            )
            plans[request_id] = plan
        if plans:
            setattr(output, PLAN_ATTRIBUTE, plans)


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
        scheduler: Any, scheduler_output: Any, *args: Any, **kwargs: Any
    ) -> Any:
        result = original_update_from_output(
            scheduler, scheduler_output, *args, **kwargs
        )
        getattr(scheduler, _STATE_ATTRIBUTE).after_output(scheduler_output)
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
    original_free = manager_cls.free
    original_get_computed_blocks = getattr(manager_cls, "get_computed_blocks", None)

    def free(manager: Any, request: Any) -> Any:
        getattr(manager, _OFFSETS_ATTRIBUTE, {}).pop(request.request_id, None)
        pending = getattr(manager, _PENDING_ATTRIBUTE, {}).pop(request.request_id, None)
        if pending is not None:
            manager.block_pool.free_blocks(reversed(pending.reserved_blocks))
            if pending.previous_eligible_step is not None:
                request.next_decode_eligible_step = pending.previous_eligible_step
        releases = getattr(manager, _DEFERRED_RELEASES_ATTRIBUTE, {}).pop(
            request.request_id, ()
        )
        for release in releases:
            manager.block_pool.free_blocks(release.blocks)
        return original_free(manager, request)

    def get_computed_blocks(manager: Any, request: Any) -> Any:
        assert original_get_computed_blocks is not None
        state = getattr(manager, _PREFIX_ADMISSION_STATE_ATTRIBUTE, None)
        cap = None if state is None else state.prefix_cache_hit_cap(request)
        if cap is None:
            return original_get_computed_blocks(manager, request)
        coordinator = manager.coordinator
        _install_prefix_lookup_hook(type(coordinator))
        token = _PREFIX_CACHE_HIT_CAP.set((id(coordinator), cap))
        try:
            return original_get_computed_blocks(manager, request)
        finally:
            _PREFIX_CACHE_HIT_CAP.reset(token)

    setattr(manager_cls, f"{_PATCH_MARKER}_original_free", original_free)
    if original_get_computed_blocks is not None:
        setattr(
            manager_cls,
            f"{_PATCH_MARKER}_original_get_computed_blocks",
            original_get_computed_blocks,
        )
        manager_cls.get_computed_blocks = get_computed_blocks
    manager_cls.free = free
    setattr(manager_cls, _PATCH_MARKER, True)


def _install_prefix_lookup_hook(coordinator_cls: type[Any]) -> None:
    """Cap only the lookup made inside the current request's admission call."""
    if coordinator_cls.__dict__.get(_PREFIX_LOOKUP_PATCH_MARKER, False):
        return
    original = coordinator_cls.find_longest_cache_hit

    def find_longest_cache_hit(
        coordinator: Any,
        block_hashes: Any,
        max_cache_hit_length: int,
    ) -> Any:
        scoped_cap = _PREFIX_CACHE_HIT_CAP.get()
        if scoped_cap is not None and scoped_cap[0] == id(coordinator):
            max_cache_hit_length = min(max_cache_hit_length, scoped_cap[1])
        return original(coordinator, block_hashes, max_cache_hit_length)

    setattr(coordinator_cls, f"{_PREFIX_LOOKUP_PATCH_MARKER}_original", original)
    coordinator_cls.find_longest_cache_hit = find_longest_cache_hit
    setattr(coordinator_cls, _PREFIX_LOOKUP_PATCH_MARKER, True)


def _install_coordinator_hooks(coordinator_cls: type[Any]) -> None:
    """Translate only the compacted full-attention group's token lengths."""
    marker = f"{_PATCH_MARKER}_coordinator"
    if coordinator_cls.__dict__.get(marker, False):
        return
    original_count = coordinator_cls.get_num_blocks_to_allocate
    original_allocate = coordinator_cls.allocate_new_blocks
    original_remove = coordinator_cls.remove_skipped_blocks

    count_signature = inspect.signature(original_count)

    def get_num_blocks_to_allocate(coordinator: Any, *args: Any, **kwargs: Any) -> int:
        bound = count_signature.bind(coordinator, *args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        request_id = values["request_id"]
        num_tokens = values["num_tokens"]
        new_computed_blocks = values["new_computed_blocks"]
        total_computed_tokens = values["total_computed_tokens"]
        num_local_computed_tokens = values.get(
            "num_local_computed_tokens", total_computed_tokens
        )
        num_tokens_main_model = values["num_tokens_main_model"]
        apply_admission_cap = values.get("apply_admission_cap", False)
        state = getattr(coordinator, _GROUP_LENGTH_STATE_ATTRIBUTE, None)
        active = None if state is None else state.active.get(request_id)
        if active is None:
            return original_count(coordinator, *args, **kwargs)
        result = 0
        for index, manager in enumerate(coordinator.single_type_managers):
            if index == state.attention_group_index:
                group_tokens = _physical_tokens(num_tokens, active)
                group_computed = _physical_tokens(total_computed_tokens, active)
                group_local = _physical_tokens(num_local_computed_tokens, active)
                group_main = _physical_tokens(num_tokens_main_model, active)
            else:
                group_tokens = num_tokens
                group_computed = total_computed_tokens
                group_local = num_local_computed_tokens
                group_main = num_tokens_main_model
            manager_values = {
                "request_id": request_id,
                "num_tokens": group_tokens,
                "new_computed_blocks": new_computed_blocks[index],
                "total_computed_tokens": group_computed,
                "num_tokens_main_model": group_main,
                "apply_admission_cap": apply_admission_cap,
            }
            manager_signature = inspect.signature(manager.get_num_blocks_to_allocate)
            if "num_local_computed_tokens" in manager_signature.parameters:
                manager_values["num_local_computed_tokens"] = group_local
            result += manager.get_num_blocks_to_allocate(**manager_values)
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

    setattr(coordinator_cls, f"{marker}_original_count", original_count)
    setattr(coordinator_cls, f"{marker}_original_allocate", original_allocate)
    setattr(coordinator_cls, f"{marker}_original_remove", original_remove)
    coordinator_cls.get_num_blocks_to_allocate = get_num_blocks_to_allocate
    coordinator_cls.allocate_new_blocks = allocate_new_blocks
    coordinator_cls.remove_skipped_blocks = remove_skipped_blocks
    setattr(coordinator_cls, marker, True)


def _physical_tokens(semantic_tokens: int, active: SchedulerActiveCompression) -> int:
    return max(0, int(semantic_tokens) - active.removed_tokens)


def _blocks_for_tokens(num_tokens: int, block_size: int) -> int:
    return (num_tokens + block_size - 1) // block_size
