# SPDX-License-Identifier: Apache-2.0
"""Plugin-local compatibility layer for repeated compression transactions."""

from __future__ import annotations

from typing import Any

from vllm.v1.kv_cache_compression import KVCacheCompressionError

from .config import ASCEND_BLOCK_SIZE

_STATEFUL_MARKER = "_ascend_kvcompress_stateful_v1"


def _validate_repeated_plan(manager: Any, request: Any, plan: Any) -> int:
    config = manager.kv_cache_compression_config
    runtime_spec = manager.kv_cache_compression_runtime_spec
    current_physical = manager._compressed_request_physical_tokens.get(
        request.request_id
    )
    if config is None or runtime_spec is None or current_physical is None:
        raise KVCacheCompressionError("repeated compression state is unavailable")
    if plan.schema_version != config.schema_version or plan.provider != config.provider:
        raise KVCacheCompressionError("repeated compression plan contract mismatch")
    if plan.request_id != request.request_id:
        raise KVCacheCompressionError("repeated compression request_id mismatch")
    if plan.semantic_num_tokens != request.num_computed_tokens:
        raise KVCacheCompressionError(
            f"request {request.request_id!r} repeated plan is stale: "
            f"plan={plan.semantic_num_tokens}, computed={request.num_computed_tokens}"
        )
    if current_physical < runtime_spec.compression_threshold_tokens:
        raise KVCacheCompressionError(
            f"request {request.request_id!r} physical length {current_physical} "
            "is below the repeated compression threshold"
        )
    if not 0 < plan.physical_num_tokens <= runtime_spec.max_physical_num_tokens:
        raise KVCacheCompressionError("repeated compression physical length is invalid")
    layer_lengths = plan.per_layer_physical_num_tokens
    expected_layers = manager.kv_cache_config.kv_cache_groups[0].layer_names
    if (
        not layer_lengths
        or {name for name, _ in layer_lengths} != set(expected_layers)
        or max(length for _, length in layer_lengths) != plan.physical_num_tokens
    ):
        raise KVCacheCompressionError("repeated compression layer lengths are invalid")

    num_blocks = (plan.physical_num_tokens + ASCEND_BLOCK_SIZE - 1) // ASCEND_BLOCK_SIZE
    try:
        if manager.enable_caching:
            destination = manager._compression_destination_reservations.get(
                request.request_id
            )
            if destination is None:
                raise ValueError(
                    f"request {request.request_id!r} has no private repeat destination"
                )
            manager.coordinator.validate_request_block_replacement(
                request.request_id,
                num_blocks,
                plan.expected_block_ids,
                destination,
            )
        else:
            manager.coordinator.validate_request_tail_truncation(
                request.request_id,
                num_blocks,
                plan.expected_block_ids,
            )
    except ValueError as error:
        raise KVCacheCompressionError(str(error)) from error
    return num_blocks


def _install_manager_hooks(manager_cls: type[Any]) -> None:
    if manager_cls.__dict__.get(_STATEFUL_MARKER, False):
        return
    original_validate = manager_cls.validate_compression_plan
    original_apply = manager_cls.apply_compression_plan

    def validate(manager: Any, request: Any, plan: Any) -> int:
        if request.request_id not in manager._compressed_request_physical_tokens:
            return original_validate(manager, request, plan)
        return _validate_repeated_plan(manager, request, plan)

    def apply(manager: Any, request: Any, plan: Any) -> Any:
        if request.request_id not in manager._compressed_request_physical_tokens:
            return original_apply(manager, request, plan)
        num_blocks = _validate_repeated_plan(manager, request, plan)
        try:
            if manager.enable_caching:
                destination = manager._compression_destination_reservations[
                    request.request_id
                ]
                source, target, released, retained = (
                    manager.coordinator.replace_request_blocks(
                        request.request_id,
                        num_blocks,
                        plan.expected_block_ids,
                        destination,
                    )
                )
                del manager._compression_destination_reservations[request.request_id]
            else:
                source = plan.expected_block_ids[0]
                released = manager.coordinator.truncate_request_tail_blocks(
                    request.request_id,
                    num_blocks,
                    plan.expected_block_ids,
                )
                target = source[:num_blocks]
                retained = ()
        except ValueError as error:
            raise KVCacheCompressionError(str(error)) from error

        manager._compressed_request_physical_tokens[request.request_id] = (
            plan.physical_num_tokens
        )
        from vllm.v1.core.kv_cache_manager import KVCacheCompressionCommitResult

        return KVCacheCompressionCommitResult(
            source_block_ids=source,
            destination_block_ids=target,
            released_block_ids=released,
            retained_hashed_source_block_ids=retained,
        )

    manager_cls.validate_compression_plan = validate
    manager_cls.apply_compression_plan = apply
    setattr(manager_cls, _STATEFUL_MARKER, True)


def _install_scheduler_hook(scheduler_cls: type[Any]) -> None:
    marker = f"{_STATEFUL_MARKER}_schedule"
    if scheduler_cls.__dict__.get(marker, False):
        return
    original_schedule = scheduler_cls.schedule

    def schedule(scheduler: Any, *args: Any, **kwargs: Any) -> Any:
        output = original_schedule(scheduler, *args, **kwargs)
        runtime_spec = getattr(
            scheduler, "kv_cache_compression_runtime_spec", None
        )
        manager = getattr(scheduler, "kv_cache_manager", None)
        if runtime_spec is None or manager is None:
            return output

        transactions = dict(output.kv_cache_compression_transaction_ids or {})
        destinations = dict(
            output.kv_cache_compression_destination_block_ids or {}
        )
        for request_id in output.num_scheduled_tokens:
            if (
                request_id in transactions
                or request_id
                in scheduler._inflight_kv_cache_compression_transactions
            ):
                continue
            physical = manager.get_compressed_physical_num_tokens(request_id)
            if physical is None or physical < runtime_spec.compression_threshold_tokens:
                continue
            request = scheduler.requests.get(request_id)
            if request is None:
                continue
            if manager.enable_caching:
                try:
                    destinations[request_id] = manager.reserve_compression_destination(
                        request
                    )
                except (KVCacheCompressionError, ValueError):
                    continue
            scheduler._arm_kv_cache_compression_transaction(
                request_id, transactions
            )

        output.kv_cache_compression_transaction_ids = transactions or None
        output.kv_cache_compression_destination_block_ids = destinations or None
        return output

    setattr(scheduler_cls, f"{marker}_original", original_schedule)
    scheduler_cls.schedule = schedule
    setattr(scheduler_cls, marker, True)


def install_stateful_compression_hooks(
    scheduler_classes: tuple[type[Any], ...],
) -> None:
    """Install repeated-compression support without modifying vLLM sources."""
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    _install_manager_hooks(KVCacheManager)
    for scheduler_cls in scheduler_classes:
        _install_scheduler_hook(scheduler_cls)
