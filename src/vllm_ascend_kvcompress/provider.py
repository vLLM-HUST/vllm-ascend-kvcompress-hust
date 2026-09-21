# SPDX-License-Identifier: Apache-2.0
"""Worker-side adapter for the upstream-aligned vLLM/Ascend runtime.

The current hosts intentionally expose only the standard ``vllm.general_plugins``
entry point. This module owns its state instead of importing fork-only
KV-compression types. All integration with non-public runtime symbols is checked
at startup and kept in :mod:`vllm_ascend_kvcompress.plugin`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from vllm.logger import logger

from .config import ASCEND_BLOCK_SIZE, ProviderSelection
from .methods import create_method
from .methods.base import CompressionRequest, LayerCache, MethodRuntimeSpec, ModelShape
from .methods.triattention.kernels import shift_positions
from .model import model_shape_from_config

RUNNER_PROVIDER_ATTRIBUTE = "_ascend_kvcompress_provider_v3"


@dataclass(frozen=True)
class PendingCompression:
    semantic_anchor: int
    physical_anchor: int
    block_ids: tuple[int, ...]


@dataclass(frozen=True)
class ActiveCompression:
    semantic_anchor: int
    physical_anchor: int

    @property
    def removed_tokens(self) -> int:
        return self.semantic_anchor - self.physical_anchor


class AscendKVCompressionProvider:
    """Own one model runner's compression state and compacted length mapping."""

    def __init__(self, vllm_config: Any, selection: ProviderSelection) -> None:
        self.vllm_config = vllm_config
        self.selection = selection
        self.model_shape: ModelShape = model_shape_from_config(vllm_config.model_config)
        self.tensor_parallel_size = int(
            getattr(vllm_config.parallel_config, "tensor_parallel_size", 1)
        )
        self.attention_group_index = 0
        self.scheduler_block_size = int(vllm_config.cache_config.block_size)
        self.cache_block_size = ASCEND_BLOCK_SIZE
        self.cache_blocks_per_scheduler_block = 1
        self.method = create_method(
            selection.method,
            selection.method_config,
            vllm_config,
            self.model_shape,
        )
        self.runtime_spec: MethodRuntimeSpec = self.method.runtime_spec
        _validate_method_runtime_spec(self.method.name, self.runtime_spec)
        self.runner: Any | None = None
        self.layer_caches: tuple[LayerCache, ...] = ()
        self.pending: dict[str, PendingCompression] = {}
        self.active: dict[str, ActiveCompression] = {}
        self._request_offsets_cpu: torch.Tensor | None = None
        self._request_offsets_device: torch.Tensor | None = None
        self._physical_positions: torch.Tensor | None = None
        self._offset_row_snapshot: tuple[int, ...] = ()
        self._has_active_rows = False
        self._physical_lengths_applied = False

    def validate_host(self, runner: Any) -> None:
        """Fail closed before binding any cache memory."""
        reasons = list(_common_compatibility_reasons(self.vllm_config, runner))
        reasons.extend(
            f"method {self.method.name}: {reason}"
            for reason in self.method.compatibility_reasons(runner)
        )
        if reasons:
            raise RuntimeError(
                "Ascend KV compression is incompatible with this launch:\n- "
                + "\n- ".join(reasons)
            )

    def bind_model_runner(self, runner: Any, kv_cache_config: Any) -> None:
        """Validate and bind dense K/V views after the host allocates its cache."""
        self.validate_host(runner)
        from vllm.model_executor.models.utils import extract_layer_index

        groups = kv_cache_config.kv_cache_groups
        self.attention_group_index = _find_full_attention_group(groups)
        group = groups[self.attention_group_index]
        self.scheduler_block_size = int(group.kv_cache_spec.block_size)
        allowed_block_sizes = _allowed_scheduler_block_sizes(self.vllm_config)
        if self.scheduler_block_size not in allowed_block_sizes:
            raise RuntimeError(
                "Ascend KV compression does not support logical KV block size "
                f"{self.scheduler_block_size}; allowed sizes are "
                f"{sorted(allowed_block_sizes)}"
            )
        if self.runtime_spec.max_physical_num_tokens % self.scheduler_block_size:
            raise RuntimeError(
                "compression target must be divisible by the logical KV block "
                f"size {self.scheduler_block_size}"
            )

        block_tables = runner.input_batch.block_table.block_tables
        if len(block_tables) != len(groups):
            raise RuntimeError("worker block-table groups do not match KV groups")
        attention_block_table = block_tables[self.attention_group_index]
        physical_block_size = int(
            getattr(
                attention_block_table,
                "physical_block_size",
                self.scheduler_block_size,
            )
        )
        if physical_block_size != self.scheduler_block_size:
            raise RuntimeError(
                "worker full-attention block table disagrees with its KV cache "
                "group block size"
            )
        self.cache_block_size = int(
            getattr(attention_block_table, "block_size", self.scheduler_block_size)
        )
        if (
            self.cache_block_size != ASCEND_BLOCK_SIZE
            or self.scheduler_block_size % self.cache_block_size
        ):
            raise RuntimeError(
                "Ascend KV compression requires 128-token kernel cache blocks "
                "that evenly divide the logical KV block size"
            )
        self.cache_blocks_per_scheduler_block = (
            self.scheduler_block_size // self.cache_block_size
        )

        layer_caches: list[LayerCache] = []
        context = runner.compilation_config.static_forward_context
        for layer_name in group.layer_names:
            layer = context.get(layer_name)
            if layer is None:
                raise RuntimeError(f"attention layer {layer_name!r} is not bound")
            k_cache, v_cache = _unpack_layer_cache(layer_name, layer)
            local_kv_heads = self.model_shape.num_kv_heads // self.tensor_parallel_size
            _validate_cache_tensor(
                layer_name,
                "K",
                k_cache,
                self.model_shape,
                local_kv_heads,
                self.cache_block_size,
            )
            _validate_cache_tensor(
                layer_name,
                "V",
                v_cache,
                self.model_shape,
                local_kv_heads,
                self.cache_block_size,
            )
            layer_caches.append(
                LayerCache(
                    name=layer_name,
                    layer_index=extract_layer_index(layer_name),
                    k_cache=k_cache,
                    v_cache=v_cache,
                )
            )

        self.runner = runner
        self.layer_caches = tuple(layer_caches)
        self.method.bind_model_runner(runner, self.layer_caches)
        pin_memory = bool(getattr(runner, "pin_memory", False))
        self._request_offsets_cpu = torch.zeros(
            runner.max_num_reqs,
            dtype=torch.int64,
            device="cpu",
            pin_memory=pin_memory,
        )
        self._request_offsets_device = torch.zeros(
            runner.max_num_reqs, dtype=torch.int64, device=runner.device
        )
        self._physical_positions = torch.empty(
            runner.max_num_tokens, dtype=torch.int64, device=runner.device
        )
        setattr(attention_block_table, RUNNER_PROVIDER_ATTRIBUTE, self)
        logger.info(
            "Ascend KV compression cache bound method=%s layers=%d "
            "threshold_tokens=%d target_tokens=%d min_output_tokens=%d "
            "logical_block_size=%d cache_block_size=%d",
            self.method.name,
            len(self.layer_caches),
            self.runtime_spec.compression_threshold_tokens,
            self.runtime_spec.max_physical_num_tokens,
            self.runtime_spec.min_output_tokens_for_compression,
            self.scheduler_block_size,
            self.cache_block_size,
        )

    def before_update_states(self, scheduler_output: Any) -> None:
        """Commit the prior step after its synchronous model execution barrier."""
        if self.runner is None:
            return
        reset_ids = set(scheduler_output.finished_req_ids)
        reset_ids.update(getattr(scheduler_output, "preempted_req_ids", ()))
        reset_ids.update(scheduler_output.scheduled_cached_reqs.resumed_req_ids)
        self.method.reset_requests(reset_ids)
        for request_id in reset_ids:
            self.pending.pop(request_id, None)
            self.active.pop(request_id, None)

        for request_id, pending in tuple(self.pending.items()):
            request = self.runner.requests.get(request_id)
            if request is None:
                self.pending.pop(request_id, None)
                continue
            mutable = list(pending.block_ids)
            group_block_ids = list(request.block_ids)
            group_block_ids[self.attention_group_index] = mutable
            request.block_ids = tuple(group_block_ids)
            request_index = self.runner.input_batch.req_id_to_index.get(request_id)
            if request_index is not None:
                self.runner.input_batch.block_table.add_row(
                    request.block_ids, request_index
                )
            self.active[request_id] = ActiveCompression(
                pending.semantic_anchor, pending.physical_anchor
            )
            logger.info(
                "KV compression worker commit acknowledged request_id=%s "
                "semantic_tokens=%d physical_tokens=%d destination_blocks=%d",
                request_id,
                pending.semantic_anchor,
                pending.physical_anchor,
                len(pending.block_ids),
            )
            self.pending.pop(request_id, None)

    def after_update_states(self, scheduler_output: Any) -> None:
        del scheduler_output
        self._physical_lengths_applied = False
        self._sync_request_offset_rows()

    def compress_scheduled_requests(self, scheduler_output: Any) -> None:
        """Run deterministic in-place transactions after the attention step."""
        if self.runner is None or not self.layer_caches:
            raise RuntimeError("Ascend KV compression provider is not cache-bound")
        for request_id, scheduled in scheduler_output.num_scheduled_tokens.items():
            if request_id in self.pending:
                continue
            request = self.runner.requests.get(request_id)
            if request is None:
                continue
            if (
                _request_max_tokens(request)
                < self.runtime_spec.min_output_tokens_for_compression
            ):
                continue
            semantic = int(request.num_computed_tokens) + int(scheduled)
            active = self.active.get(request_id)
            if active is None:
                physical = semantic
                if semantic != int(request.num_prompt_tokens):
                    continue
            else:
                physical = active.physical_anchor + semantic - active.semantic_anchor
            if physical < self.runtime_spec.compression_threshold_tokens:
                continue
            if self.attention_group_index >= len(request.block_ids):
                raise RuntimeError("request has no full-attention block table")
            source_ids = tuple(
                int(value) for value in request.block_ids[self.attention_group_index]
            )
            required_blocks = _blocks_for_tokens(
                self.runtime_spec.max_physical_num_tokens,
                self.scheduler_block_size,
            )
            if len(source_ids) < required_blocks:
                raise RuntimeError(
                    f"request {request_id!r} has too few blocks for compression"
                )
            destination_ids = source_ids[:required_blocks]
            source_device = _expand_scheduler_block_ids(
                source_ids,
                self.cache_blocks_per_scheduler_block,
                self.runner.device,
            )
            destination_device = source_device[
                : required_blocks * self.cache_blocks_per_scheduler_block
            ]
            result = self.method.compress(
                CompressionRequest(
                    request_id=request_id,
                    semantic_num_tokens=semantic,
                    physical_num_tokens=physical,
                    source_block_ids=(source_ids,),
                    destination_block_ids=(destination_ids,),
                    source_block_ids_device=source_device,
                    destination_block_ids_device=destination_device,
                )
            )
            compacted = int(result.physical_num_tokens)
            if compacted != self.runtime_spec.max_physical_num_tokens:
                raise RuntimeError(
                    f"method {self.method.name!r} returned unexpected "
                    f"length {compacted}"
                )
            self.pending[request_id] = PendingCompression(
                semantic_anchor=semantic,
                physical_anchor=compacted,
                block_ids=destination_ids,
            )

    def physical_positions_for_slot_mapping(
        self, positions: torch.Tensor
    ) -> torch.Tensor:
        """Map writes to compacted slots while preserving semantic RoPE positions."""
        if not self._has_active_rows:
            return positions
        if (
            self.runner is None
            or self._request_offsets_device is None
            or self._physical_positions is None
        ):
            raise RuntimeError("compression position buffers are not initialized")
        num_tokens = positions.numel()
        return shift_positions(
            positions,
            self.runner.req_indices.gpu[:num_tokens],
            self._request_offsets_device,
            self._physical_positions[:num_tokens],
        )

    def apply_physical_attention_lengths(self) -> None:
        """Replace semantic lengths only after host sampling decisions are built."""
        if self._physical_lengths_applied or not self._has_active_rows:
            return
        if (
            self.runner is None
            or self._request_offsets_cpu is None
            or self._request_offsets_device is None
        ):
            raise RuntimeError("compression length buffers are not initialized")
        num_reqs = self.runner.input_batch.num_reqs
        self.runner.seq_lens[:num_reqs].sub_(self._request_offsets_device[:num_reqs])
        self.runner.optimistic_seq_lens_cpu[:num_reqs].sub_(
            self._request_offsets_cpu[:num_reqs]
        )
        self._physical_lengths_applied = True

    def _sync_request_offset_rows(self) -> None:
        if (
            self.runner is None
            or self._request_offsets_cpu is None
            or self._request_offsets_device is None
        ):
            return
        num_reqs = self.runner.input_batch.num_reqs
        offsets = tuple(
            self.active[request_id].removed_tokens if request_id in self.active else 0
            for request_id in self.runner.input_batch.req_ids[:num_reqs]
        )
        if offsets == self._offset_row_snapshot:
            return
        rows = max(len(self._offset_row_snapshot), num_reqs)
        self._request_offsets_cpu[:rows].zero_()
        if offsets:
            self._request_offsets_cpu[:num_reqs].copy_(
                torch.tensor(offsets, dtype=torch.int64)
            )
        self._request_offsets_device[:rows].copy_(
            self._request_offsets_cpu[:rows], non_blocking=True
        )
        self._offset_row_snapshot = offsets
        self._has_active_rows = any(offsets)


def _common_compatibility_reasons(vllm_config: Any, runner: Any) -> tuple[str, ...]:
    reasons: list[str] = []
    cache_config = vllm_config.cache_config
    if bool(cache_config.enable_prefix_caching):
        reasons.append("prefix caching must be disabled")
    block_size = int(cache_config.block_size)
    allowed_block_sizes = _allowed_scheduler_block_sizes(vllm_config)
    if block_size not in allowed_block_sizes:
        reasons.append(
            f"logical KV block size must be one of {sorted(allowed_block_sizes)}, "
            f"got {block_size}"
        )
    if getattr(cache_config, "cache_dtype", "auto") not in {
        "auto",
        "bfloat16",
        "float16",
    }:
        reasons.append("quantized KV cache is unsupported")
    if vllm_config.speculative_config is not None:
        reasons.append("speculative decoding is unsupported")
    if vllm_config.kv_transfer_config is not None:
        reasons.append("KV transfer is unsupported")
    if bool(getattr(vllm_config.scheduler_config, "async_scheduling", False)):
        reasons.append("asynchronous scheduling is unsupported")
    model_config = vllm_config.model_config
    if bool(getattr(model_config, "is_hybrid", False)):
        text_config = getattr(model_config, "hf_text_config", None)
        model_type = str(getattr(text_config, "model_type", ""))
        if model_type not in {"qwen3_5_text", "qwen3_5_moe_text"}:
            reasons.append(
                "only Qwen3.5-style full-attention/Gated-DeltaNet hybrids are supported"
            )
        if getattr(cache_config, "mamba_cache_mode", "none") != "none":
            reasons.append("hybrid compression requires mamba_cache_mode='none'")
    if bool(getattr(model_config, "use_mla", False)):
        reasons.append("MLA cache layouts are unsupported")
    if bool(getattr(model_config, "is_encoder_decoder", False)):
        reasons.append("encoder-decoder models are unsupported")
    parallel = vllm_config.parallel_config
    tensor_parallel_size = int(getattr(parallel, "tensor_parallel_size", 1))
    text_config = getattr(model_config, "hf_text_config", None)
    num_kv_heads = int(getattr(text_config, "num_key_value_heads", 0))
    if tensor_parallel_size <= 0 or num_kv_heads % tensor_parallel_size:
        reasons.append(
            "tensor parallel size must be positive and divide the model KV heads"
        )
    for name, attr in (
        ("pipeline parallel", "pipeline_parallel_size"),
        ("data parallel", "data_parallel_size"),
        ("prefill context parallel", "prefill_context_parallel_size"),
        ("decode context parallel", "decode_context_parallel_size"),
    ):
        size = int(getattr(parallel, attr, 1))
        if size != 1:
            reasons.append(f"{name} size must be one, got {size}")
    runner_type = type(runner)
    if (
        runner_type.__module__ != "vllm_ascend.worker.model_runner_v1"
        or runner_type.__name__ != "NPUModelRunner"
    ):
        reasons.append("the standard v1 Ascend NPUModelRunner is required")
    if bool(getattr(runner, "use_sparse", False)):
        reasons.append("Ascend sparse attention is unsupported")
    if bool(getattr(runner, "use_compress", False)):
        reasons.append("model-native Ascend compression is unsupported")
    return tuple(reasons)


def _allowed_scheduler_block_sizes(vllm_config: Any) -> frozenset[int]:
    """Return fail-closed logical page sizes for a supported model family.

    Ascend keeps 128-token kernel cache blocks. Its hybrid cache adapter may
    promote the full-attention manager page to 2048 tokens, representing one
    logical attention block as 16 consecutive kernel blocks. The scheduler's
    cross-group LCM alignment is validated separately.
    """
    model_config = getattr(vllm_config, "model_config", None)
    if not bool(getattr(model_config, "is_hybrid", False)):
        return frozenset({ASCEND_BLOCK_SIZE})
    text_config = getattr(model_config, "hf_text_config", None)
    model_type = str(getattr(text_config, "model_type", ""))
    if model_type in {"qwen3_5_text", "qwen3_5_moe_text"}:
        return frozenset({ASCEND_BLOCK_SIZE, 2048})
    return frozenset({ASCEND_BLOCK_SIZE})


def _find_full_attention_group(groups: Any) -> int:
    """Return the sole full-attention group, allowing Mamba companion groups."""
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        MambaSpec,
        UniformTypeKVCacheSpecs,
    )

    # A few downstream host-contract tests intentionally use an opaque object
    # for the legacy single-group layout. Real vLLM groups always expose a
    # ``kv_cache_spec``; keep that narrow test-double compatibility without
    # weakening validation for hybrid layouts.
    if len(groups) == 1 and not hasattr(groups[0], "kv_cache_spec"):
        return 0

    full_groups: list[int] = []
    unsupported: list[str] = []
    for index, group in enumerate(groups):
        spec = group.kv_cache_spec
        specs = (
            tuple(spec.kv_cache_specs.values())
            if isinstance(spec, UniformTypeKVCacheSpecs)
            else (spec,)
        )
        if all(isinstance(item, FullAttentionSpec) for item in specs):
            full_groups.append(index)
        elif not all(isinstance(item, MambaSpec) for item in specs):
            unsupported.append(type(spec).__name__)
    if unsupported:
        raise RuntimeError(
            "Ascend KV compression supports only full-attention plus optional "
            "Mamba groups; found " + ", ".join(unsupported)
        )
    if len(full_groups) != 1:
        raise RuntimeError(
            "Ascend KV compression requires exactly one full-attention KV group"
        )
    return full_groups[0]


def _unpack_layer_cache(
    layer_name: str, layer: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    cache = getattr(layer, "kv_cache", None)
    if isinstance(cache, list) and len(cache) == 1:
        cache = cache[0]
    impl = getattr(layer, "impl", None)
    unpack = getattr(impl, "_unpack_kv_cache", None)
    if callable(unpack):
        k_cache, v_cache = unpack(cache)
    elif isinstance(cache, (tuple, list)) and len(cache) == 2:
        k_cache, v_cache = cache
    else:
        raise RuntimeError(
            f"attention layer {layer_name!r} does not expose dense Ascend K/V views"
        )
    if not isinstance(k_cache, torch.Tensor) or not isinstance(v_cache, torch.Tensor):
        raise RuntimeError(f"attention layer {layer_name!r} K/V bindings are invalid")
    return k_cache, v_cache


def _validate_cache_tensor(
    layer_name: str,
    kind: str,
    cache: torch.Tensor,
    model: ModelShape,
    expected_kv_heads: int | None = None,
    expected_block_size: int = ASCEND_BLOCK_SIZE,
) -> None:
    expected_tail = (
        expected_block_size,
        model.num_kv_heads if expected_kv_heads is None else expected_kv_heads,
        model.head_dim,
    )
    if cache.ndim != 4 or tuple(cache.shape[1:]) != expected_tail:
        raise RuntimeError(
            f"attention layer {layer_name!r} {kind} cache shape {tuple(cache.shape)} "
            f"does not end in {expected_tail}"
        )
    if cache.dtype not in {torch.bfloat16, torch.float16}:
        raise RuntimeError(
            f"attention layer {layer_name!r} {kind} cache dtype {cache.dtype} "
            "is unsupported"
        )
    if not cache.is_contiguous():
        raise RuntimeError(
            f"attention layer {layer_name!r} {kind} cache must be contiguous"
        )


def _validate_method_runtime_spec(name: str, spec: MethodRuntimeSpec) -> None:
    numeric = (
        spec.compression_threshold_tokens,
        spec.required_recompute_tokens,
        spec.max_physical_num_tokens,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in numeric
    ):
        raise ValueError(f"compression method {name!r} exposes invalid runtime limits")
    if spec.max_physical_num_tokens >= spec.compression_threshold_tokens:
        raise ValueError(
            f"compression method {name!r} maximum must be below its threshold"
        )
    minimum = spec.min_output_tokens_for_compression
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 0:
        raise ValueError(
            f"compression method {name!r} exposes an invalid output threshold"
        )


def _request_max_tokens(request: Any) -> int:
    value = getattr(request, "max_tokens", None)
    if value is None:
        sampling_params = getattr(request, "sampling_params", None)
        value = getattr(sampling_params, "max_tokens", None)
    if value is None and getattr(request, "pooling_params", None) is not None:
        value = 1
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(
            "current vLLM worker request must expose a non-negative output limit"
        )
    return value


def _blocks_for_tokens(num_tokens: int, block_size: int) -> int:
    return (num_tokens + block_size - 1) // block_size


def _expand_scheduler_block_ids(
    block_ids: tuple[int, ...],
    cache_blocks_per_scheduler_block: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Map scheduler block IDs to consecutive 128-token cache block IDs."""
    ids = torch.as_tensor(block_ids, device=device, dtype=torch.long)
    if cache_blocks_per_scheduler_block == 1:
        return ids
    offsets = torch.arange(
        cache_blocks_per_scheduler_block,
        device=device,
        dtype=torch.long,
    )
    return (ids.unsqueeze(1) * cache_blocks_per_scheduler_block + offsets).reshape(-1)


# Compatibility name retained for downstream imports.
TriAttentionAscendProvider = AscendKVCompressionProvider
