# SPDX-License-Identifier: Apache-2.0
"""Common Ascend adapter for registered KV compression methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from vllm.v1.kv_cache_compression import (
    KVCacheCompressionCompatibility,
    KVCacheCompressionPlan,
    KVCacheCompressionRuntimeSpec,
)

from .config import ASCEND_BLOCK_SIZE, PROVIDER_NAME, SCHEMA_VERSION, ProviderSelection
from .methods import create_method
from .methods.base import CompressionRequest, LayerCache, MethodRuntimeSpec, ModelShape
from .methods.triattention.kernels import shift_positions
from .model import model_shape_from_config

PROVIDER_FACTORY_QUALNAME = "vllm_ascend_kvcompress.provider:create_provider"
WORKER_PROVIDER_ATTRIBUTE = "_ascend_kvcompress_provider"
RUNNER_PROVIDER_ATTRIBUTE = "_ascend_kvcompress_provider"


@dataclass(frozen=True)
class PendingCompression:
    semantic_num_tokens: int
    physical_num_tokens: int
    destination_block_ids: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class ActiveCompression:
    semantic_anchor: int
    physical_anchor: int

    @property
    def removed_tokens(self) -> int:
        return self.semantic_anchor - self.physical_anchor


class AscendKVCompressionProvider:
    """Method-neutral bridge to vLLM-HUST's native compression lifecycle."""

    def __init__(self, vllm_config: Any) -> None:
        self.vllm_config = vllm_config
        self.selection = ProviderSelection.from_core_config(
            vllm_config.kv_cache_compression_config
        )
        self.provider_name = self.selection.provider_name
        self.model_shape = model_shape_from_config(vllm_config.model_config)
        self.method = create_method(
            self.selection.method,
            self.selection.method_config,
            vllm_config,
            self.model_shape,
        )
        self.method_runtime_spec = self.method.runtime_spec
        _validate_method_runtime_spec(self.method.name, self.method_runtime_spec)
        self.runner: Any | None = None
        self.layer_caches: tuple[LayerCache, ...] = ()
        self.pending: dict[str, PendingCompression] = {}
        self.active: dict[str, ActiveCompression] = {}
        self._request_offsets_cpu: torch.Tensor | None = None
        self._request_offsets_device: torch.Tensor | None = None
        self._physical_positions: torch.Tensor | None = None
        self._offset_row_snapshot: tuple[int, ...] = ()
        self._has_active_rows = False

    def compatibility_report(self, worker: Any) -> KVCacheCompressionCompatibility:
        """Return a serializable fail-closed report before KV allocation."""
        reasons = list(self._compatibility_reasons(worker))
        runtime_spec = None
        if not reasons:
            method_spec = self.method_runtime_spec
            runtime_spec = KVCacheCompressionRuntimeSpec(
                schema_version=SCHEMA_VERSION,
                provider=self.provider_name,
                requires_private_destination=method_spec.requires_private_destination,
                compression_threshold_tokens=method_spec.compression_threshold_tokens,
                required_recompute_tokens=method_spec.required_recompute_tokens,
                max_physical_num_tokens=method_spec.max_physical_num_tokens,
            )
        runner = getattr(worker, "model_runner", None)
        backend = getattr(runner, "attn_backend", None)
        architecture = _model_architecture(self.vllm_config.model_config)
        return KVCacheCompressionCompatibility(
            schema_version=SCHEMA_VERSION,
            provider=self.provider_name,
            supported=not reasons,
            reasons=tuple(reasons),
            platform=str(getattr(worker.current_platform, "device_type", "unknown")),
            provider_factory=PROVIDER_FACTORY_QUALNAME,
            backend=(
                f"{backend.__module__}.{backend.__name__}"
                if isinstance(backend, type)
                else type(backend).__module__ + "." + type(backend).__name__
                if backend is not None
                else None
            ),
            model_architecture=architecture,
            dtype=str(getattr(worker, "cache_dtype", None)),
            cache_layout="separate_kv_bshd",
            block_size=int(self.vllm_config.cache_config.block_size),
            runtime_spec=runtime_spec,
        )

    def _compatibility_reasons(self, worker: Any) -> tuple[str, ...]:
        reasons: list[str] = []
        if getattr(worker.current_platform, "device_type", None) != "npu":
            reasons.append("provider requires the Ascend NPU platform")

        from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

        if get_ascend_device_type() == AscendDeviceType._310P:
            reasons.append("Ascend 310P is unsupported by schema v1")

        worker_type = type(worker)
        if not (
            worker_type.__name__ == "NPUWorker"
            and worker_type.__module__ == "vllm_ascend.worker.worker"
        ):
            reasons.append(
                "provider requires the standard vllm_ascend.worker.worker.NPUWorker"
            )
        runner = getattr(worker, "model_runner", None)
        runner_type = type(runner)
        if not (
            runner_type.__name__ == "NPUModelRunner"
            and runner_type.__module__ == "vllm_ascend.worker.model_runner_v1"
        ):
            reasons.append("provider requires the standard v1 Ascend NPUModelRunner")
        backend = getattr(runner, "attn_backend", None)
        if not (
            isinstance(backend, type)
            and backend.__name__ == "AscendAttentionBackend"
            and backend.__module__ == "vllm_ascend.attention.attention_v1"
        ):
            reasons.append("provider requires the standard AscendAttentionBackend")
        if getattr(worker, "use_v2_model_runner", False):
            reasons.append("v2 model runner is unsupported")

        from vllm import envs

        prefix_caching = bool(self.vllm_config.cache_config.enable_prefix_caching)
        if (
            not prefix_caching
            and envs.VLLM_KNORM_ENABLED
            and envs.VLLM_KNORM_COMPRESSION_RATIO < 1.0
        ):
            reasons.append(
                "built-in Knorm KV compression conflicts with the selected method; "
                "set VLLM_KNORM_ENABLED=0"
            )

        parallel_config = self.vllm_config.parallel_config
        parallel_sizes = {
            "tensor parallel": getattr(parallel_config, "tensor_parallel_size", 1),
            "pipeline parallel": getattr(parallel_config, "pipeline_parallel_size", 1),
            "data parallel": getattr(parallel_config, "data_parallel_size", 1),
            "prefill context parallel": getattr(
                parallel_config, "prefill_context_parallel_size", 1
            ),
            "decode context parallel": getattr(
                parallel_config, "decode_context_parallel_size", 1
            ),
        }
        for name, size in parallel_sizes.items():
            if int(size) != 1:
                reasons.append(f"{name} size must be one, got {size}")

        model_config = self.vllm_config.model_config
        scheduler_config = self.vllm_config.scheduler_config
        if bool(getattr(scheduler_config, "async_scheduling", False)):
            reasons.append("asynchronous scheduling is not enabled in schema v1")
        if self.vllm_config.speculative_config is not None:
            reasons.append("speculative decoding is unsupported")
        if getattr(self.vllm_config, "kv_transfer_config", None) is not None:
            reasons.append("KV transfer is unsupported")
        if getattr(model_config, "is_hybrid", False):
            reasons.append("hybrid attention/state-space models are unsupported")
        if getattr(model_config, "use_mla", False):
            reasons.append("MLA cache layouts are unsupported")
        if getattr(runner, "use_sparse", False):
            reasons.append("Ascend sparse-attention cache layouts are unsupported")
        if getattr(runner, "use_compress", False):
            reasons.append("model-native Ascend KV compression is unsupported")
        if getattr(runner, "use_hybrid_blocks", False):
            reasons.append("hybrid allocation blocks are unsupported")
        if getattr(runner, "enable_hamming_sparse", False):
            reasons.append("Ascend Hamming sparse KV compression is unsupported")
        if getattr(runner, "is_pooling_model", False):
            reasons.append("pooling model runners are unsupported")

        cache_dtype = getattr(worker, "cache_dtype", None)
        if cache_dtype not in {torch.bfloat16, torch.float16}:
            reasons.append(
                f"KV cache dtype must be bfloat16 or float16, got {cache_dtype}"
            )
        if int(self.vllm_config.cache_config.block_size) != ASCEND_BLOCK_SIZE:
            reasons.append(
                f"KV cache block size must be {ASCEND_BLOCK_SIZE}, got "
                f"{self.vllm_config.cache_config.block_size}"
            )

        reasons.extend(_validate_worker_cache_specs(worker, self.model_shape))
        reasons.extend(
            f"method {self.method.name}: {reason}"
            for reason in self.method.compatibility_reasons(worker)
        )
        return tuple(reasons)

    def bind_model_runner(self, runner: Any, kv_cache_config: Any) -> None:
        """Validate common cache tensors, then bind the selected method."""
        from vllm.model_executor.models.utils import extract_layer_index
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        groups = kv_cache_config.kv_cache_groups
        if len(groups) != 1:
            raise RuntimeError("Ascend KV compression requires exactly one KV group")
        group = groups[0]
        if type(group.kv_cache_spec) is not FullAttentionSpec:
            raise RuntimeError(
                "Ascend KV compression requires a plain FullAttentionSpec"
            )
        _validate_plain_full_attention_spec(group.kv_cache_spec, "cache group")
        if group.kv_cache_spec.block_size != ASCEND_BLOCK_SIZE:
            raise RuntimeError(
                f"Ascend KV compression requires block size {ASCEND_BLOCK_SIZE}"
            )

        layer_caches: list[LayerCache] = []
        for layer_name in group.layer_names:
            module = runner.compilation_config.static_forward_context.get(layer_name)
            if module is None:
                raise RuntimeError(f"attention layer {layer_name!r} is not bound")
            bound_cache = getattr(module, "kv_cache", None)
            k_cache, v_cache = _unpack_separate_kv_cache(layer_name, bound_cache)
            _validate_cache_tensor(layer_name, "K", k_cache, self.model_shape)
            _validate_cache_tensor(layer_name, "V", v_cache, self.model_shape)
            if k_cache.shape != v_cache.shape:
                raise RuntimeError(
                    f"attention layer {layer_name!r} K/V shapes do not match"
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
        for block_table in runner.input_batch.block_table.block_tables:
            setattr(block_table, RUNNER_PROVIDER_ATTRIBUTE, self)

    def compress_transactions(
        self, scheduler_output: Any
    ) -> list[KVCacheCompressionPlan]:
        """Delegate initial and repeated stateful transactions to the method."""
        transaction_ids = scheduler_output.kv_cache_compression_transaction_ids or {}
        if not transaction_ids:
            return []
        if self.runner is None or not self.layer_caches:
            raise RuntimeError(
                "Ascend KV compression provider is not bound to KV cache"
            )

        plans: list[KVCacheCompressionPlan] = []
        for request_id in transaction_ids:
            if request_id in self.pending:
                raise RuntimeError(
                    f"request {request_id!r} already has pending compression state"
                )
            request = self.runner.requests.get(request_id)
            if request is None:
                raise RuntimeError(
                    f"compression request {request_id!r} is missing from model runner"
                )
            scheduled_tokens = scheduler_output.num_scheduled_tokens.get(request_id)
            if scheduled_tokens is None:
                raise RuntimeError(
                    f"compression request {request_id!r} was not scheduled"
                )
            semantic_num_tokens = request.num_computed_tokens + int(scheduled_tokens)
            active = self.active.get(request_id)
            if active is None:
                physical_num_tokens = semantic_num_tokens
                if semantic_num_tokens != request.num_prompt_tokens:
                    raise RuntimeError(
                        f"request {request_id!r} initial compression is not at "
                        f"final prefill: semantic={semantic_num_tokens}, "
                        f"prompt={request.num_prompt_tokens}"
                    )
            else:
                physical_num_tokens = (
                    active.physical_anchor
                    + semantic_num_tokens
                    - active.semantic_anchor
                )
            if (
                physical_num_tokens
                < self.method_runtime_spec.compression_threshold_tokens
            ):
                raise RuntimeError(
                    f"request {request_id!r} physical length {physical_num_tokens} "
                    "does not cross compression threshold"
                )

            source_block_ids = tuple(tuple(ids) for ids in request.block_ids)
            if len(source_block_ids) != 1:
                raise RuntimeError(
                    "Ascend KV compression requires exactly one block table"
                )
            destination = self._resolve_destination_blocks(
                scheduler_output, request_id, source_block_ids
            )
            source_ids_device = torch.tensor(
                source_block_ids[0], device=self.runner.device, dtype=torch.long
            )
            destination_ids_device = torch.tensor(
                destination[0], device=self.runner.device, dtype=torch.long
            )
            result = self.method.compress(
                CompressionRequest(
                    request_id=request_id,
                    semantic_num_tokens=semantic_num_tokens,
                    physical_num_tokens=physical_num_tokens,
                    source_block_ids=source_block_ids,
                    destination_block_ids=destination,
                    source_block_ids_device=source_ids_device,
                    destination_block_ids_device=destination_ids_device,
                )
            )
            physical_num_tokens = int(result.physical_num_tokens)
            if not (
                0
                < physical_num_tokens
                <= self.method_runtime_spec.max_physical_num_tokens
            ):
                raise RuntimeError(
                    f"method {self.method.name!r} produced invalid physical length "
                    f"{physical_num_tokens}"
                )
            required_blocks = _blocks_for_tokens(physical_num_tokens)
            committed_destination = (destination[0][:required_blocks],)
            layer_lengths = result.per_layer_physical_num_tokens
            if layer_lengths is None:
                layer_lengths = tuple(
                    (layer.name, physical_num_tokens) for layer in self.layer_caches
                )
            else:
                layer_lengths = _validate_method_layer_lengths(
                    self.method.name,
                    layer_lengths,
                    self.layer_caches,
                    physical_num_tokens,
                )
            plan = KVCacheCompressionPlan(
                schema_version=SCHEMA_VERSION,
                provider=self.provider_name,
                request_id=request_id,
                semantic_num_tokens=semantic_num_tokens,
                physical_num_tokens=physical_num_tokens,
                per_layer_physical_num_tokens=layer_lengths,
                expected_block_ids=source_block_ids,
                kv_cache_group_id=0,
            )
            self.pending[request_id] = PendingCompression(
                semantic_num_tokens=semantic_num_tokens,
                physical_num_tokens=physical_num_tokens,
                destination_block_ids=committed_destination,
            )
            plans.append(plan)
        return plans

    def _resolve_destination_blocks(
        self,
        scheduler_output: Any,
        request_id: str,
        source_block_ids: tuple[tuple[int, ...], ...],
    ) -> tuple[tuple[int, ...], ...]:
        required_blocks = _blocks_for_tokens(
            self.method_runtime_spec.max_physical_num_tokens
        )
        destinations = scheduler_output.kv_cache_compression_destination_block_ids
        if self.vllm_config.cache_config.enable_prefix_caching:
            if destinations is None or request_id not in destinations:
                raise RuntimeError(
                    f"request {request_id!r} has no private compression destination"
                )
            destination = tuple(tuple(group) for group in destinations[request_id])
        else:
            destination = (source_block_ids[0][:required_blocks],)
        if len(destination) != 1 or len(destination[0]) < required_blocks:
            raise RuntimeError(
                f"request {request_id!r} destination is smaller than method maximum"
            )
        return (destination[0][:required_blocks],)

    def consume_block_table_updates(self, scheduler_output: Any) -> None:
        """Apply scheduler commit acknowledgements and activate decode offsets."""
        if self.runner is None:
            return
        for request_id in scheduler_output.finished_req_ids:
            self.pending.pop(request_id, None)
            self.active.pop(request_id, None)
        updates = scheduler_output.kv_cache_compression_block_table_updates or {}
        for request_id, raw_tables in updates.items():
            pending = self.pending.pop(request_id, None)
            if pending is None:
                raise RuntimeError(
                    f"unexpected compression block-table update for {request_id!r}"
                )
            tables = tuple(tuple(group) for group in raw_tables)
            expected_prefix = pending.destination_block_ids[0]
            if len(tables) != 1 or tables[0][: len(expected_prefix)] != expected_prefix:
                raise RuntimeError(
                    f"request {request_id!r} compression commit acknowledgement "
                    "does not start with the materialized destination"
                )
            request = self.runner.requests.get(request_id)
            request_index = self.runner.input_batch.req_id_to_index.get(request_id)
            if request is None or request_index is None:
                raise RuntimeError(
                    f"committed compression request {request_id!r} is not active"
                )
            mutable_tables = tuple(list(group) for group in tables)
            request.block_ids = mutable_tables
            self.runner.input_batch.block_table.add_row(mutable_tables, request_index)
            self.active[request_id] = ActiveCompression(
                semantic_anchor=pending.semantic_num_tokens,
                physical_anchor=pending.physical_num_tokens,
            )
        self._sync_request_offset_rows()

    def _sync_request_offset_rows(self) -> None:
        """Update device offsets only when batch membership or state changes."""
        if (
            self.runner is None
            or self._request_offsets_cpu is None
            or self._request_offsets_device is None
        ):
            return
        num_reqs = self.runner.input_batch.num_reqs
        offsets = tuple(
            self.active[request_id].removed_tokens
            if request_id in self.active
            else 0
            for request_id in self.runner.input_batch.req_ids[:num_reqs]
        )
        if offsets == self._offset_row_snapshot:
            return
        previous_rows = len(self._offset_row_snapshot)
        rows_to_update = max(previous_rows, num_reqs)
        cpu = self._request_offsets_cpu
        cpu[:rows_to_update].zero_()
        if offsets:
            cpu[:num_reqs].copy_(torch.tensor(offsets, dtype=torch.int64))
        self._request_offsets_device[:rows_to_update].copy_(
            cpu[:rows_to_update], non_blocking=True
        )
        self._offset_row_snapshot = offsets
        self._has_active_rows = any(offsets)

    def physical_positions_for_slot_mapping(
        self, positions: torch.Tensor
    ) -> torch.Tensor:
        """Return physical cache positions without changing logical RoPE positions."""
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

    def apply_physical_decode_state(
        self,
        scheduler_output: Any,
        num_scheduled_tokens: np.ndarray,
    ) -> None:
        """Use compacted physical lengths while preserving semantic positions."""
        del num_scheduled_tokens
        if not self._has_active_rows:
            return
        if (
            self.runner is None
            or self._request_offsets_cpu is None
            or self._request_offsets_device is None
        ):
            raise RuntimeError(
                "Ascend KV compression decode buffers are not initialized"
            )
        num_reqs = self.runner.input_batch.num_reqs
        self.runner.seq_lens[:num_reqs].sub_(self._request_offsets_device[:num_reqs])
        self.runner.optimistic_seq_lens_cpu[:num_reqs].sub_(
            self._request_offsets_cpu[:num_reqs]
        )


def create_provider(worker: Any) -> AscendKVCompressionProvider:
    """Create or return the provider owned by one Ascend worker."""
    provider = getattr(worker, WORKER_PROVIDER_ATTRIBUTE, None)
    if provider is None:
        provider = AscendKVCompressionProvider(worker.vllm_config)
        setattr(worker, WORKER_PROVIDER_ATTRIBUTE, provider)
    if not isinstance(provider, AscendKVCompressionProvider):
        raise RuntimeError("worker KV compression provider has an invalid type")
    return provider


def unsupported_report(worker: Any, reason: str) -> KVCacheCompressionCompatibility:
    """Build a provider-shaped report when construction fails safely."""
    core_config = worker.vllm_config.kv_cache_compression_config
    return KVCacheCompressionCompatibility(
        schema_version=getattr(core_config, "schema_version", SCHEMA_VERSION),
        provider=getattr(core_config, "provider", PROVIDER_NAME),
        supported=False,
        reasons=(reason,),
        platform=str(getattr(worker.current_platform, "device_type", "unknown")),
        provider_factory=PROVIDER_FACTORY_QUALNAME,
    )


def _blocks_for_tokens(num_tokens: int) -> int:
    return (num_tokens + ASCEND_BLOCK_SIZE - 1) // ASCEND_BLOCK_SIZE


def _validate_method_runtime_spec(name: str, spec: MethodRuntimeSpec) -> None:
    if not isinstance(spec.requires_private_destination, bool):
        raise ValueError(
            f"compression method {name!r} runtime field "
            "'requires_private_destination' must be a boolean"
        )
    numeric = {
        "compression_threshold_tokens": spec.compression_threshold_tokens,
        "required_recompute_tokens": spec.required_recompute_tokens,
        "max_physical_num_tokens": spec.max_physical_num_tokens,
    }
    for field, value in numeric.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"compression method {name!r} runtime field {field!r} must be "
                "a positive integer"
            )
    if spec.max_physical_num_tokens >= spec.compression_threshold_tokens:
        raise ValueError(
            f"compression method {name!r} maximum physical length must be below "
            "its compression threshold"
        )


def _validate_method_layer_lengths(
    method_name: str,
    layer_lengths: tuple[tuple[str, int], ...],
    layer_caches: tuple[LayerCache, ...],
    physical_num_tokens: int,
) -> tuple[tuple[str, int], ...]:
    expected_names = {layer.name for layer in layer_caches}
    normalized: list[tuple[str, int]] = []
    seen_names: set[str] = set()
    for entry in layer_lengths:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise RuntimeError(
                f"method {method_name!r} returned an invalid per-layer entry"
            )
        layer_name, raw_length = entry
        if not isinstance(layer_name, str) or not layer_name:
            raise RuntimeError(f"method {method_name!r} returned an invalid layer name")
        if layer_name in seen_names:
            raise RuntimeError(
                f"method {method_name!r} returned duplicate layer {layer_name!r}"
            )
        if (
            isinstance(raw_length, bool)
            or not isinstance(raw_length, int)
            or not 0 < raw_length <= physical_num_tokens
        ):
            raise RuntimeError(
                f"method {method_name!r} returned invalid physical length "
                f"{raw_length!r} for layer {layer_name!r}"
            )
        seen_names.add(layer_name)
        normalized.append((layer_name, raw_length))

    if seen_names != expected_names:
        missing = sorted(expected_names - seen_names)
        unknown = sorted(seen_names - expected_names)
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unknown:
            details.append("unknown=" + ",".join(unknown))
        raise RuntimeError(
            f"method {method_name!r} returned incomplete per-layer lengths "
            f"({' '.join(details)})"
        )
    return tuple(normalized)


def _model_architecture(model_config: Any) -> str | None:
    architectures = getattr(model_config, "architectures", None)
    if architectures:
        return str(architectures[0])
    hf_config = getattr(model_config, "hf_config", None)
    architectures = getattr(hf_config, "architectures", None)
    return str(architectures[0]) if architectures else None


def _validate_worker_cache_specs(worker: Any, model: ModelShape) -> tuple[str, ...]:
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    reasons: list[str] = []
    try:
        cache_specs = worker.get_kv_cache_spec()
    except Exception as error:
        return (f"failed to inspect KV cache specs: {type(error).__name__}: {error}",)
    if len(cache_specs) != model.num_layers:
        reasons.append(
            f"worker exposes {len(cache_specs)} KV layers; expected {model.num_layers}"
        )
    for layer_name, spec in cache_specs.items():
        if type(spec) is not FullAttentionSpec:
            reasons.append(
                f"layer {layer_name!r} does not use a plain FullAttentionSpec"
            )
            continue
        spec_reasons = _plain_full_attention_spec_reasons(spec)
        reasons.extend(f"layer {layer_name!r} {reason}" for reason in spec_reasons)
        if spec.block_size != ASCEND_BLOCK_SIZE:
            reasons.append(
                f"layer {layer_name!r} block size is {spec.block_size}, expected "
                f"{ASCEND_BLOCK_SIZE}"
            )
        if spec.dtype not in {torch.bfloat16, torch.float16}:
            reasons.append(
                f"layer {layer_name!r} KV dtype is {spec.dtype}, expected BF16/FP16"
            )
        if spec.num_kv_heads != model.num_kv_heads:
            reasons.append(
                f"layer {layer_name!r} has {spec.num_kv_heads} KV heads, expected "
                f"{model.num_kv_heads}"
            )
        if spec.head_size != model.head_dim:
            reasons.append(
                f"layer {layer_name!r} head size is {spec.head_size}, expected "
                f"{model.head_dim}"
            )
    return tuple(reasons)


def _plain_full_attention_spec_reasons(spec: Any) -> tuple[str, ...]:
    from vllm.v1.kv_cache_interface import KVQuantMode

    reasons: list[str] = []
    if getattr(spec, "sliding_window", None) is not None:
        reasons.append("uses sliding-window attention")
    if getattr(spec, "attention_chunk_size", None) is not None:
        reasons.append("uses chunked attention semantics")
    if bool(getattr(spec, "non_causal", False)):
        reasons.append("uses non-causal attention")
    if getattr(spec, "head_size_v", spec.head_size) != spec.head_size:
        reasons.append("uses unequal K/V head dimensions")
    kv_quant_mode = getattr(spec, "kv_quant_mode", None)
    if kv_quant_mode is not None and kv_quant_mode != KVQuantMode.NONE:
        reasons.append(f"uses quantized KV layout {kv_quant_mode}")
    if getattr(spec, "page_size_padded", None) is not None:
        reasons.append("uses padded KV pages")
    if bool(getattr(spec, "indexes_kv_by_block_stride", False)):
        reasons.append("uses block-strided KV indexing")
    return tuple(reasons)


def _validate_plain_full_attention_spec(spec: Any, owner: str) -> None:
    reasons = _plain_full_attention_spec_reasons(spec)
    if reasons:
        raise RuntimeError(f"{owner} " + "; ".join(reasons))


def _unpack_separate_kv_cache(
    layer_name: str, bound_cache: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return K/V tensors from the current or legacy Ascend binding shape."""
    cache = bound_cache
    if (
        isinstance(cache, list)
        and len(cache) == 1
        and isinstance(cache[0], (tuple, list))
    ):
        cache = cache[0]
    if not isinstance(cache, (tuple, list)) or len(cache) != 2:
        raise RuntimeError(
            f"attention layer {layer_name!r} must use separate K/V tensors"
        )
    k_cache, v_cache = cache
    if not isinstance(k_cache, torch.Tensor) or not isinstance(v_cache, torch.Tensor):
        raise RuntimeError(
            f"attention layer {layer_name!r} K/V bindings must be tensors"
        )
    return k_cache, v_cache


def _validate_cache_tensor(
    layer_name: str, kind: str, cache: Any, model: ModelShape
) -> None:
    if not isinstance(cache, torch.Tensor) or cache.ndim != 4:
        raise RuntimeError(
            f"attention layer {layer_name!r} {kind} cache must be rank-four tensor"
        )
    expected_tail = (ASCEND_BLOCK_SIZE, model.num_kv_heads, model.head_dim)
    if tuple(cache.shape[1:]) != expected_tail:
        raise RuntimeError(
            f"attention layer {layer_name!r} {kind} cache shape "
            f"{tuple(cache.shape)} does not end in {expected_tail}"
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


# Compatibility alias for downstream imports from the first release.
TriAttentionAscendProvider = AscendKVCompressionProvider
