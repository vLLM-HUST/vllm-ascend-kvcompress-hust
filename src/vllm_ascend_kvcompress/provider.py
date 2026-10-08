# SPDX-License-Identifier: Apache-2.0
"""Worker-side adapter for the upstream-aligned vLLM/Ascend runtime.

The current hosts intentionally expose only the standard ``vllm.general_plugins``
entry point. This module owns its state instead of importing fork-only
KV-compression types. All integration with non-public runtime symbols is checked
at startup and kept in :mod:`vllm_ascend_kvcompress.plugin`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any

import torch
from vllm.logger import logger

from .config import ASCEND_BLOCK_SIZE, ProviderSelection
from .methods import create_method
from .methods.base import (
    CompressionRequest,
    KVCompressionMethod,
    LayerCache,
    MethodRuntimeSpec,
    ModelShape,
    QueryBatchSpan,
    QueryObservation,
)
from .methods.triattention.kernels import shift_positions
from .model import model_shape_from_config
from .transaction import PLAN_ATTRIBUTE

RUNNER_PROVIDER_ATTRIBUTE = "_ascend_kvcompress_provider_v3"
ATTENTION_QUERY_PROVIDER_ATTRIBUTE = "_ascend_kvcompress_query_provider_v1"


@dataclass(frozen=True)
class PendingCompression:
    semantic_anchor: int
    physical_anchor: int
    block_ids: tuple[int, ...]
    per_layer_physical_num_tokens: tuple[tuple[str, int], ...] | None = None


@dataclass(frozen=True)
class ActiveCompression:
    semantic_anchor: int
    physical_anchor: int
    per_layer_physical_num_tokens: tuple[tuple[str, int], ...] | None = None

    @property
    def removed_tokens(self) -> int:
        return self.semantic_anchor - self.physical_anchor


class AscendKVCompressionProvider:
    """Own one model runner's TriAttention state and compacted length mapping."""

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
        self.requires_per_layer_physical_state = (
            self.method.requires_per_layer_physical_state
        )
        if not isinstance(self.requires_per_layer_physical_state, bool):
            raise TypeError(
                "method requires_per_layer_physical_state must be a boolean"
            )
        self.query_window_tokens = self.method.query_window_tokens
        if (
            isinstance(self.query_window_tokens, bool)
            or not isinstance(self.query_window_tokens, int)
            or self.query_window_tokens < 0
        ):
            raise ValueError(
                "method query_window_tokens must be a non-negative integer"
            )
        if self.query_window_tokens and any(
            getattr(type(self.method), name) is getattr(KVCompressionMethod, name)
            for name in (
                "capture_query",
                "complete_query_observation",
                "discard_query_observation",
            )
        ):
            raise TypeError(
                "a query-observing method must implement capture, completion, "
                "and discard hooks"
            )
        self.runner: Any | None = None
        self.layer_caches: tuple[LayerCache, ...] = ()
        self.query_layer_caches: tuple[LayerCache, ...] = ()
        self.speculative_cache_layer: LayerCache | None = None
        self.pending: dict[str, PendingCompression] = {}
        self.active: dict[str, ActiveCompression] = {}
        self._request_offsets_cpu: torch.Tensor | None = None
        self._request_offsets_device: torch.Tensor | None = None
        self._physical_positions: torch.Tensor | None = None
        self._semantic_seq_lens_device: torch.Tensor | None = None
        self._semantic_seq_lens_cpu: torch.Tensor | None = None
        self._offset_row_snapshot: tuple[int, ...] = ()
        self._has_active_rows = False
        self._has_per_layer_rows = False
        self._per_layer_metadata_seen = False
        self._per_layer_step_requires_metadata = False
        self._per_layer_slot_buffers: dict[str, torch.Tensor] = {}
        self._physical_lengths_applied = False
        self._query_step_output: Any | None = None
        self._query_seen_layers: set[int] = set()
        self._query_forward_complete = False
        self._query_tracked_ids: set[str] = set()

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
        self.query_layer_caches = self._resolve_query_layer_caches()
        mtp_caches = tuple(
            cache for cache in self.layer_caches if ".mtp.layers." in f".{cache.name}."
        )
        if getattr(self.vllm_config, "speculative_config", None) is not None:
            if len(mtp_caches) != 1:
                raise RuntimeError(
                    "qualified MTP2 requires exactly one auxiliary attention cache"
                )
            self.speculative_cache_layer = mtp_caches[0]
        if self.requires_per_layer_physical_state:
            self._validate_per_layer_host()
        if self.query_window_tokens:
            for cache in self.query_layer_caches:
                layer = context[cache.name]
                setattr(layer, ATTENTION_QUERY_PROVIDER_ATTRIBUTE, self)
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
        if self.requires_per_layer_physical_state:
            self._per_layer_slot_buffers = {
                cache.name: torch.empty(
                    runner.max_num_tokens,
                    dtype=torch.int64,
                    device=runner.device,
                )
                for cache in self.layer_caches
            }
        self._semantic_seq_lens_device = torch.empty_like(runner.seq_lens)
        self._semantic_seq_lens_cpu = torch.empty_like(runner.optimistic_seq_lens_cpu)
        setattr(attention_block_table, RUNNER_PROVIDER_ATTRIBUTE, self)
        logger.info(
            "Ascend KV compression cache bound method=%s "
            "materialization_layers=%d calibrated_scoring_layers=%d "
            "threshold_tokens=%d target_tokens=%d min_output_tokens=%d "
            "logical_block_size=%d cache_block_size=%d",
            self.method.name,
            len(self.layer_caches),
            len(getattr(self.method, "layer_caches", self.layer_caches)),
            self.runtime_spec.compression_threshold_tokens,
            self.runtime_spec.max_physical_num_tokens,
            self.runtime_spec.min_output_tokens_for_compression,
            self.scheduler_block_size,
            self.cache_block_size,
        )

    def _resolve_query_layer_caches(self) -> tuple[LayerCache, ...]:
        if not self.query_window_tokens:
            return ()
        requested = self.method.query_layer_indices
        if requested is None:
            return self.layer_caches
        if (
            not isinstance(requested, tuple)
            or not requested
            or any(
                isinstance(index, bool) or not isinstance(index, int)
                for index in requested
            )
            or len(set(requested)) != len(requested)
        ):
            raise RuntimeError(
                "query_layer_indices must be a non-empty tuple of unique integers"
            )
        by_index: dict[int, LayerCache] = {}
        duplicate_indices: set[int] = set()
        for cache in self.layer_caches:
            if cache.layer_index in by_index:
                duplicate_indices.add(cache.layer_index)
            else:
                by_index[cache.layer_index] = cache
        if duplicate_indices & set(requested):
            raise RuntimeError(
                "query_layer_indices are ambiguous across bound cache layers"
            )
        if any(index not in by_index for index in requested):
            raise RuntimeError("query_layer_indices contain an unbound model layer")
        return tuple(by_index[index] for index in requested)

    def before_update_states(self, scheduler_output: Any) -> None:
        """Commit the prior step after its synchronous model execution barrier."""
        if self.runner is None:
            return
        reset_ids = set(scheduler_output.finished_req_ids)
        reset_ids.update(getattr(scheduler_output, "preempted_req_ids", ()))
        reset_ids.update(scheduler_output.scheduled_cached_reqs.resumed_req_ids)
        for request_id in reset_ids:
            self.pending.pop(request_id, None)
            self.active.pop(request_id, None)
            if self.query_window_tokens:
                self.method.discard_query_observation(request_id)
                self._query_tracked_ids.discard(request_id)

        for request_id, pending in tuple(self.pending.items()):
            # An async scheduler can execute other requests while this
            # compression output is still in flight. The scheduler freezes
            # this request until acknowledgement, so its next scheduled step
            # is the commit boundary on the worker too.
            if request_id not in scheduler_output.num_scheduled_tokens:
                continue
            request = self.runner.requests.get(request_id)
            if request is None:
                self.pending.pop(request_id, None)
                if self.query_window_tokens:
                    self.method.discard_query_observation(request_id)
                    self._query_tracked_ids.discard(request_id)
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
                pending.semantic_anchor,
                pending.physical_anchor,
                pending.per_layer_physical_num_tokens,
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

    def capture_attention_query(self, layer_name: str, query: torch.Tensor) -> None:
        """Observe query rows from the attention custom-op execution path.

        Module forward hooks run while Dynamo traces a piecewise graph but are
        not invoked when the compiled graph is replayed.  Ascend's split
        ``unified_attention_with_output`` custom op does execute its backend
        forward for every prefill replay, so the plugin wrapper calls this
        method there.  Keeping the request spans outside the graph also avoids
        baking one scheduler batch into the compiled program.
        """
        if self._query_step_output is None:
            return
        cache = next(
            (cache for cache in self.query_layer_caches if cache.name == layer_name),
            None,
        )
        if cache is None:
            raise RuntimeError(
                f"attention layer {layer_name!r} is not a query-observation layer"
            )
        spans = self._query_spans()
        if not spans:
            return
        if query.ndim < 2 or max(span.end for span in spans) > query.shape[0]:
            raise RuntimeError(f"attention layer {cache.name!r} query rows changed")
        self.method.capture_query(cache, query, spans)
        self._query_tracked_ids.update(span.request_id for span in spans)
        self._query_seen_layers.add(cache.layer_index)

    def _query_spans(self) -> tuple[QueryBatchSpan, ...]:
        output = self._query_step_output
        if output is None or self.runner is None:
            return ()
        batch = self.runner.input_batch
        offsets = self.runner.query_start_loc.cpu
        spans: list[QueryBatchSpan] = []
        for request_id, scheduled in output.num_scheduled_tokens.items():
            request = self.runner.requests.get(request_id)
            index = batch.req_id_to_index.get(request_id)
            if request is None or index is None or scheduled <= 0:
                continue
            remaining = int(request.num_prompt_tokens) - int(
                request.num_computed_tokens
            )
            if remaining <= 0:
                continue
            start = int(offsets[index])
            end = start + min(int(scheduled), remaining)
            if end > int(offsets[index + 1]):
                raise RuntimeError("prefill query rows exceed the runner batch span")
            spans.append(QueryBatchSpan(request_id, start, end))
        return tuple(spans)

    def begin_query_step(self, scheduler_output: Any) -> None:
        if not self.query_window_tokens:
            return
        for request_id in self._query_tracked_ids - self.runner.requests.keys():
            self.method.discard_query_observation(request_id)
            self._query_tracked_ids.discard(request_id)
        self._query_step_output = scheduler_output
        self._query_seen_layers.clear()
        self._query_forward_complete = False

    def finish_query_step(self) -> None:
        if not self.query_window_tokens:
            return
        try:
            if self._query_spans() and self._query_seen_layers != {
                layer.layer_index for layer in self.query_layer_caches
            }:
                raise RuntimeError(
                    "query observation missed a full-attention layer; "
                    "the current graph replay path is unvalidated"
                )
            self._query_forward_complete = True
        finally:
            self._query_step_output = None

    def abort_query_step(self, scheduler_output: Any) -> None:
        if not self.query_window_tokens:
            return
        self._query_step_output = None
        self._query_forward_complete = False
        self._query_seen_layers.clear()
        for request_id in scheduler_output.num_scheduled_tokens:
            self.method.discard_query_observation(request_id)
            self._query_tracked_ids.discard(request_id)

    def after_update_states(self, scheduler_output: Any) -> None:
        del scheduler_output
        if self.runner is not None:
            live = self.runner.requests.keys()
            for request_id in (self.pending.keys() | self.active.keys()) - live:
                self.pending.pop(request_id, None)
                self.active.pop(request_id, None)
        self._physical_lengths_applied = False
        self._sync_request_offset_rows()

    def compress_scheduled_requests(self, scheduler_output: Any) -> None:
        """Run scheduler-authorized transactions after the attention step."""
        if self.runner is None or not self.layer_caches:
            raise RuntimeError("Ascend KV compression provider is not cache-bound")
        plans = getattr(scheduler_output, PLAN_ATTRIBUTE, {})
        for request_id, scheduled in scheduler_output.num_scheduled_tokens.items():
            plan = plans.get(request_id)
            if plan is None:
                continue
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
            if semantic != plan.semantic_anchor:
                raise RuntimeError(
                    f"request {request_id!r} semantic length disagrees with "
                    "the scheduler compression plan"
                )
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
            if source_ids != plan.source_block_ids:
                raise RuntimeError(
                    f"request {request_id!r} source block table disagrees "
                    "with the scheduler compression plan"
                )
            destination_ids = plan.destination_block_ids
            if len(destination_ids) != required_blocks or (
                set(destination_ids) & set(source_ids)
            ):
                raise RuntimeError(
                    f"request {request_id!r} requires private destination blocks"
                )
            source_device = _expand_scheduler_block_ids(
                source_ids,
                self.cache_blocks_per_scheduler_block,
                self.runner.device,
            )
            destination_device = _expand_scheduler_block_ids(
                destination_ids,
                self.cache_blocks_per_scheduler_block,
                self.runner.device,
            )
            if self.query_window_tokens:
                if not self._query_forward_complete:
                    raise RuntimeError(
                        "query observation requires a completed model forward"
                    )
                self.method.complete_query_observation(
                    QueryObservation(
                        request_id=request_id,
                        plan=plan,
                        semantic_num_tokens=semantic,
                        window_tokens=self.query_window_tokens,
                        layer_indices=tuple(
                            layer.layer_index for layer in self.query_layer_caches
                        ),
                    )
                )
            try:
                result = self.method.compress(
                    CompressionRequest(
                        request_id=request_id,
                        semantic_num_tokens=semantic,
                        physical_num_tokens=physical,
                        source_block_ids=(source_ids,),
                        destination_block_ids=(destination_ids,),
                        source_block_ids_device=source_device,
                        destination_block_ids_device=destination_device,
                        plan=plan,
                        per_layer_physical_num_tokens=(
                            tuple(
                                (
                                    name,
                                    length + semantic - active.semantic_anchor,
                                )
                                for name, length in active.per_layer_physical_num_tokens
                            )
                            if active is not None
                            and active.per_layer_physical_num_tokens is not None
                            else None
                        ),
                    )
                )
            finally:
                if self.query_window_tokens:
                    self.method.discard_query_observation(request_id)
                    self._query_tracked_ids.discard(request_id)
            compacted = result.physical_num_tokens
            if isinstance(compacted, bool) or not isinstance(compacted, int):
                raise RuntimeError("method returned a non-integer physical length")
            if compacted != self.runtime_spec.max_physical_num_tokens:
                raise RuntimeError(
                    f"method {self.method.name!r} returned unexpected "
                    f"length {compacted}"
                )
            per_layer = self._validate_per_layer_lengths(
                result.per_layer_physical_num_tokens, compacted
            )
            self.pending[request_id] = PendingCompression(
                semantic_anchor=semantic,
                physical_anchor=compacted,
                block_ids=destination_ids,
                per_layer_physical_num_tokens=per_layer,
            )

    def _validate_per_layer_lengths(
        self, lengths: tuple[tuple[str, int], ...] | None, maximum: int
    ) -> tuple[tuple[str, int], ...] | None:
        if lengths is None:
            return None
        names = tuple(layer.name for layer in self.layer_caches)
        if not isinstance(lengths, tuple) or len(lengths) != len(names):
            raise RuntimeError(
                "per-layer physical lengths must cover every full-attention layer"
            )
        values: dict[str, int] = {}
        for entry in lengths:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise RuntimeError("invalid per-layer physical length entry")
            name, length = entry
            if not isinstance(name, str) or name in values or name not in names:
                raise RuntimeError(
                    "per-layer physical lengths contain an unknown or duplicate layer"
                )
            if (
                isinstance(length, bool)
                or not isinstance(length, int)
                or not 0 < length <= maximum
            ):
                raise RuntimeError(
                    "per-layer physical lengths must be positive and <= group maximum"
                )
            values[name] = length
        if set(values) != set(names):
            raise RuntimeError("per-layer physical lengths omit a full-attention layer")
        ordered = tuple((name, values[name]) for name in names)
        if any(length != maximum for _, length in ordered):
            self._validate_per_layer_host()
            return ordered
        # Uniform methods, including TriAttention, retain the existing fast path.
        return None

    def _validate_per_layer_host(self) -> None:
        if self.runner is None:
            raise RuntimeError("per-layer physical state requires a bound runner")
        mode = getattr(
            getattr(self.runner, "compilation_config", None), "cudagraph_mode", None
        )
        mode_name = getattr(mode, "name", None)
        if mode_name == "FULL_AND_PIECEWISE":
            if not self.requires_per_layer_physical_state:
                raise RuntimeError(
                    "graph replay requires the method to declare per-layer "
                    "physical state before capture"
                )
            try:
                from vllm_ascend.attention.attention_v1 import (
                    AscendAttentionBackendImpl,
                )
            except (ImportError, OSError, RuntimeError) as error:
                raise RuntimeError(
                    "per-layer graph replay backend is unavailable"
                ) from error
            if not getattr(
                AscendAttentionBackendImpl,
                "_ascend_kvcompress_layer_aware_graph_v1",
                False,
            ):
                raise RuntimeError(
                    "per-layer graph replay requires the layer-aware FIA update seam"
                )
        elif mode_name != "NONE":
            raise RuntimeError(
                "per-layer physical state supports eager or "
                "FULL_AND_PIECEWISE execution only"
            )
        if any(
            getattr(self.runner, name, False) for name in ("use_dcp", "pcp_enabled")
        ):
            raise RuntimeError(
                "per-layer physical state is unvalidated with "
                "context-parallel attention"
            )
        speculative = getattr(self.vllm_config, "speculative_config", None)
        if speculative is not None and not _supports_mtp2(self.vllm_config):
            raise RuntimeError(
                "per-layer physical state supports only the qualified Qwen3.5 MTP2 path"
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
            or self._semantic_seq_lens_device is None
            or self._semantic_seq_lens_cpu is None
        ):
            raise RuntimeError("compression length buffers are not initialized")
        num_reqs = self.runner.input_batch.num_reqs
        self._semantic_seq_lens_device[:num_reqs].copy_(self.runner.seq_lens[:num_reqs])
        self._semantic_seq_lens_cpu[:num_reqs].copy_(
            self.runner.optimistic_seq_lens_cpu[:num_reqs]
        )
        self.runner.seq_lens[:num_reqs].sub_(self._request_offsets_device[:num_reqs])
        self.runner.optimistic_seq_lens_cpu[:num_reqs].sub_(
            self._request_offsets_cpu[:num_reqs]
        )
        self._physical_lengths_applied = True

    def semantic_gdn_metadata(self, metadata: Any) -> Any:
        """Give GDN its original recurrent positions, not attention offsets."""
        if not self._physical_lengths_applied:
            return metadata
        if (
            self._semantic_seq_lens_device is None
            or self._semantic_seq_lens_cpu is None
        ):
            raise RuntimeError("semantic GDN length buffers are not initialized")
        num_reqs = metadata.num_reqs
        return replace(
            metadata,
            seq_lens=self._semantic_seq_lens_device[:num_reqs],
            _seq_lens_cpu=self._semantic_seq_lens_cpu[:num_reqs],
            seq_lens_cpu_upper_bound=self._semantic_seq_lens_cpu[:num_reqs],
        )

    def per_layer_attention_metadata(
        self, result: Any, num_tokens: int, num_reqs: int
    ) -> Any:
        """Give each full-attention layer distinct KV lengths and write slots."""
        if not (self.requires_per_layer_physical_state or self._has_per_layer_rows):
            return result
        self._validate_per_layer_host()
        if self.runner is None or not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("per-layer attention metadata hook was bypassed")
        metadata, common = result
        if not isinstance(metadata, dict):
            raise RuntimeError(
                "per-layer attention requires a single eager metadata batch"
            )
        runner = self.runner
        batch = runner.input_batch
        request_ids = batch.req_ids[:num_reqs]
        if num_tokens < 0:
            raise RuntimeError("per-layer attention batch dimensions changed")
        if len(request_ids) != num_reqs:
            # FULL graph capture builds synthetic decode batches before the
            # runner has admitted any requests.  There are no compressed rows
            # to specialize in that phase, and the native layer-keyed metadata
            # must remain intact so Ascend can record the graph tasks.  Keep
            # failing closed for a real or partially populated input batch.
            batch_num_reqs = int(getattr(batch, "num_reqs", len(batch.req_ids)))
            if self._has_active_rows or batch_num_reqs != 0 or batch.req_ids:
                raise RuntimeError("per-layer attention batch dimensions changed")
            self._per_layer_metadata_seen = True
            return result
        table = batch.block_table.block_tables[self.attention_group_index]
        native = getattr(
            type(table), "_ascend_kvcompress_patch_v3_slot_mapping_original", None
        )
        if not callable(native):
            raise RuntimeError(
                "per-layer attention requires the native slot mapping hook"
            )
        if self._request_offsets_cpu is None:
            raise RuntimeError("per-layer attention offset buffers are unavailable")
        if self._has_active_rows and not self._physical_lengths_applied:
            raise RuntimeError("per-layer attention requires saved semantic lengths")
        global_offsets = tuple(
            int(value) for value in self._request_offsets_cpu[:num_reqs]
        )
        original_slots = table.slot_mapping.gpu[:num_tokens].clone()
        speculative_view: (
            tuple[list[int], tuple[int, ...], torch.Tensor, torch.Tensor] | None
        ) = None
        try:
            for layer in self.layer_caches:
                offsets = tuple(
                    self.active[request_id].semantic_anchor
                    - dict(
                        self.active[request_id].per_layer_physical_num_tokens or ()
                    ).get(layer.name, self.active[request_id].physical_anchor)
                    if request_id in self.active
                    else 0
                    for request_id in request_ids
                )
                base = metadata.get(layer.name)
                if (
                    type(base).__name__ != "AscendMetadata"
                    or type(base).__module__ != "vllm_ascend.attention.attention_v1"
                    or base.seq_lens is None
                    or base.seq_lens_list is None
                    or base.slot_mapping is None
                ):
                    raise RuntimeError(
                        "per-layer attention backend metadata is unvalidated"
                    )
                delta = [
                    global_offset - offset
                    for global_offset, offset in zip(
                        global_offsets, offsets, strict=True
                    )
                ]
                seq_lens = base.seq_lens.clone()
                if any(delta):
                    seq_lens[:num_reqs] += torch.tensor(
                        delta, dtype=seq_lens.dtype, device=seq_lens.device
                    )
                seq_lens_list = list(base.seq_lens_list)
                for index, correction in enumerate(delta):
                    seq_lens_list[index] += correction
                slots = self._per_layer_slot_buffers.get(layer.name)
                if slots is None and not self.requires_per_layer_physical_state:
                    slots = torch.empty(
                        max(num_tokens, int(getattr(runner, "max_num_tokens", 0))),
                        dtype=torch.int64,
                        device=original_slots.device,
                    )
                    self._per_layer_slot_buffers[layer.name] = slots
                if slots is None or slots.numel() < num_tokens:
                    raise RuntimeError(
                        "per-layer attention slot buffers were not prepared "
                        "before capture"
                    )
                if offsets == global_offsets:
                    slots[:num_tokens].copy_(original_slots)
                else:
                    positions = runner.positions[:num_tokens]
                    req_indices = runner.req_indices.gpu[:num_tokens]
                    offset_tensor = torch.tensor(
                        offsets, dtype=positions.dtype, device=positions.device
                    )
                    native(
                        table,
                        num_reqs,
                        runner.query_start_loc.gpu[: num_reqs + 1],
                        positions - offset_tensor[req_indices.long()],
                    )
                    slots[:num_tokens].copy_(table.slot_mapping.gpu[:num_tokens])
                metadata[layer.name] = replace(
                    base,
                    seq_lens=seq_lens,
                    seq_lens_cpu=seq_lens,
                    seq_lens_list=seq_lens_list,
                    slot_mapping=slots[:num_tokens],
                )
                if layer is self.speculative_cache_layer:
                    speculative_view = delta, offsets, seq_lens, slots[:num_tokens]
        finally:
            table.slot_mapping.gpu[:num_tokens].copy_(original_slots)
        if self.speculative_cache_layer is not None:
            if speculative_view is None or common is None:
                raise RuntimeError("MTP2 per-layer attention metadata is incomplete")
            delta, offsets, _, slots = speculative_view
            common = self._speculative_common_metadata(
                common,
                delta,
                offsets,
                slots,
                num_reqs,
            )
        self._per_layer_metadata_seen = True
        return metadata, common

    @staticmethod
    def _speculative_common_metadata(
        common: Any,
        delta: list[int],
        offsets: tuple[int, ...],
        slots: torch.Tensor,
        num_reqs: int,
    ) -> Any:
        required = (
            "seq_lens",
            "_seq_lens_cpu",
            "seq_lens_cpu_upper_bound",
            "slot_mapping",
        )
        if any(not hasattr(common, name) for name in required):
            raise RuntimeError("MTP2 common attention metadata ABI changed")

        def adjusted(value: Any, corrections: list[int] | tuple[int, ...]) -> Any:
            if value is None:
                return None
            if not isinstance(value, torch.Tensor) or value.numel() < num_reqs:
                raise RuntimeError("MTP2 sequence-length metadata ABI changed")
            result = value.clone()
            if any(corrections):
                result[:num_reqs] += torch.tensor(
                    corrections,
                    dtype=result.dtype,
                    device=result.device,
                )
            return result

        changes = {
            "seq_lens": adjusted(common.seq_lens, delta),
            "_seq_lens_cpu": adjusted(common._seq_lens_cpu, delta),
            "seq_lens_cpu_upper_bound": adjusted(
                common.seq_lens_cpu_upper_bound,
                delta,
            ),
            "slot_mapping": slots,
        }
        if hasattr(common, "seq_lens_cpu"):
            changes["seq_lens_cpu"] = adjusted(common.seq_lens_cpu, delta)
        if hasattr(common, "num_computed_tokens_cpu"):
            changes["num_computed_tokens_cpu"] = adjusted(
                common.num_computed_tokens_cpu,
                tuple(-offset for offset in offsets),
            )
        try:
            return replace(common, **changes)
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                "MTP2 common attention metadata cannot be specialized by layer"
            ) from error

    def begin_per_layer_step(self, scheduler_output: Any) -> None:
        enabled = self.requires_per_layer_physical_state or self._has_per_layer_rows
        scheduled = getattr(scheduler_output, "num_scheduled_tokens", {})
        self._per_layer_step_requires_metadata = enabled and any(
            int(tokens) > 0 for tokens in scheduled.values()
        )
        if self._per_layer_step_requires_metadata:
            self._per_layer_metadata_seen = False

    def finish_per_layer_step(self) -> None:
        try:
            if (
                self._per_layer_step_requires_metadata
                and not self._per_layer_metadata_seen
            ):
                raise RuntimeError(
                    "per-layer attention metadata hook was bypassed by the host"
                )
        finally:
            self._per_layer_step_requires_metadata = False

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
        self._has_per_layer_rows = any(
            self.active[request_id].per_layer_physical_num_tokens is not None
            for request_id in self.runner.input_batch.req_ids[:num_reqs]
            if request_id in self.active
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


def _supports_mtp2(vllm_config: Any) -> bool:
    """Gate the unqualified MTP2 path behind explicit experimental opt-in.

    Compression is armed only at the exact prompt boundary; subsequent
    speculative decode steps cannot trigger another transaction because their
    optimistic KV tail can roll back after acceptance. The attention manager
    sees physical lengths, while Mamba state remains in semantic length space.
    """
    speculative = getattr(vllm_config, "speculative_config", None)
    if speculative is None:
        return False
    if os.getenv("VLLM_ASCEND_KVCOMPRESS_EXPERIMENTAL_MTP2") != "1":
        return False
    model = vllm_config.model_config
    text_config = getattr(model, "hf_text_config", None)
    return (
        getattr(speculative, "method", None) == "mtp"
        and getattr(speculative, "num_speculative_tokens", None) == 2
        and not getattr(speculative, "num_speculative_tokens_per_batch_size", None)
        and bool(getattr(model, "is_hybrid", False))
        and getattr(text_config, "model_type", None)
        in {"qwen3_5_text", "qwen3_5_moe_text"}
        and getattr(vllm_config.cache_config, "mamba_cache_mode", None) == "align"
    )


def _common_compatibility_reasons(vllm_config: Any, runner: Any) -> tuple[str, ...]:
    reasons: list[str] = []
    cache_config = vllm_config.cache_config
    # Prefix hits may share immutable source blocks with other requests. The
    # scheduler reserves disjoint destination blocks and swaps the requesting
    # row only after the worker has copied into them; its cache_blocks hook
    # also prevents private compacted KV from receiving semantic prefix hashes.
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
    if vllm_config.speculative_config is not None and not _supports_mtp2(vllm_config):
        reasons.append(
            "speculative decoding is unqualified; experimental Qwen3.5 MTP2 "
            "requires mamba_cache_mode='align' and explicit opt-in"
        )
    if vllm_config.kv_transfer_config is not None:
        reasons.append("KV transfer is unsupported")
    # The scheduler adapter freezes a request after arming its private-block
    # copy. It makes the block-table switch only after the accepted worker
    # output has acknowledged the transaction, so another in-flight batch
    # cannot decode from a half-written destination.
    model_config = vllm_config.model_config
    if bool(getattr(model_config, "is_hybrid", False)):
        text_config = getattr(model_config, "hf_text_config", None)
        model_type = str(getattr(text_config, "model_type", ""))
        if model_type not in {"qwen3_5_text", "qwen3_5_moe_text"}:
            reasons.append(
                "only Qwen3.5-style full-attention/Gated-DeltaNet hybrids are supported"
            )
        if getattr(cache_config, "mamba_cache_mode", "none") not in {
            "none",
            "align",
        }:
            reasons.append(
                "hybrid compression requires mamba_cache_mode='none' or 'align'"
            )
    if bool(getattr(model_config, "use_mla", False)):
        reasons.append("MLA cache layouts are unsupported")
    if bool(getattr(model_config, "is_encoder_decoder", False)):
        reasons.append("encoder-decoder models are unsupported")
    aux_output_config = getattr(vllm_config, "aux_output_config", None)
    if bool(getattr(aux_output_config, "enable_return_routed_experts", False)):
        reasons.append("routed-expert output is unsupported on this host ABI")
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
    promote the full-attention manager page to 1152 or 2048 tokens,
    representing one logical attention block as 9 or 16 consecutive kernel
    blocks. The scheduler's cross-group LCM alignment is validated separately.
    """
    model_config = getattr(vllm_config, "model_config", None)
    if not bool(getattr(model_config, "is_hybrid", False)):
        return frozenset({ASCEND_BLOCK_SIZE})
    text_config = getattr(model_config, "hf_text_config", None)
    model_type = str(getattr(text_config, "model_type", ""))
    if model_type in {"qwen3_5_text", "qwen3_5_moe_text"}:
        return frozenset({ASCEND_BLOCK_SIZE, 1152, 2048})
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
    inner_stride = (expected_tail[1] * expected_tail[2], expected_tail[2], 1)
    page_elements = expected_tail[0] * inner_stride[0]
    if cache.stride()[1:] != inner_stride or cache.stride(0) < page_elements:
        raise RuntimeError(
            f"attention layer {layer_name!r} {kind} cache must have contiguous "
            "inner blocks and a nonoverlapping page stride"
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
