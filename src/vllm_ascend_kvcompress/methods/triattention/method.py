# SPDX-License-Identifier: Apache-2.0
"""TriAttention token-selection method for Ascend KV caches."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from ...config import ASCEND_BLOCK_SIZE, JsonScalar
from ..base import (
    CompressionRequest,
    CompressionResult,
    KVCompressionMethod,
    LayerCache,
    MethodRuntimeSpec,
    ModelShape,
)
from .cache import gather_paged_range, materialize_token_slots, token_slots
from .config import TriAttentionConfig
from .kernels import (
    aggregate_normalized_scores,
    prepare_mean_phase_coefficients,
    score_paged_keys_mean_precomputed,
)
from .scoring import (
    build_geometric_offsets,
    normalize_head_scores,
    score_post_rope_keys,
)
from .selection import select_keep_indices
from .stats import CalibrationStats, DeviceLayerCalibrationStats

METHOD_NAME = "triattention"


def _queries_per_kv_head(model_shape: ModelShape) -> int:
    if model_shape.num_attention_heads % model_shape.num_kv_heads:
        raise RuntimeError("TriAttention requires query heads divisible by KV heads")
    return model_shape.num_attention_heads // model_shape.num_kv_heads


@dataclass(frozen=True)
class TriAttentionLayerCache:
    name: str
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    stats: DeviceLayerCalibrationStats
    frequency_scale: torch.Tensor | None = None
    extra_coefficient: torch.Tensor | None = None
    offset_cos_mean: torch.Tensor | None = None
    offset_sin_mean: torch.Tensor | None = None


class TriAttentionMethod(KVCompressionMethod):
    """Correctness-first TriAttention implementation behind the method API."""

    def __init__(
        self,
        options: Mapping[str, JsonScalar],
        vllm_config: Any,
        model_shape: ModelShape,
    ) -> None:
        self.config = TriAttentionConfig.from_method_config(options)
        self.vllm_config = vllm_config
        self.model_shape = model_shape
        self.calibration = CalibrationStats.load(self.config.stats_path)
        self.layer_caches: tuple[TriAttentionLayerCache, ...] = ()
        self.cache_block_size = ASCEND_BLOCK_SIZE
        self.offsets: torch.Tensor | None = None
        self.k_workspace: torch.Tensor | None = None
        self.v_workspace: torch.Tensor | None = None
        self.score_workspace: torch.Tensor | None = None
        self.aggregate_workspace: torch.Tensor | None = None
        self.dense_indices: torch.Tensor | None = None
        self.scoring_omega: torch.Tensor | None = None
        self.scoring_offset_cos: torch.Tensor | None = None
        self.scoring_offset_sin: torch.Tensor | None = None
        self.phase_cos_workspace: torch.Tensor | None = None
        self.phase_sin_workspace: torch.Tensor | None = None

    @property
    def name(self) -> str:
        return METHOD_NAME

    @property
    def runtime_spec(self) -> MethodRuntimeSpec:
        return self.config.runtime_spec

    def compatibility_reasons(self, worker: Any) -> tuple[str, ...]:
        del worker
        return self.calibration.validate(self.model_shape)

    def bind_model_runner(
        self,
        runner: Any,
        layer_caches: tuple[LayerCache, ...],
    ) -> None:
        parallel = self.vllm_config.parallel_config
        self.tensor_parallel_size = int(parallel.tensor_parallel_size)
        tensor_parallel_rank = 0
        if self.tensor_parallel_size > 1:
            from vllm.distributed import get_tp_group

            tensor_parallel_rank = int(get_tp_group().rank_in_group)
        device_stats = self.calibration.to_device(
            self.model_shape,
            runner.device,
            tensor_parallel_rank=tensor_parallel_rank,
            tensor_parallel_size=self.tensor_parallel_size,
        )
        self.offsets = build_geometric_offsets(
            int(self.vllm_config.model_config.max_model_len), runner.device
        )
        if len(layer_caches) != len(self.model_shape.full_attention_layer_indices):
            raise RuntimeError(
                "allocated full-attention layer count does not match calibration"
            )
        bound: list[TriAttentionLayerCache] = []
        seen_layer_indices: set[int] = set()
        for layer in layer_caches:
            if layer.layer_index not in device_stats:
                raise RuntimeError(
                    f"attention layer {layer.name!r} resolves to invalid model "
                    f"layer index {layer.layer_index}"
                )
            if layer.layer_index in seen_layer_indices:
                raise RuntimeError(
                    "multiple cache layers resolve to model layer index "
                    f"{layer.layer_index}"
                )
            seen_layer_indices.add(layer.layer_index)
            stats = device_stats[layer.layer_index]
            frequency_scale = torch.sqrt(stats.freq_scale_sq)
            q_mean_abs = torch.sqrt(
                stats.q_mean_real.square() + stats.q_mean_imag.square() + 1e-8
            )
            offset_phases = self.offsets.unsqueeze(1) * stats.omega.unsqueeze(0)
            bound.append(
                TriAttentionLayerCache(
                    name=layer.name,
                    k_cache=layer.k_cache,
                    v_cache=layer.v_cache,
                    stats=stats,
                    frequency_scale=frequency_scale,
                    extra_coefficient=stats.q_abs_mean - q_mean_abs,
                    offset_cos_mean=torch.cos(offset_phases).mean(dim=0),
                    offset_sin_mean=torch.sin(offset_phases).mean(dim=0),
                )
            )
        self.layer_caches = tuple(bound)
        scoring_layers = self.layer_caches[:: self.config.score_layer_stride]
        if any(
            layer.offset_cos_mean is None or layer.offset_sin_mean is None
            for layer in scoring_layers
        ):
            raise RuntimeError("TriAttention phase coefficients are not initialized")
        self.scoring_omega = torch.stack(
            [layer.stats.omega for layer in scoring_layers]
        ).contiguous()
        self.scoring_offset_cos = torch.stack(
            [layer.offset_cos_mean for layer in scoring_layers]
        ).contiguous()
        self.scoring_offset_sin = torch.stack(
            [layer.offset_sin_mean for layer in scoring_layers]
        ).contiguous()
        self.phase_cos_workspace = torch.empty_like(self.scoring_omega)
        self.phase_sin_workspace = torch.empty_like(self.scoring_omega)
        cache = self.layer_caches[0].k_cache
        self.cache_block_size = int(cache.shape[1])
        if self.cache_block_size != ASCEND_BLOCK_SIZE:
            raise RuntimeError(
                "TriAttention requires Ascend's 128-token kernel cache blocks"
            )
        workspace_shape = (
            self.config.kv_budget,
            cache.shape[2],
            cache.shape[3],
        )
        self.k_workspace = torch.empty(
            workspace_shape, dtype=cache.dtype, device=runner.device
        )
        self.v_workspace = torch.empty_like(self.k_workspace)
        expected_local_kv_heads = (
            self.model_shape.num_kv_heads // self.tensor_parallel_size
        )
        if cache.shape[2] != expected_local_kv_heads:
            raise RuntimeError(
                "local KV-cache head count does not match tensor-parallel model "
                f"shape: expected {expected_local_kv_heads}, got {cache.shape[2]}"
            )
        # GQA group width is a model-global ratio and is unchanged by tensor
        # parallel sharding. Dividing global query heads by the local cache
        # head count would incorrectly double the workspace width at TP=2.
        queries_per_kv = _queries_per_kv_head(self.model_shape)
        self.score_workspace = torch.empty(
            (
                cache.shape[2],
                queries_per_kv,
                int(self.vllm_config.model_config.max_model_len),
            ),
            dtype=torch.float32,
            device=runner.device,
        )
        self.aggregate_workspace = torch.empty(
            int(self.vllm_config.model_config.max_model_len),
            dtype=torch.float32,
            device=runner.device,
        )
        self.dense_indices = torch.arange(
            self.config.kv_budget, device=runner.device, dtype=torch.long
        )

    def compress(self, request: CompressionRequest) -> CompressionResult:
        if (
            self.offsets is None
            or not self.layer_caches
            or self.k_workspace is None
            or self.v_workspace is None
            or self.score_workspace is None
            or self.dense_indices is None
        ):
            raise RuntimeError("TriAttention method is not bound to KV cache")
        keep_indices = self._select_keep_indices(
            request.source_block_ids_device,
            request.physical_num_tokens,
            request.semantic_num_tokens,
        )
        cache_block_size = getattr(self, "cache_block_size", ASCEND_BLOCK_SIZE)
        source_slots = token_slots(
            request.source_block_ids_device, keep_indices, cache_block_size
        )
        destination_slots = token_slots(
            request.destination_block_ids_device,
            self.dense_indices,
            cache_block_size,
        )
        for layer in self.layer_caches:
            materialize_token_slots(
                layer.k_cache,
                layer.v_cache,
                source_slots,
                destination_slots,
                self.k_workspace,
                self.v_workspace,
            )
        layer_lengths = tuple(
            (layer.name, self.config.kv_budget) for layer in self.layer_caches
        )
        return CompressionResult(
            physical_num_tokens=self.config.kv_budget,
            per_layer_physical_num_tokens=layer_lengths,
        )

    def _select_keep_indices(
        self,
        source_block_ids: torch.Tensor,
        physical_num_tokens: int,
        round_start: int,
    ) -> torch.Tensor:
        assert self.offsets is not None
        if self.aggregate_workspace is None or self.score_workspace is None:
            raise RuntimeError("TriAttention score workspaces are not initialized")
        aggregated_scores = self.aggregate_workspace[:physical_num_tokens]
        cache_block_size = getattr(self, "cache_block_size", ASCEND_BLOCK_SIZE)
        tensor_parallel_size = getattr(self, "tensor_parallel_size", 1)
        # Scoring every model layer serializes dozens of small reductions in the
        # scheduler hot path. Uniformly sample calibrated layers while still
        # materializing the selected KV tokens for every cache layer.
        scoring_layers = self.layer_caches[:: self.config.score_layer_stride]
        layer_count = len(scoring_layers)
        phase_inputs = (
            getattr(self, "scoring_omega", None),
            getattr(self, "scoring_offset_cos", None),
            getattr(self, "scoring_offset_sin", None),
            getattr(self, "phase_cos_workspace", None),
            getattr(self, "phase_sin_workspace", None),
        )
        phases_prepared = all(value is not None for value in phase_inputs) and (
            prepare_mean_phase_coefficients(
                self.scoring_omega,
                self.scoring_offset_cos,
                self.scoring_offset_sin,
                round_start,
                self.phase_cos_workspace,
                self.phase_sin_workspace,
            )
        )
        for layer_index, layer in enumerate(scoring_layers):
            score_workspace = self.score_workspace
            kernel_inputs = (
                layer.frequency_scale,
                layer.extra_coefficient,
                layer.offset_cos_mean,
                layer.offset_sin_mean,
            )
            if (
                self.config.score_aggregation == "mean"
                and phases_prepared
                and all(value is not None for value in kernel_inputs)
                and (
                    getattr(layer.stats, "rotary_dim", None) is None
                    or getattr(layer.stats, "rotary_dim", None)
                    == layer.k_cache.shape[-1]
                )
            ):
                raw_scores = score_workspace[:, :, :physical_num_tokens]
                used_kernel = score_paged_keys_mean_precomputed(
                    layer.k_cache,
                    source_block_ids,
                    layer.stats.q_mean_real,
                    layer.stats.q_mean_imag,
                    layer.frequency_scale,
                    layer.extra_coefficient,
                    self.phase_cos_workspace[layer_index],
                    self.phase_sin_workspace[layer_index],
                    physical_num_tokens,
                    raw_scores,
                    cache_block_size,
                    layer.stats.rope_style,
                )
            else:
                used_kernel = False
            if used_kernel:
                head_variance, head_mean = torch.var_mean(
                    raw_scores, dim=-1, correction=0
                )
                if tensor_parallel_size != 1 or not aggregate_normalized_scores(
                    raw_scores,
                    head_mean,
                    head_variance,
                    aggregated_scores,
                    physical_num_tokens,
                    layer_aggregation=self.config.layer_aggregation,
                    first_layer=layer_index == 0,
                ):
                    self._aggregate_torch(
                        raw_scores,
                        head_mean,
                        head_variance,
                        aggregated_scores,
                        layer_index,
                    )
                continue
            raw_scores = score_workspace[:, :, :physical_num_tokens]
            for start in range(0, physical_num_tokens, self.config.score_chunk_size):
                count = min(self.config.score_chunk_size, physical_num_tokens - start)
                keys = gather_paged_range(
                    layer.k_cache,
                    source_block_ids,
                    start=start,
                    count=count,
                    block_size=cache_block_size,
                )
                chunk_scores = score_post_rope_keys(
                    keys,
                    layer.stats,
                    round_start=round_start,
                    offsets=self.offsets,
                    aggregation=self.config.score_aggregation,
                )
                raw_scores[:, :, start : start + count].copy_(chunk_scores)
            head_variance, head_mean = torch.var_mean(raw_scores, dim=-1, correction=0)
            if tensor_parallel_size != 1 or not aggregate_normalized_scores(
                raw_scores,
                head_mean,
                head_variance,
                aggregated_scores,
                physical_num_tokens,
                layer_aggregation=self.config.layer_aggregation,
                first_layer=layer_index == 0,
            ):
                self._aggregate_torch(
                    raw_scores,
                    head_mean,
                    head_variance,
                    aggregated_scores,
                    layer_index,
                )

        if self.config.layer_aggregation == "mean":
            aggregated_scores.div_(float(layer_count))
        aggregated_scores.nan_to_num_(
            nan=float("-inf"), posinf=1e30, neginf=float("-inf")
        )
        return select_keep_indices(
            aggregated_scores,
            budget=self.config.kv_budget,
            protected_prefix=self.config.protected_prefix_window,
            protected_recent=self.config.protected_recent_window,
            segments=self.config.position_segments,
            policy=self.config.position_policy,
        )

    def _aggregate_torch(
        self,
        raw_scores: torch.Tensor,
        head_mean: torch.Tensor,
        head_variance: torch.Tensor,
        aggregated_scores: torch.Tensor,
        layer_index: int,
    ) -> None:
        normalized = normalize_head_scores(raw_scores, head_mean, head_variance)
        layer_scores = normalized.amax(dim=(0, 1))
        if getattr(self, "tensor_parallel_size", 1) > 1:
            import torch.distributed as dist
            from vllm.distributed import get_tp_group

            dist.all_reduce(
                layer_scores,
                op=dist.ReduceOp.MAX,
                group=get_tp_group().device_group,
            )
        if layer_index == 0:
            aggregated_scores.copy_(layer_scores)
        elif self.config.layer_aggregation == "mean":
            aggregated_scores.add_(layer_scores)
        else:
            torch.maximum(aggregated_scores, layer_scores, out=aggregated_scores)


def create_triattention_method(
    options: Mapping[str, JsonScalar],
    vllm_config: Any,
    model_shape: ModelShape,
) -> KVCompressionMethod:
    return TriAttentionMethod(options, vllm_config, model_shape)


# Backward-compatible import name from the pre-registry layout.
TriAttentionAscendConfig = TriAttentionConfig
