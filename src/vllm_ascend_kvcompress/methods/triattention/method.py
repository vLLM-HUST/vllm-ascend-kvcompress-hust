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
from .cache import gather_paged_range, materialize_selected_tokens
from .config import TriAttentionConfig
from .scoring import (
    build_geometric_offsets,
    normalize_head_scores,
    score_post_rope_keys,
)
from .stats import CalibrationStats, DeviceLayerCalibrationStats

METHOD_NAME = "triattention"


@dataclass(frozen=True)
class TriAttentionLayerCache:
    name: str
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    stats: DeviceLayerCalibrationStats


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
        self.offsets: torch.Tensor | None = None

    @property
    def name(self) -> str:
        return METHOD_NAME

    @property
    def runtime_spec(self) -> MethodRuntimeSpec:
        return MethodRuntimeSpec(
            requires_private_destination=True,
            compression_threshold_tokens=self.config.compression_threshold_tokens,
            required_recompute_tokens=self.config.recompute_window,
            max_physical_num_tokens=self.config.kv_budget,
        )

    def compatibility_reasons(self, worker: Any) -> tuple[str, ...]:
        del worker
        return self.calibration.validate(self.model_shape)

    def bind_model_runner(
        self,
        runner: Any,
        layer_caches: tuple[LayerCache, ...],
    ) -> None:
        device_stats = self.calibration.to_device(self.model_shape, runner.device)
        if len(layer_caches) != self.model_shape.num_layers:
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
            bound.append(
                TriAttentionLayerCache(
                    name=layer.name,
                    k_cache=layer.k_cache,
                    v_cache=layer.v_cache,
                    stats=device_stats[layer.layer_index],
                )
            )
        self.layer_caches = tuple(bound)
        self.offsets = build_geometric_offsets(
            int(self.vllm_config.model_config.max_model_len), runner.device
        )

    def compress(self, request: CompressionRequest) -> CompressionResult:
        if self.offsets is None or not self.layer_caches:
            raise RuntimeError("TriAttention method is not bound to KV cache")
        keep_indices = self._select_keep_indices(
            request.source_block_ids_device,
            request.semantic_num_tokens,
        )
        for layer in self.layer_caches:
            materialize_selected_tokens(
                layer.k_cache,
                layer.v_cache,
                request.source_block_ids_device,
                request.destination_block_ids_device,
                keep_indices,
                ASCEND_BLOCK_SIZE,
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
        semantic_num_tokens: int,
    ) -> torch.Tensor:
        assert self.offsets is not None
        aggregated_scores: torch.Tensor | None = None
        layer_count = len(self.layer_caches)
        for layer in self.layer_caches:
            head_sum = torch.zeros(
                layer.stats.q_mean_real.shape[:2],
                dtype=torch.float32,
                device=source_block_ids.device,
            )
            head_square_sum = torch.zeros_like(head_sum)
            score_chunks: list[torch.Tensor] = []
            for start in range(0, semantic_num_tokens, self.config.score_chunk_size):
                count = min(self.config.score_chunk_size, semantic_num_tokens - start)
                keys = gather_paged_range(
                    layer.k_cache,
                    source_block_ids,
                    start=start,
                    count=count,
                    block_size=ASCEND_BLOCK_SIZE,
                )
                raw_scores = score_post_rope_keys(
                    keys,
                    layer.stats,
                    round_start=semantic_num_tokens,
                    offsets=self.offsets,
                    aggregation=self.config.score_aggregation,
                )
                score_chunks.append(raw_scores)
                head_sum.add_(raw_scores.sum(dim=-1))
                head_square_sum.add_(raw_scores.square().sum(dim=-1))
            head_mean = head_sum / float(semantic_num_tokens)
            head_variance = (
                head_square_sum / float(semantic_num_tokens) - head_mean.square()
            ).clamp_min_(0.0)

            start = 0
            for raw_scores in score_chunks:
                count = raw_scores.shape[-1]
                normalized = normalize_head_scores(raw_scores, head_mean, head_variance)
                layer_scores = normalized.amax(dim=(0, 1))
                if aggregated_scores is None:
                    fill = (
                        0.0
                        if self.config.layer_aggregation == "mean"
                        else float("-inf")
                    )
                    aggregated_scores = torch.full(
                        (semantic_num_tokens,),
                        fill,
                        dtype=torch.float32,
                        device=source_block_ids.device,
                    )
                target = aggregated_scores[start : start + count]
                if self.config.layer_aggregation == "mean":
                    target.add_(layer_scores)
                else:
                    torch.maximum(target, layer_scores, out=target)
                start += count

        if aggregated_scores is None:
            raise RuntimeError("no layer scores were produced")
        if self.config.layer_aggregation == "mean":
            aggregated_scores.div_(float(layer_count))
        aggregated_scores.nan_to_num_(
            nan=float("-inf"), posinf=1e30, neginf=float("-inf")
        )
        protected = min(self.config.protected_recent_window, semantic_num_tokens)
        if protected:
            aggregated_scores[semantic_num_tokens - protected :] = float("inf")
        selected = torch.topk(
            aggregated_scores,
            k=self.config.kv_budget,
            largest=True,
            sorted=False,
        ).indices
        return torch.sort(selected).values.contiguous()


def create_triattention_method(
    options: Mapping[str, JsonScalar],
    vllm_config: Any,
    model_shape: ModelShape,
) -> KVCompressionMethod:
    return TriAttentionMethod(options, vllm_config, model_shape)


# Backward-compatible import name from the pre-registry layout.
TriAttentionAscendConfig = TriAttentionConfig
