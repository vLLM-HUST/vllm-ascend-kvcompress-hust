# SPDX-License-Identifier: Apache-2.0
"""V@O integration with the synchronous Ascend compression lifecycle."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ...config import JsonScalar
from ..base import (
    CompressionRequest,
    CompressionResult,
    KVCompressionMethod,
    LayerCache,
    MethodRuntimeSpec,
    ModelShape,
)
from ..triattention.cache import token_slots
from .config import VATOConfig
from .observation import OutputObserver
from .scoring import score_paged_values, select_keep_indices


class VATOMethod(KVCompressionMethod):
    def __init__(
        self,
        options: Mapping[str, JsonScalar],
        vllm_config: Any,
        model_shape: ModelShape,
    ) -> None:
        self.config = VATOConfig.from_method_config(options)
        self.vllm_config = vllm_config
        self.model_shape = model_shape
        self.layer_caches: tuple[LayerCache, ...] = ()
        self.observer: OutputObserver | None = None
        self._hooks: list[Any] = []

    @property
    def name(self) -> str:
        return "vato"

    @property
    def runtime_spec(self) -> MethodRuntimeSpec:
        return self.config.runtime_spec

    def compatibility_reasons(self, worker: Any) -> tuple[str, ...]:
        reasons = []
        if not bool(getattr(self.vllm_config.model_config, "enforce_eager", False)):
            reasons.append("V@O output observation requires --enforce-eager")
        parallel = self.vllm_config.parallel_config
        if bool(getattr(parallel, "enable_dbo", False)):
            reasons.append("V@O output observation does not support DBO microbatching")
        shape = self.model_shape
        if shape.num_kv_heads <= 0 or shape.num_attention_heads % shape.num_kv_heads:
            reasons.append("V@O requires query heads divisible by KV heads")
        return tuple(reasons)

    def bind_model_runner(
        self, runner: Any, layer_caches: tuple[LayerCache, ...]
    ) -> None:
        reasons = self.compatibility_reasons(runner)
        if reasons:
            raise RuntimeError("; ".join(reasons))
        if {layer.layer_index for layer in layer_caches} != set(
            self.model_shape.full_attention_layer_indices
        ) or len(layer_caches) != len(self.model_shape.full_attention_layer_indices):
            raise RuntimeError(
                "V@O requires all full-attention cache layers exactly once"
            )
        tp = int(self.vllm_config.parallel_config.tensor_parallel_size)
        if tp <= 0 or self.model_shape.num_kv_heads % tp:
            raise RuntimeError("V@O tensor parallel size must divide KV heads")
        context = runner.compilation_config.static_forward_context
        modules = [context[layer.name] for layer in layer_caches]
        if any(
            not callable(getattr(module, "register_forward_hook", None))
            for module in modules
        ):
            raise RuntimeError("V@O requires hookable full-attention modules")
        for layer in layer_caches:
            if layer.k_cache.shape != layer.v_cache.shape or layer.k_cache.shape[
                2:
            ] != (self.model_shape.num_kv_heads // tp, self.model_shape.head_dim):
                raise RuntimeError("V@O cache shape does not match local KV heads")
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self.observer = OutputObserver(
            runner,
            window_size=self.config.window_size,
            num_kv_heads=self.model_shape.num_kv_heads // tp,
            num_query_heads=self.model_shape.num_attention_heads // tp,
            head_dim=self.model_shape.head_dim,
        )
        for layer, module in zip(layer_caches, modules, strict=True):
            self._hooks.append(
                module.register_forward_hook(self.observer.hook(layer.name))
            )
        self.layer_caches = layer_caches

    def reset_requests(self, request_ids: set[str]) -> None:
        if self.observer is not None:
            self.observer.reset_requests(request_ids)

    @torch.no_grad()
    def compress(self, request: CompressionRequest) -> CompressionResult:
        if self.observer is None or not self.layer_caches:
            raise RuntimeError("V@O method is not bound to KV cache")
        # Validate every observation before any in-place cache writes.
        observations = [
            self.observer.mean(
                layer.name, request.request_id, request.semantic_num_tokens
            )
            for layer in self.layer_caches
        ]
        budget = min(self.config.kv_budget, request.physical_num_tokens)
        dense = torch.arange(budget, device=request.destination_block_ids_device.device)
        for layer, observation in zip(self.layer_caches, observations, strict=True):
            scores = score_paged_values(
                layer.v_cache,
                request.source_block_ids_device,
                observation,
                num_tokens=request.physical_num_tokens,
                chunk_size=self.config.score_chunk_size,
                variant=self.config.variant,
            )
            keep = select_keep_indices(
                scores,
                budget=budget,
                sink_size=self.config.sink_size,
                window_size=self.config.window_size,
                kernel_size=self.config.kernel_size,
            )
            block_size = layer.k_cache.shape[1]
            source = (
                token_slots(request.source_block_ids_device, keep.flatten(), block_size)
                .reshape(keep.shape)
                .T
            )
            destination = token_slots(
                request.destination_block_ids_device, dense, block_size
            )
            heads = torch.arange(layer.k_cache.shape[2], device=source.device)
            flat_k = layer.k_cache.view(
                -1, layer.k_cache.shape[2], layer.k_cache.shape[3]
            )
            flat_v = layer.v_cache.view_as(flat_k)
            # Each head owns different token indices. Gather both tensors before
            # writing because the provider reuses the first source blocks.
            keys = flat_k[source, heads]
            values = flat_v[source, heads]
            flat_k.index_copy_(0, destination, keys)
            flat_v.index_copy_(0, destination, values)
        return CompressionResult(
            budget, tuple((layer.name, budget) for layer in self.layer_caches)
        )
