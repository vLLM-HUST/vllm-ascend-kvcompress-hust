# SPDX-License-Identifier: Apache-2.0
"""Stable contracts implemented by Ascend KV compression methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from ..transaction import CompressionPlan


@dataclass(frozen=True)
class ModelShape:
    """Method-facing, framework-neutral transformer shape information."""

    model_type: str
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    rope_theta: float
    has_rope_scaling: bool
    rotary_dim: int | None = None
    attention_layer_indices: tuple[int, ...] | None = None

    @property
    def effective_rotary_dim(self) -> int:
        return self.head_dim if self.rotary_dim is None else self.rotary_dim

    @property
    def full_attention_layer_indices(self) -> tuple[int, ...]:
        if self.attention_layer_indices is None:
            return tuple(range(self.num_layers))
        return self.attention_layer_indices

    @property
    def is_hybrid(self) -> bool:
        return len(self.full_attention_layer_indices) != self.num_layers


@dataclass(frozen=True)
class MethodRuntimeSpec:
    """Scheduling limits advertised by a compression method."""

    requires_private_destination: bool
    compression_threshold_tokens: int
    required_recompute_tokens: int
    max_physical_num_tokens: int
    min_output_tokens_for_compression: int = 0


@dataclass(frozen=True)
class LayerCache:
    """One validated full-attention K/V cache binding."""

    name: str
    layer_index: int
    k_cache: torch.Tensor
    v_cache: torch.Tensor


@dataclass(frozen=True)
class CompressionRequest:
    """Method-facing view of one stateful compression transaction."""

    request_id: str
    semantic_num_tokens: int
    physical_num_tokens: int
    source_block_ids: tuple[tuple[int, ...], ...]
    destination_block_ids: tuple[tuple[int, ...], ...]
    source_block_ids_device: torch.Tensor
    destination_block_ids_device: torch.Tensor
    plan: CompressionPlan | None = None


@dataclass(frozen=True)
class CompressionResult:
    """Physical state produced by a compression method."""

    physical_num_tokens: int
    per_layer_physical_num_tokens: tuple[tuple[str, int], ...] | None = None


@dataclass(frozen=True)
class QueryBatchSpan:
    """Query rows for one request in the current model-forward batch."""

    request_id: str
    start: int
    end: int


@dataclass(frozen=True)
class QueryObservation:
    """Identity of a completed final-prefill compression transaction.

    Query tensors remain in the method's own buffers, populated by
    ``capture_query``. ``plan`` is the exact scheduler-authorized transaction
    subsequently passed to ``compress``.
    """

    request_id: str
    plan: CompressionPlan
    semantic_num_tokens: int
    window_tokens: int
    layer_indices: tuple[int, ...]


class KVCompressionMethod(ABC):
    """Algorithm boundary beneath the common vLLM/Ascend lifecycle adapter."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the stable configuration name for this method."""

    @property
    @abstractmethod
    def runtime_spec(self) -> MethodRuntimeSpec:
        """Return scheduler-visible limits before KV allocation."""

    @abstractmethod
    def compatibility_reasons(self, worker: Any) -> tuple[str, ...]:
        """Return method-specific incompatibilities without side effects."""

    @abstractmethod
    def bind_model_runner(
        self,
        runner: Any,
        layer_caches: tuple[LayerCache, ...],
    ) -> None:
        """Bind method-specific runtime state after KV allocation."""

    @abstractmethod
    def compress(self, request: CompressionRequest) -> CompressionResult:
        """Materialize one compression transaction into destination blocks."""

    @property
    def query_window_tokens(self) -> int:
        """Trailing prefill query rows required per full-attention layer.

        Zero is the default and installs no observation hooks. A method that
        opts in owns its capture buffers and must override the hooks below.
        """
        return 0

    def capture_query(
        self,
        layer: LayerCache,
        query: torch.Tensor,
        spans: tuple[QueryBatchSpan, ...],
    ) -> None:
        """Copy prefill query rows into method-owned, address-stable buffers."""
        return None

    def complete_query_observation(self, observation: QueryObservation) -> None:
        """Publish buffered queries only after a successful full forward."""
        return None

    def discard_query_observation(self, request_id: str) -> None:
        """Release one request's uncommitted observation on reset or commit."""
        return None
