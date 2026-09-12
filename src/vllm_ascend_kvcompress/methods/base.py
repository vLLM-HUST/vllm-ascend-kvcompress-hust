# SPDX-License-Identifier: Apache-2.0
"""Stable contracts implemented by Ascend KV compression methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch


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


@dataclass(frozen=True)
class CompressionResult:
    """Physical state produced by a compression method."""

    physical_num_tokens: int
    per_layer_physical_num_tokens: tuple[tuple[str, int], ...] | None = None


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
