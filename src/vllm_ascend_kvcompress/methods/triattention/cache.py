# SPDX-License-Identifier: Apache-2.0
"""TriAttention paged-cache gathers and selected-token materialization."""

from __future__ import annotations

import torch


def token_slots(
    block_ids: torch.Tensor, token_indices: torch.Tensor, block_size: int
) -> torch.Tensor:
    """Map logical token indices to flattened physical cache slots."""
    block_offsets = torch.div(token_indices, block_size, rounding_mode="floor")
    within_block = torch.remainder(token_indices, block_size)
    return block_ids.index_select(0, block_offsets) * block_size + within_block


def gather_paged_tokens(
    cache: torch.Tensor,
    block_ids: torch.Tensor,
    token_indices: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Gather tokens from ``[blocks, block, heads, dim]`` cache storage."""
    if cache.ndim != 4 or cache.shape[1] != block_size:
        raise ValueError("cache must use [blocks, block, heads, dim] layout")
    slots = token_slots(block_ids, token_indices, block_size)
    flattened = cache.view(-1, cache.shape[2], cache.shape[3])
    return flattened.index_select(0, slots)


def gather_paged_range(
    cache: torch.Tensor,
    block_ids: torch.Tensor,
    *,
    start: int,
    count: int,
    block_size: int,
) -> torch.Tensor:
    """Gather one contiguous logical range without a device-to-host transfer."""
    indices = torch.arange(
        start,
        start + count,
        device=block_ids.device,
        dtype=torch.long,
    )
    return gather_paged_tokens(cache, block_ids, indices, block_size)


def materialize_selected_tokens(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    source_block_ids: torch.Tensor,
    destination_block_ids: torch.Tensor,
    keep_indices: torch.Tensor,
    block_size: int,
) -> None:
    """Gather K and V before writing selected tokens into dense destinations."""
    source_slots = token_slots(source_block_ids, keep_indices, block_size)
    flat_k = k_cache.view(-1, k_cache.shape[2], k_cache.shape[3])
    flat_v = v_cache.view(-1, v_cache.shape[2], v_cache.shape[3])
    gathered_k = flat_k.index_select(0, source_slots)
    gathered_v = flat_v.index_select(0, source_slots)

    dense_indices = torch.arange(
        keep_indices.shape[0], device=keep_indices.device, dtype=torch.long
    )
    destination_slots = token_slots(destination_block_ids, dense_indices, block_size)
    flat_k.index_copy_(0, destination_slots, gathered_k)
    flat_v.index_copy_(0, destination_slots, gathered_v)
