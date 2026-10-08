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


def _gather_slots(
    cache: torch.Tensor, slots: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    block_size = cache.shape[1]
    if cache.is_contiguous():
        flat = cache.view(-1, cache.shape[2], cache.shape[3])
        if out is not None:
            return torch.index_select(flat, 0, slots, out=out)
        return flat.index_select(0, slots)
    blocks = torch.div(slots, block_size, rounding_mode="floor")
    offsets = torch.remainder(slots, block_size)
    gathered = cache[blocks, offsets]
    if out is not None:
        out.copy_(gathered)
        return out
    return gathered


def _write_slots(
    cache: torch.Tensor, slots: torch.Tensor, values: torch.Tensor
) -> None:
    block_size = cache.shape[1]
    if cache.is_contiguous():
        cache.view(-1, cache.shape[2], cache.shape[3]).index_copy_(0, slots, values)
        return
    blocks = torch.div(slots, block_size, rounding_mode="floor")
    offsets = torch.remainder(slots, block_size)
    cache[blocks, offsets] = values


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
    return _gather_slots(cache, slots)


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
    gathered_k = _gather_slots(k_cache, source_slots)
    gathered_v = _gather_slots(v_cache, source_slots)

    dense_indices = torch.arange(
        keep_indices.shape[0], device=keep_indices.device, dtype=torch.long
    )
    destination_slots = token_slots(destination_block_ids, dense_indices, block_size)
    _write_slots(k_cache, destination_slots, gathered_k)
    _write_slots(v_cache, destination_slots, gathered_v)


def materialize_token_slots(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    source_slots: torch.Tensor,
    destination_slots: torch.Tensor,
    k_workspace: torch.Tensor,
    v_workspace: torch.Tensor,
) -> None:
    """Move precomputed slots through persistent, overlap-safe workspaces."""
    _gather_slots(k_cache, source_slots, out=k_workspace)
    _gather_slots(v_cache, source_slots, out=v_workspace)
    _write_slots(k_cache, destination_slots, k_workspace)
    _write_slots(v_cache, destination_slots, v_workspace)
