# SPDX-License-Identifier: Apache-2.0
"""TriAttention 分页缓存读取与所选 token 的实际写入。"""

from __future__ import annotations

import torch


# block_ids 是该请求的 cache 页表，而非全局连续页区间。
# 逻辑 token i → block_ids[i // block_size] * block_size + i % block_size。
# 这里的 slot 是线性 token 槽编号，实际张量元素地址还由 cache stride 决定。
def token_slots(
    block_ids: torch.Tensor, token_indices: torch.Tensor, block_size: int
) -> torch.Tensor:
    """将逻辑 token 索引映射为展平后的物理缓存槽位。"""
    block_offsets = torch.div(token_indices, block_size, rounding_mode="floor")
    within_block = torch.remainder(token_indices, block_size)
    return block_ids.index_select(0, block_offsets) * block_size + within_block


def _gather_slots(
    cache: torch.Tensor, slots: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    block_size = cache.shape[1]
    # 只有完整连续 cache 才能把页和 token 维直接 view 成线性槽。
    # 页间有 padding 时按 (blocks,offsets) 索引，不能把空洞当成有效 token。
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
    """从 [blocks, block, heads, dim] 布局的缓存中收集 token。"""
    _validate_cache_layout(cache, block_size)
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
    """收集连续的逻辑区间，避免设备到主机的数据传输。"""
    _validate_cache_layout(cache, block_size)
    return _gather_paged_range(
        cache, block_ids, start=start, count=count, block_size=block_size
    )


def _validate_cache_layout(cache: torch.Tensor, block_size: int) -> None:
    """独立缓存读取入口的布局约束，运行时绑定后无需按评分分块重复验证。"""
    if cache.ndim != 4 or cache.shape[1] != block_size:
        raise ValueError("cache must use [blocks, block, heads, dim] layout")


def _gather_paged_range(
    cache: torch.Tensor,
    block_ids: torch.Tensor,
    *,
    start: int,
    count: int,
    block_size: int,
) -> torch.Tensor:
    """从已验证布局的缓存读取连续区间，供绑定后的评分循环使用。"""
    indices = torch.arange(
        start,
        start + count,
        device=block_ids.device,
        dtype=torch.long,
    )
    return _gather_slots(cache, token_slots(block_ids, indices, block_size))


# 这是按索引直接搬运的路径，会临时分配 gather 结果；
# 常规 method 使用下面的持久 workspace 版本。
def materialize_selected_tokens(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    source_block_ids: torch.Tensor,
    destination_block_ids: torch.Tensor,
    keep_indices: torch.Tensor,
    block_size: int,
) -> None:
    """先收集 K 和 V，再将所选 token 写入稠密目标区域。"""
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
    """通过持久工作区搬运预计算槽位，保证源与目标重叠时也安全。"""
    # 先完成 K、V 两份 gather，随后才写，避免潜在重叠覆盖尚未读取的源值。
    # 当前 provider 另有更强约束：事务使用与源块不相交的私有目的块。
    _gather_slots(k_cache, source_slots, out=k_workspace)
    _gather_slots(v_cache, source_slots, out=v_workspace)
    _write_slots(k_cache, destination_slots, k_workspace)
    _write_slots(v_cache, destination_slots, v_workspace)
