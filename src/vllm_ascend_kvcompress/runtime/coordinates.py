# SPDX-License-Identifier: Apache-2.0
"""换算管理器逻辑页与压缩后的 token 坐标。"""

from __future__ import annotations

import torch


# 向上取整计算所需逻辑页数；block_size 由调用者指定所属缓存组。
# 不要把 scheduler 的跨组 LCM 对齐当作 full-attention manager 页大小。
def blocks_for_tokens(num_tokens: int, block_size: int) -> int:
    return (num_tokens + block_size - 1) // block_size


def expand_scheduler_block_ids(
    block_ids: tuple[int, ...],
    cache_blocks_per_scheduler_block: int,
    device: torch.device | str,
) -> torch.Tensor:
    """将调度器块 ID 展开为连续的 128-token 内核缓存块 ID。"""
    # 若一个逻辑页含 k 个内核页，则逻辑页 b 展开为 b*k 到 b*k+k-1。
    # 展开保持原页表顺序；不同逻辑页之间不要求连续。
    ids = torch.as_tensor(block_ids, device=device, dtype=torch.long)
    if cache_blocks_per_scheduler_block == 1:
        return ids
    offsets = torch.arange(
        cache_blocks_per_scheduler_block,
        device=device,
        dtype=torch.long,
    )
    return (ids.unsqueeze(1) * cache_blocks_per_scheduler_block + offsets).reshape(-1)


def shift_positions(
    positions: torch.Tensor,
    request_indices: torch.Tensor,
    request_offsets: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """使用已测量选定的快速 NPU 算子减去设备端偏移。"""
    # request_indices 把每个 token 映射到当前批次的请求行。
    # 先按行取累计删除量，再写入独立 output，保留原始语义 positions 不变。
    torch.sub(
        positions,
        request_offsets.index_select(0, request_indices.to(torch.long)),
        out=output,
    )
    return output
