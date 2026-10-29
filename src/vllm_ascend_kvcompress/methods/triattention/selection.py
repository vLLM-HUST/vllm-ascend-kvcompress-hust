# SPDX-License-Identifier: Apache-2.0
"""TriAttention 中考虑位置结构的 token 选择策略。"""

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import torch


# 缓存的是按形状/设备确定的配额与索引模板，不是某请求选出的 token 或评分。
# lru_cache 最多保留 64 个模板，减少多轮相同长度的设备小张量构建。
class _V3BatchedPlan(NamedTuple):
    width: int
    max_quota: int
    sorted_topk: bool
    quotas: torch.Tensor
    ranks: torch.Tensor
    segment_offsets: torch.Tensor
    target_offsets: torch.Tensor
    prefix_indices: torch.Tensor
    recent_indices: torch.Tensor


@lru_cache(maxsize=64)
def _v3_batched_plan(
    device: torch.device,
    token_count: int,
    budget: int,
    prefix: int,
    recent: int,
    segments: int,
) -> _V3BatchedPlan | None:
    """跨压缩轮次缓存依赖形状的小型索引张量。"""
    middle_count = token_count - prefix - recent
    # 每段先分配“删除数”而不是直接平均保留数；整数前缀差分保证总删除数精确。
    # 只有中间区域可等宽分段且各段至少保留一个 token 时采用 batched top-k。
    evict_total = token_count - budget
    if middle_count == 0 or middle_count % segments:
        return None
    width = middle_count // segments
    # 利用相邻整数累计删除量的差，确定每段保留数。
    # 这比独立四舍五入各段配额更容易保证总数恰好等于 budget。
    quotas = [
        width
        - evict_total * ((segment + 1) * width) // middle_count
        + evict_total * (segment * width) // middle_count
        for segment in range(segments)
    ]
    if min(quotas) <= 0:
        return None
    max_quota = max(quotas)
    return _V3BatchedPlan(
        width=width,
        max_quota=max_quota,
        sorted_topk=min(quotas) != max_quota,
        quotas=torch.tensor(quotas, device=device)[:, None],
        ranks=torch.arange(max_quota, device=device)[None, :],
        segment_offsets=(torch.arange(segments, device=device) * width + prefix)[
            :, None
        ],
        target_offsets=torch.tensor(
            [prefix + sum(quotas[:segment]) for segment in range(segments)],
            device=device,
        )[:, None],
        prefix_indices=torch.arange(prefix, device=device),
        recent_indices=torch.arange(token_count - recent, token_count, device=device),
    )


def _select_v3_batched(
    scores: torch.Tensor,
    *,
    budget: int,
    prefix: int,
    recent: int,
    segments: int,
) -> torch.Tensor | None:
    """在 Ascend 上通过一次批量 top-k 处理等宽片段。

    宽度不同或配额为零的场景仍走下方原有路径。此快速路径保持相同
    的整数配额边界，并返回有序输出。
    """
    plan = _v3_batched_plan(
        scores.device, scores.numel(), budget, prefix, recent, segments
    )
    if plan is None:
        return None
    rows = scores[prefix : scores.numel() - recent if recent else None].reshape(
        segments, plan.width
    )
    # 配额不同时，部分行需丢弃额外返回的排名，
    # 因此必须按分数降序排列；配额相同时会保留所有返回索引，
    # 可使用更快的无序 top-k。
    ranked = torch.topk(rows, k=plan.max_quota, dim=1, sorted=plan.sorted_topk).indices
    chosen = torch.where(plan.ranks < plan.quotas, ranked, plan.width)
    chosen = torch.sort(chosen, dim=1).values
    chosen += plan.segment_offsets
    targets = torch.where(plan.ranks < plan.quotas, plan.ranks + plan.target_offsets, 0)
    result = torch.empty(budget, device=scores.device, dtype=torch.long)
    # target_offsets 把各段选中位置拼成密集输出，prefix/recent 占固定首尾区。
    # 不足配额的占位项都写到零号槽，下面必须补回零号槽的真实值。
    result.scatter_(0, targets.flatten(), chosen.flatten())
    # 多余的 top-k 条目均写入槽位零，因此 scatter 后须恢复其真实值；
    # 这也覆盖了合法的无前缀场景。
    if prefix:
        result[:prefix] = plan.prefix_indices
    else:
        result[0] = chosen[0, 0]
    if recent:
        result[-recent:] = plan.recent_indices
    return result.contiguous()


def select_keep_indices(
    scores: torch.Tensor,
    *,
    budget: int,
    protected_prefix: int,
    protected_recent: int,
    segments: int,
    policy: str,
) -> torch.Tensor:
    """在保留结构锚点的前提下，选择并排序 token 位置。

    global 复现插件原始策略。v3 使用 TriAttention V3 策略：
    硬性保护前缀与近期位置，并按比例将淘汰量分配到等宽的中间片段。
    """
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    token_count = int(scores.numel())
    if budget <= 0 or budget > token_count:
        raise ValueError("budget must be in [1, token_count]")
    if protected_prefix < 0 or protected_recent < 0:
        raise ValueError("protected windows must be non-negative")
    if protected_prefix + protected_recent > budget:
        raise ValueError("protected windows cannot exceed the selection budget")
    if segments <= 0:
        raise ValueError("segments must be positive")
    if policy not in ("global", "v3"):
        raise ValueError(f"unsupported position policy: {policy}")
    return _select_keep_indices(
        scores,
        budget=budget,
        protected_prefix=protected_prefix,
        protected_recent=protected_recent,
        segments=segments,
        policy=policy,
    )


def _select_keep_indices(
    scores: torch.Tensor,
    *,
    budget: int,
    protected_prefix: int,
    protected_recent: int,
    segments: int,
    policy: str,
) -> torch.Tensor:
    """对已验证的评分与选择参数执行计算，供公开入口和绑定后的 method 复用。

    method 的一维评分来自固定工作区，配置已验证窗口和策略，
    provider 仅在物理长度达到压缩阈值后调用，因此预算不会超过 token 数。
    """
    token_count = int(scores.numel())

    prefix = min(protected_prefix, token_count)
    recent = min(protected_recent, token_count - prefix)
    protected = torch.zeros(token_count, dtype=torch.bool, device=scores.device)
    if prefix:
        protected[:prefix] = True
    if recent:
        protected[token_count - recent :] = True

    # 两种策略都输出按原物理位置升序排列的恰好 budget 个索引。
    # 升序保证保留 token 的原有时间顺序；top-k 的分数排序只用于决定入选集合。
    if policy == "global":
        ranked = scores.clone()
        ranked[protected] = float("inf")
        selected = torch.topk(ranked, k=budget, largest=True, sorted=False).indices
        return torch.sort(selected).values.contiguous()

    # NPU 批量路径仅优化配额和 top-k 的执行；不满足形状条件则回落下面逐段实现。
    # 分数相同的候选仍遵循底层 top-k 的 tie 行为，不能据“配额相同”推断索引逐项相同。
    if scores.device.type == "npu":
        batched = _select_v3_batched(
            scores,
            budget=budget,
            prefix=prefix,
            recent=recent,
            segments=segments,
        )
        if batched is not None:
            return batched

    # V3 按比例为各个中间片段分配淘汰配额。
    # 利用整数前缀和保证总量精确，
    # 避免额外的主机端收尾步骤或 O(n^2) 成员检查。
    middle_start = prefix
    middle_stop = token_count - recent
    middle_count = middle_stop - middle_start
    evict_total = token_count - budget
    # 入口已保证保护区总长不超过预算，因此淘汰数不会超过中间区长度。
    keep_mask = protected.clone()
    for segment in range(segments):
        start = middle_start + middle_count * segment // segments
        stop = middle_start + middle_count * (segment + 1) // segments
        segment_count = stop - start
        if segment_count == 0:
            continue
        evict_before = evict_total * (start - middle_start) // middle_count
        evict_after = evict_total * (stop - middle_start) // middle_count
        keep_count = segment_count - (evict_after - evict_before)
        if keep_count <= 0:
            continue
        if keep_count == segment_count:
            keep_mask[start:stop] = True
            continue
        local = torch.topk(
            scores[start:stop], k=keep_count, largest=True, sorted=False
        ).indices
        keep_mask[start + local] = True

    selected = torch.nonzero(keep_mask, as_tuple=False).flatten()
    return selected.contiguous()
