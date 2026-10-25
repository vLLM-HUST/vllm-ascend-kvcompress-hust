# SPDX-License-Identifier: Apache-2.0
"""Position-aware token selection policies for TriAttention."""

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import torch


class _V3BatchedPlan(NamedTuple):
    width: int
    max_quota: int
    sorted_topk: bool
    quotas: torch.Tensor
    ranks: torch.Tensor
    segment_offsets: torch.Tensor
    gather_indices: torch.Tensor | None
    valid_scores: torch.Tensor | None
    valid_positions: torch.Tensor
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
    """Cache small, shape-dependent index tensors across compression rounds."""
    middle_count = token_count - prefix - recent
    evict_total = token_count - budget
    if middle_count == 0:
        return None
    starts = [middle_count * segment // segments for segment in range(segments)]
    stops = [middle_count * (segment + 1) // segments for segment in range(segments)]
    widths = [stop - start for start, stop in zip(starts, stops, strict=True)]
    if min(widths) == 0:
        return None
    width = max(widths)
    quotas = [
        segment_width
        - evict_total * stop // middle_count
        + evict_total * start // middle_count
        for start, stop, segment_width in zip(starts, stops, widths, strict=True)
    ]
    if min(quotas) <= 0:
        return None
    max_quota = max(quotas)
    uneven = min(widths) != width
    gather_indices = None
    valid_scores = None
    if uneven:
        columns = torch.arange(width, device=device)[None, :]
        row_starts = torch.tensor(starts, device=device)[:, None]
        row_stops = torch.tensor(stops, device=device)[:, None]
        valid_scores = columns < torch.tensor(widths, device=device)[:, None]
        gather_indices = (
            torch.minimum(row_starts + columns, row_stops - 1) + prefix
        ).flatten()
    return _V3BatchedPlan(
        width=width,
        max_quota=max_quota,
        sorted_topk=min(quotas) != max_quota,
        quotas=torch.tensor(quotas, device=device)[:, None],
        ranks=torch.arange(max_quota, device=device)[None, :],
        segment_offsets=(torch.tensor(starts, device=device) + prefix)[:, None],
        gather_indices=gather_indices,
        valid_scores=valid_scores,
        valid_positions=torch.tensor(
            [
                segment * max_quota + rank
                for segment, quota in enumerate(quotas)
                for rank in range(quota)
            ],
            device=device,
        ),
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
    """Select middle-context segments with one batched top-k on Ascend.

    Zero-width and zero-quota cases retain the original path below. Uneven
    segments pad only their score rows, preserving proportional quota bounds.
    """
    plan = _v3_batched_plan(
        scores.device, scores.numel(), budget, prefix, recent, segments
    )
    if plan is None:
        return None
    if plan.gather_indices is None:
        rows = scores[prefix : scores.numel() - recent if recent else None].reshape(
            segments, plan.width
        )
    else:
        assert plan.valid_scores is not None
        rows = scores.index_select(0, plan.gather_indices).reshape(segments, plan.width)
        # Genuine -inf scores must rank ahead of padding, even when a row's
        # quota includes every genuine position.
        lowest = torch.finfo(rows.dtype).min
        rows = torch.where(
            plan.valid_scores,
            torch.nan_to_num(rows, nan=lowest, neginf=lowest),
            float("-inf"),
        )
    # Unequal quotas discard the extra returned rank for some rows, so these
    # rows require descending score order. Equal quotas may use the faster
    # unordered top-k because every returned index is retained.
    ranked = torch.topk(rows, k=plan.max_quota, dim=1, sorted=plan.sorted_topk).indices
    chosen = (
        torch.where(plan.ranks < plan.quotas, ranked, plan.width)
        if plan.sorted_topk
        else ranked
    )
    chosen = torch.sort(chosen, dim=1).values
    chosen += plan.segment_offsets
    middle = chosen.flatten()
    if plan.sorted_topk:
        middle = middle.index_select(0, plan.valid_positions)
    return torch.cat((plan.prefix_indices, middle, plan.recent_indices))


def select_keep_indices(
    scores: torch.Tensor,
    *,
    budget: int,
    protected_prefix: int,
    protected_recent: int,
    segments: int,
    policy: str,
) -> torch.Tensor:
    """Select sorted token positions while preserving structural anchors.

    ``global`` reproduces the original plugin policy. ``v3`` follows the
    TriAttention V3 policy: prefix and recent positions are hard protected and
    eviction is distributed proportionally over equal middle-context segments.
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

    prefix = min(protected_prefix, token_count)
    recent = min(protected_recent, token_count - prefix)
    if policy == "global":
        protected = torch.zeros(token_count, dtype=torch.bool, device=scores.device)
        if prefix:
            protected[:prefix] = True
        if recent:
            protected[token_count - recent :] = True
        ranked = scores.clone()
        ranked[protected] = float("inf")
        selected = torch.topk(ranked, k=budget, largest=True, sorted=False).indices
        return torch.sort(selected).values.contiguous()
    if policy != "v3":
        raise ValueError(f"unsupported position policy: {policy}")

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

    protected = torch.zeros(token_count, dtype=torch.bool, device=scores.device)
    if prefix:
        protected[:prefix] = True
    if recent:
        protected[token_count - recent :] = True

    # V3 assigns each middle segment a proportional eviction quota. Using
    # integer prefix sums makes the total exact and avoids a host-side cleanup
    # pass or O(n^2) membership checks.
    middle_start = prefix
    middle_stop = token_count - recent
    middle_count = middle_stop - middle_start
    evict_total = token_count - budget
    if evict_total > middle_count:
        raise ValueError("selection budget leaves too few slots for protected tokens")
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
    if int(selected.numel()) != budget:
        raise RuntimeError(
            f"V3 selected {selected.numel()} tokens, expected exactly {budget}"
        )
    return selected.contiguous()
