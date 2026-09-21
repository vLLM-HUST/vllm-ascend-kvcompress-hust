# SPDX-License-Identifier: Apache-2.0
"""Position-aware token selection policies for TriAttention."""

from __future__ import annotations

import torch


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
    protected = torch.zeros(token_count, dtype=torch.bool, device=scores.device)
    if prefix:
        protected[:prefix] = True
    if recent:
        protected[token_count - recent :] = True

    if policy == "global":
        ranked = scores.clone()
        ranked[protected] = float("inf")
        selected = torch.topk(ranked, k=budget, largest=True, sorted=False).indices
        return torch.sort(selected).values.contiguous()
    if policy != "v3":
        raise ValueError(f"unsupported position policy: {policy}")

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
