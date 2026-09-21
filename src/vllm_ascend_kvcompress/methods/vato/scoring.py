# SPDX-License-Identifier: Apache-2.0
"""Paged V@O scoring, adapted from TriAttention methods/v_at_o.py.

Reference snapshot: abdf5d7145e7a286b4a1ff387e4c2f348e080fa4 (Apache-2.0).
Scores and selection are independent for each KV head.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..triattention.cache import gather_paged_range


def score_paged_values(
    cache: torch.Tensor,
    block_ids: torch.Tensor,
    observation: torch.Tensor,
    *,
    num_tokens: int,
    chunk_size: int,
    variant: str,
) -> torch.Tensor:
    """Score ``[blocks, block, heads, dim]`` V with bounded gather memory.

    The reference's ``cosine`` divides only by the value norm. Its
    ``centered_norm`` preserves signed scores; it does not clamp them to zero.
    Centering is over the entire sequence, including protected tokens.
    """
    if variant not in {"dot", "abs", "cosine", "centered", "centered_norm"}:
        raise ValueError(f"unsupported V@O variant {variant!r}")
    if num_tokens <= 0 or chunk_size <= 0:
        raise ValueError("V@O scoring requires positive token and chunk sizes")
    scores = torch.empty(
        cache.shape[2], num_tokens, device=cache.device, dtype=torch.float32
    )
    norms = torch.empty_like(scores) if variant == "centered_norm" else None
    for start in range(0, num_tokens, chunk_size):
        count = min(chunk_size, num_tokens - start)
        values = gather_paged_range(
            cache,
            block_ids,
            start=start,
            count=count,
            block_size=cache.shape[1],
        ).float()
        dot = (values * observation.unsqueeze(0)).sum(dim=-1).T
        if variant == "abs":
            dot = dot.abs()
        if variant in {"cosine", "centered_norm"}:
            norm = values.norm(dim=-1).clamp_min(1e-8).T
            if norms is not None:
                norms[:, start : start + count] = norm
            else:
                dot = dot / norm
        scores[:, start : start + count] = dot
    if variant in {"centered", "centered_norm"}:
        scores.sub_(scores.mean(dim=-1, keepdim=True))
    if norms is not None:
        scores.mul_(norms)
    return scores


def select_keep_indices(
    scores: torch.Tensor,
    *,
    budget: int,
    sink_size: int,
    window_size: int,
    kernel_size: int,
) -> torch.Tensor:
    """Return sorted ``[heads, budget]`` indices with sink/recent protection."""
    heads, length = scores.shape
    if length <= budget:
        return torch.arange(length, device=scores.device).expand(heads, -1)
    middle = scores[:, sink_size : length - window_size]
    if kernel_size > 1:
        middle = F.max_pool1d(
            middle.unsqueeze(0), kernel_size, stride=1, padding=kernel_size // 2
        ).squeeze(0)
    middle = middle.nan_to_num(nan=float("-inf"), posinf=1e30, neginf=float("-inf"))
    keep = middle.topk(budget - sink_size - window_size, dim=-1).indices + sink_size
    return torch.cat(
        [
            torch.arange(sink_size, device=scores.device).expand(heads, -1),
            keep.sort(dim=-1).values,
            torch.arange(length - window_size, length, device=scores.device).expand(
                heads, -1
            ),
        ],
        dim=-1,
    )
