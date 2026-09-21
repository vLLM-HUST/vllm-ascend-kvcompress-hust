# SPDX-License-Identifier: Apache-2.0
"""Vectorized PyTorch-NPU scoring owned by the TriAttention method."""

from __future__ import annotations

import torch

from .stats import DeviceLayerCalibrationStats


def build_geometric_offsets(maximum: int, device: torch.device) -> torch.Tensor:
    """Return 1, 2, 4, ... offsets through ``maximum`` on ``device``."""
    if maximum < 1:
        raise ValueError("maximum scoring offset must be positive")
    offsets: list[float] = []
    value = 1
    while value <= maximum:
        offsets.append(float(value))
        value *= 2
    return torch.tensor(offsets, device=device, dtype=torch.float32)


def score_post_rope_keys(
    keys: torch.Tensor,
    stats: DeviceLayerCalibrationStats,
    *,
    round_start: int,
    offsets: torch.Tensor,
    aggregation: str,
) -> torch.Tensor:
    """Score post-RoPE keys without transferring device data to the host.

    Args:
        keys: Cache keys shaped ``[tokens, kv_heads, head_dim]``.
        stats: Query calibration statistics grouped by KV head.
        round_start: Absolute semantic position of the first future query.
        offsets: Future-query offsets sampled geometrically.
        aggregation: ``mean`` or ``max`` across future offsets.

    Returns:
        Scores shaped ``[kv_heads, query_heads_per_kv_head, tokens]``.
    """
    if keys.ndim != 3:
        raise ValueError("keys must have shape [tokens, kv_heads, head_dim]")
    token_count, kv_heads, head_dim = keys.shape
    if head_dim % 2:
        raise ValueError("key head dimension must be even")
    rotary_dim = head_dim if stats.rotary_dim is None else stats.rotary_dim
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError("calibration rotary dimension is incompatible with keys")
    freq_count = rotary_dim // 2
    if stats.q_mean_real.shape[0] != kv_heads:
        raise ValueError("calibration KV-head count does not match cache keys")
    if stats.q_mean_real.shape[-1] != freq_count:
        raise ValueError("calibration frequency count does not match cache keys")

    keys_fp32 = keys.to(dtype=torch.float32).permute(1, 0, 2)
    rotated_keys = keys_fp32[..., :rotary_dim]
    if stats.rope_style == "interleaved":
        key_real = rotated_keys[..., 0::2]
        key_imag = rotated_keys[..., 1::2]
    else:
        key_real = rotated_keys[..., :freq_count]
        key_imag = rotated_keys[..., freq_count:rotary_dim]

    q_real = stats.q_mean_real.unsqueeze(2)
    q_imag = stats.q_mean_imag.unsqueeze(2)
    key_real = key_real.unsqueeze(1)
    key_imag = key_imag.unsqueeze(1)
    product_real = q_real * key_real + q_imag * key_imag
    product_imag = q_imag * key_real - q_real * key_imag
    # Cached keys already contain one RoPE attention-scaling factor. Apply
    # only the matching query-side factor here. TriAttention's unrotated-key
    # path instead multiplies by the squared factor.
    frequency_scale = torch.sqrt(stats.freq_scale_sq).unsqueeze(2)

    phases = (offsets.unsqueeze(1) + float(round_start)) * stats.omega.unsqueeze(0)
    if aggregation == "mean":
        cosine = torch.cos(phases).mean(dim=0).view(1, 1, 1, freq_count)
        sine = torch.sin(phases).mean(dim=0).view(1, 1, 1, freq_count)
        scores = (frequency_scale * (product_real * cosine - product_imag * sine)).sum(
            dim=-1
        )
    elif aggregation == "max":
        scores = torch.full(
            (kv_heads, stats.q_mean_real.shape[1], token_count),
            float("-inf"),
            dtype=torch.float32,
            device=keys.device,
        )
        for phase in phases.unbind(0):
            cosine = torch.cos(phase).view(1, 1, 1, freq_count)
            sine = torch.sin(phase).view(1, 1, 1, freq_count)
            candidate = (
                frequency_scale * (product_real * cosine - product_imag * sine)
            ).sum(dim=-1)
            scores = torch.maximum(scores, candidate)
    else:
        raise ValueError(f"unsupported score aggregation: {aggregation}")

    key_abs = torch.sqrt(key_real.square() + key_imag.square())
    q_mean_abs = torch.sqrt(
        stats.q_mean_real.square() + stats.q_mean_imag.square() + 1e-8
    )
    extra_coefficient = (stats.q_abs_mean - q_mean_abs).unsqueeze(2)
    extra = (key_abs * extra_coefficient * frequency_scale).sum(dim=-1)
    scores = scores + extra

    # Partial-RoPE models such as Qwen3.5 leave most key dimensions unrotated.
    # Their direct calibrated Q·K contribution is position independent and must
    # be included; a phase-only score is otherwise blind to 75% of each key.
    if rotary_dim < head_dim:
        if stats.q_pass_mean is None:
            raise ValueError("partial-RoPE scoring requires q_pass_mean")
        if stats.q_pass_mean.shape[-1] != head_dim - rotary_dim:
            raise ValueError("q_pass_mean width does not match unrotated keys")
        key_pass = keys_fp32[..., rotary_dim:].unsqueeze(1)
        content_score = (stats.q_pass_mean.unsqueeze(2) * key_pass).sum(dim=-1)
        scores = scores + content_score
    return scores


def normalize_head_scores(
    scores: torch.Tensor, mean: torch.Tensor, variance: torch.Tensor
) -> torch.Tensor:
    """Normalize each calibrated query head along the token dimension."""
    denominator = torch.sqrt(variance.clamp_min(1e-6)).unsqueeze(-1)
    return (scores - mean.unsqueeze(-1)) / denominator
