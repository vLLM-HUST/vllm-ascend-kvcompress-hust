# SPDX-License-Identifier: Apache-2.0
"""Ascend Triton kernels used by the TriAttention hot paths."""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton


def shift_positions(
    positions: torch.Tensor,
    request_indices: torch.Tensor,
    request_offsets: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Subtract device-resident offsets using the fastest measured NPU ops."""
    torch.sub(
        positions,
        request_offsets.index_select(0, request_indices.to(torch.long)),
        out=output,
    )
    return output


@triton.jit
def _score_paged_keys_mean_kernel(
    k_cache_ptr,
    source_block_ids_ptr,
    q_real_ptr,
    q_imag_ptr,
    frequency_scale_ptr,
    extra_coefficient_ptr,
    omega_ptr,
    offset_cos_mean_ptr,
    offset_sin_mean_ptr,
    output_ptr,
    round_start,
    num_tokens,
    queries_per_kv,
    frequency_count,
    cache_block_stride,
    cache_token_stride,
    cache_head_stride,
    output_head_stride,
    BLOCK_SIZE: tl.constexpr,
    ROPE_STYLE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    query_head = tl.program_id(0)
    token_offsets = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    frequency_offsets = tl.arange(0, BLOCK_F)
    token_mask = token_offsets < num_tokens
    frequency_mask = frequency_offsets < frequency_count

    kv_head = query_head // queries_per_kv
    stat_base = query_head * frequency_count
    q_real = tl.load(
        q_real_ptr + stat_base + frequency_offsets,
        mask=frequency_mask,
        other=0.0,
    ).to(tl.float32)
    q_imag = tl.load(
        q_imag_ptr + stat_base + frequency_offsets,
        mask=frequency_mask,
        other=0.0,
    ).to(tl.float32)
    scale = tl.load(
        frequency_scale_ptr + stat_base + frequency_offsets,
        mask=frequency_mask,
        other=0.0,
    ).to(tl.float32)
    extra_coefficient = tl.load(
        extra_coefficient_ptr + stat_base + frequency_offsets,
        mask=frequency_mask,
        other=0.0,
    ).to(tl.float32)
    omega = tl.load(
        omega_ptr + frequency_offsets, mask=frequency_mask, other=0.0
    ).to(tl.float32)
    offset_cos = tl.load(
        offset_cos_mean_ptr + frequency_offsets,
        mask=frequency_mask,
        other=0.0,
    ).to(tl.float32)
    offset_sin = tl.load(
        offset_sin_mean_ptr + frequency_offsets,
        mask=frequency_mask,
        other=0.0,
    ).to(tl.float32)
    round_phase = round_start * omega
    round_cos = tl.cos(round_phase)
    round_sin = tl.sin(round_phase)
    cosine = round_cos * offset_cos - round_sin * offset_sin
    sine = round_sin * offset_cos + round_cos * offset_sin

    block_offsets = token_offsets // BLOCK_SIZE
    within_blocks = token_offsets - block_offsets * BLOCK_SIZE
    block_ids = tl.load(source_block_ids_ptr + block_offsets, mask=token_mask)
    cache_base = (
        block_ids[:, None] * cache_block_stride
        + within_blocks[:, None] * cache_token_stride
        + kv_head * cache_head_stride
    )
    if ROPE_STYLE == 0:
        real_dimensions = frequency_offsets[None, :] * 2
        imag_dimensions = real_dimensions + 1
    else:
        real_dimensions = frequency_offsets[None, :]
        imag_dimensions = real_dimensions + frequency_count
    matrix_mask = token_mask[:, None] & frequency_mask[None, :]
    key_real = tl.load(
        k_cache_ptr + cache_base + real_dimensions,
        mask=matrix_mask,
        other=0.0,
    ).to(tl.float32)
    key_imag = tl.load(
        k_cache_ptr + cache_base + imag_dimensions,
        mask=matrix_mask,
        other=0.0,
    ).to(tl.float32)
    product_real = q_real[None, :] * key_real + q_imag[None, :] * key_imag
    product_imag = q_imag[None, :] * key_real - q_real[None, :] * key_imag
    base_score = tl.sum(
        scale[None, :]
        * (product_real * cosine[None, :] - product_imag * sine[None, :]),
        axis=1,
    )
    key_abs = tl.sqrt(key_real * key_real + key_imag * key_imag)
    extra = tl.sum(
        key_abs * extra_coefficient[None, :] * scale[None, :], axis=1
    )
    tl.store(
        output_ptr + query_head * output_head_stride + token_offsets,
        base_score + extra,
        mask=token_mask,
    )


def score_paged_keys_mean(
    k_cache: torch.Tensor,
    source_block_ids: torch.Tensor,
    q_real: torch.Tensor,
    q_imag: torch.Tensor,
    frequency_scale: torch.Tensor,
    extra_coefficient: torch.Tensor,
    omega: torch.Tensor,
    offset_cos_mean: torch.Tensor,
    offset_sin_mean: torch.Tensor,
    round_start: int,
    num_tokens: int,
    output: torch.Tensor,
    block_size: int,
    rope_style: str,
) -> bool:
    """Score paged post-RoPE keys without first gathering them."""
    if k_cache.device.type != "npu":
        return False
    kv_heads, queries_per_kv, frequency_count = q_real.shape
    block_frequency = triton.next_power_of_2(frequency_count)
    _score_paged_keys_mean_kernel[
        (kv_heads * queries_per_kv, triton.cdiv(num_tokens, 16))
    ](
        k_cache,
        source_block_ids,
        q_real,
        q_imag,
        frequency_scale,
        extra_coefficient,
        omega,
        offset_cos_mean,
        offset_sin_mean,
        output,
        round_start,
        num_tokens,
        queries_per_kv,
        frequency_count,
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        output.stride(1),
        BLOCK_SIZE=block_size,
        ROPE_STYLE=0 if rope_style == "interleaved" else 1,
        BLOCK_N=16,
        BLOCK_F=block_frequency,
    )
    return True
