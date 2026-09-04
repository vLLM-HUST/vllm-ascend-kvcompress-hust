# SPDX-License-Identifier: Apache-2.0
"""Ascend Triton kernels used by the TriAttention hot paths."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except (ImportError, ModuleNotFoundError):
    # Static manifest discovery and CPU-only validation must not import the
    # accelerator toolchain. NPU calls below still fail closed.
    class _UnavailableTriton:
        @staticmethod
        def jit(function=None, **kwargs):
            del kwargs

            def decorate(candidate):
                return candidate

            return decorate(function) if function is not None else decorate

    class _UnavailableLanguage:
        constexpr = object()

    triton = _UnavailableTriton()  # type: ignore[assignment]
    tl = _UnavailableLanguage()  # type: ignore[assignment]


def _triton_available() -> bool:
    return hasattr(triton, "cdiv") and hasattr(triton, "next_power_of_2")


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


@triton.jit(do_not_specialize=["round_start", "num_tokens"])
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
    omega = tl.load(omega_ptr + frequency_offsets, mask=frequency_mask, other=0.0).to(
        tl.float32
    )
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
    extra = tl.sum(key_abs * extra_coefficient[None, :] * scale[None, :], axis=1)
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
    if not _triton_available():
        raise RuntimeError("triton-ascend is unavailable or failed to import")
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


@triton.jit(do_not_specialize=["num_tokens"])
def _aggregate_normalized_scores_kernel(
    scores_ptr,
    mean_ptr,
    variance_ptr,
    aggregate_ptr,
    num_tokens,
    score_head_stride,
    num_heads: tl.constexpr,
    MODE: tl.constexpr,
    FIRST_LAYER: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token_offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    head_offsets = tl.arange(0, BLOCK_H)
    token_mask = token_offsets < num_tokens
    head_mask = head_offsets < num_heads
    scores = tl.load(
        scores_ptr + head_offsets[None, :] * score_head_stride + token_offsets[:, None],
        mask=token_mask[:, None] & head_mask[None, :],
        other=float("-inf"),
    ).to(tl.float32)
    mean = tl.load(mean_ptr + head_offsets, mask=head_mask, other=0.0).to(tl.float32)
    variance = tl.load(variance_ptr + head_offsets, mask=head_mask, other=0.0).to(
        tl.float32
    )
    normalized = (scores - mean[None, :]) * tl.rsqrt(variance[None, :] + 1e-6)
    layer_score = tl.max(normalized, axis=1)
    if FIRST_LAYER:
        result = layer_score
    else:
        previous = tl.load(aggregate_ptr + token_offsets, mask=token_mask, other=0.0)
        if MODE == 0:
            result = previous + layer_score
        else:
            result = tl.maximum(previous, layer_score)
    tl.store(aggregate_ptr + token_offsets, result, mask=token_mask)


def aggregate_normalized_scores(
    scores: torch.Tensor,
    mean: torch.Tensor,
    variance: torch.Tensor,
    aggregate: torch.Tensor,
    num_tokens: int,
    *,
    layer_aggregation: str,
    first_layer: bool,
) -> bool:
    """Fuse normalization, query-head max, and cross-layer accumulation."""
    if scores.device.type != "npu":
        return False
    if not _triton_available():
        raise RuntimeError("triton-ascend is unavailable or failed to import")
    num_heads = mean.numel()
    block_heads = triton.next_power_of_2(num_heads)
    _aggregate_normalized_scores_kernel[(triton.cdiv(num_tokens, 128),)](
        scores,
        mean,
        variance,
        aggregate,
        num_tokens,
        scores.stride(-2),
        num_heads=num_heads,
        MODE=0 if layer_aggregation == "mean" else 1,
        FIRST_LAYER=first_layer,
        BLOCK_N=128,
        BLOCK_H=block_heads,
    )
    return True
