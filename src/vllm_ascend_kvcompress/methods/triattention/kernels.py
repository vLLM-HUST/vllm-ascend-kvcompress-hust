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


@triton.jit
def _map_token_slots_kernel(
    block_ids_ptr,
    token_indices_ptr,
    output_ptr,
    num_tokens,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < num_tokens
    indices = tl.load(token_indices_ptr + offsets, mask=mask, other=0)
    block_offsets = indices // BLOCK_SIZE
    block_ids = tl.load(block_ids_ptr + block_offsets, mask=mask, other=0)
    tl.store(
        output_ptr + offsets,
        block_ids * BLOCK_SIZE + indices % BLOCK_SIZE,
        mask=mask,
    )


def map_token_slots(
    block_ids: torch.Tensor,
    token_indices: torch.Tensor,
    block_size: int,
    output: torch.Tensor,
) -> bool:
    """Map logical positions to physical slots in one Ascend launch."""
    if token_indices.device.type != "npu":
        return False
    if not _triton_available():
        raise RuntimeError("triton-ascend is unavailable or failed to import")
    if (
        block_ids.device != token_indices.device
        or output.device != token_indices.device
    ):
        raise ValueError("slot mapping tensors must be on the same device")
    if output.shape != token_indices.shape or output.dtype != torch.long:
        raise ValueError("slot output must match token indices and use int64")
    _map_token_slots_kernel[(triton.cdiv(token_indices.numel(), 256),)](
        block_ids,
        token_indices,
        output,
        token_indices.numel(),
        BLOCK_SIZE=block_size,
        BLOCK_N=256,
    )
    return True


@triton.jit
def _copy_private_kv_slots_kernel(
    k_ptr,
    v_ptr,
    source_slots_ptr,
    destination_slots_ptr,
    num_tokens,
    K_PAGE_STRIDE: tl.constexpr,
    V_PAGE_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tokens = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    dims = tl.arange(0, HEAD_DIM)
    token_mask = tokens < num_tokens
    source = tl.load(source_slots_ptr + tokens, mask=token_mask, other=0)
    destination = tl.load(destination_slots_ptr + tokens, mask=token_mask, other=0)
    if K_PAGE_STRIDE == BLOCK_SIZE * HEAD_DIM:
        k_source = source[:, None] * HEAD_DIM + dims[None, :]
        k_destination = destination[:, None] * HEAD_DIM + dims[None, :]
    else:
        source_block = source // BLOCK_SIZE
        source_token = source % BLOCK_SIZE
        destination_block = destination // BLOCK_SIZE
        destination_token = destination % BLOCK_SIZE
        k_source = (
            source_block[:, None] * K_PAGE_STRIDE
            + source_token[:, None] * HEAD_DIM
            + dims[None, :]
        )
        k_destination = (
            destination_block[:, None] * K_PAGE_STRIDE
            + destination_token[:, None] * HEAD_DIM
            + dims[None, :]
        )
    if V_PAGE_STRIDE == BLOCK_SIZE * HEAD_DIM:
        v_source = source[:, None] * HEAD_DIM + dims[None, :]
        v_destination = destination[:, None] * HEAD_DIM + dims[None, :]
    else:
        source_block = source // BLOCK_SIZE
        source_token = source % BLOCK_SIZE
        destination_block = destination // BLOCK_SIZE
        destination_token = destination % BLOCK_SIZE
        v_source = (
            source_block[:, None] * V_PAGE_STRIDE
            + source_token[:, None] * HEAD_DIM
            + dims[None, :]
        )
        v_destination = (
            destination_block[:, None] * V_PAGE_STRIDE
            + destination_token[:, None] * HEAD_DIM
            + dims[None, :]
        )
    key = tl.load(k_ptr + k_source, mask=token_mask[:, None], other=0)
    value = tl.load(v_ptr + v_source, mask=token_mask[:, None], other=0)
    tl.store(k_ptr + k_destination, key, mask=token_mask[:, None])
    tl.store(v_ptr + v_destination, value, mask=token_mask[:, None])


def copy_private_kv_slots(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    source_slots: torch.Tensor,
    destination_slots: torch.Tensor,
) -> bool:
    """Copy K/V in one launch when source and destination blocks are disjoint."""
    if k_cache.device.type != "npu":
        return False
    if not _triton_available():
        raise RuntimeError("triton-ascend is unavailable or failed to import")
    if k_cache.shape != v_cache.shape or k_cache.ndim != 4:
        raise ValueError("K/V caches must have matching four-dimensional shapes")
    if k_cache.device != v_cache.device or source_slots.device != k_cache.device:
        raise ValueError("K/V copy tensors must be on the same device")
    if destination_slots.device != k_cache.device:
        raise ValueError("K/V copy tensors must be on the same device")
    if source_slots.shape != destination_slots.shape:
        raise ValueError("source and destination slots must have matching shapes")
    head_dim = k_cache.shape[3]
    if not (
        k_cache.shape[2] == 1
        and head_dim == triton.next_power_of_2(head_dim)
        and k_cache.stride()[1:] == (head_dim, head_dim, 1)
        and v_cache.stride()[1:] == (head_dim, head_dim, 1)
        and k_cache.dtype == v_cache.dtype
    ):
        return False
    _copy_private_kv_slots_kernel[(triton.cdiv(source_slots.numel(), 4),)](
        k_cache,
        v_cache,
        source_slots,
        destination_slots,
        source_slots.numel(),
        K_PAGE_STRIDE=k_cache.stride(0),
        V_PAGE_STRIDE=v_cache.stride(0),
        BLOCK_SIZE=k_cache.shape[1],
        HEAD_DIM=head_dim,
        BLOCK_N=4,
    )
    return True


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


@triton.jit(do_not_specialize=["round_start", "num_values"])
def _prepare_mean_phase_kernel(
    omega_ptr,
    offset_cos_mean_ptr,
    offset_sin_mean_ptr,
    phase_cos_ptr,
    phase_sin_ptr,
    round_start,
    num_values,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < num_values
    omega = tl.load(omega_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    offset_cos = tl.load(offset_cos_mean_ptr + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    offset_sin = tl.load(offset_sin_mean_ptr + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    phase = round_start * omega
    round_cos = tl.cos(phase)
    round_sin = tl.sin(phase)
    tl.store(
        phase_cos_ptr + offsets,
        round_cos * offset_cos - round_sin * offset_sin,
        mask=mask,
    )
    tl.store(
        phase_sin_ptr + offsets,
        round_sin * offset_cos + round_cos * offset_sin,
        mask=mask,
    )


def prepare_mean_phase_coefficients(
    omega: torch.Tensor,
    offset_cos_mean: torch.Tensor,
    offset_sin_mean: torch.Tensor,
    round_start: int,
    phase_cos: torch.Tensor,
    phase_sin: torch.Tensor,
) -> bool:
    """Prepare one mean-offset RoPE coefficient per layer and frequency.

    The scoring grid has one program for every query head and token tile. Doing
    trigonometry there repeats identical work thousands of times. This compact
    kernel computes it once for each sampled layer/frequency pair instead.
    """
    if omega.device.type != "npu":
        return False
    if not _triton_available():
        raise RuntimeError("triton-ascend is unavailable or failed to import")
    if not (
        omega.shape
        == offset_cos_mean.shape
        == offset_sin_mean.shape
        == phase_cos.shape
        == phase_sin.shape
    ):
        raise ValueError("phase coefficient tensors must have matching shapes")
    num_values = omega.numel()
    _prepare_mean_phase_kernel[(triton.cdiv(num_values, 256),)](
        omega,
        offset_cos_mean,
        offset_sin_mean,
        phase_cos,
        phase_sin,
        round_start,
        num_values,
        BLOCK=256,
    )
    return True


@triton.jit(do_not_specialize=["num_tokens"])
def _score_paged_keys_mean_kernel(
    k_cache_ptr,
    source_block_ids_ptr,
    q_real_ptr,
    q_imag_ptr,
    q_pass_ptr,
    frequency_scale_ptr,
    extra_coefficient_ptr,
    phase_cos_ptr,
    phase_sin_ptr,
    output_ptr,
    num_tokens,
    queries_per_kv,
    frequency_count,
    cache_block_stride,
    cache_token_stride,
    cache_head_stride,
    output_head_stride,
    BLOCK_SIZE: tl.constexpr,
    ROPE_STYLE: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    PASS_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_F: tl.constexpr,
    BLOCK_P: tl.constexpr,
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
    cosine = tl.load(
        phase_cos_ptr + frequency_offsets,
        mask=frequency_mask,
        other=0.0,
    ).to(tl.float32)
    sine = tl.load(
        phase_sin_ptr + frequency_offsets,
        mask=frequency_mask,
        other=0.0,
    ).to(tl.float32)

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
    score = base_score + extra
    if PASS_DIM > 0:
        pass_offsets = tl.arange(0, BLOCK_P)
        pass_mask = pass_offsets < PASS_DIM
        q_pass = tl.load(
            q_pass_ptr + query_head * PASS_DIM + pass_offsets,
            mask=pass_mask,
            other=0.0,
        ).to(tl.float32)
        key_pass = tl.load(
            k_cache_ptr + cache_base + ROTARY_DIM + pass_offsets[None, :],
            mask=token_mask[:, None] & pass_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        score += tl.sum(key_pass * q_pass[None, :], axis=1)
    tl.store(
        output_ptr + query_head * output_head_stride + token_offsets,
        score,
        mask=token_mask,
    )


def score_paged_keys_mean_precomputed(
    k_cache: torch.Tensor,
    source_block_ids: torch.Tensor,
    q_real: torch.Tensor,
    q_imag: torch.Tensor,
    frequency_scale: torch.Tensor,
    extra_coefficient: torch.Tensor,
    phase_cos: torch.Tensor,
    phase_sin: torch.Tensor,
    num_tokens: int,
    output: torch.Tensor,
    block_size: int,
    rope_style: str,
    *,
    q_pass_mean: torch.Tensor | None = None,
    rotary_dim: int | None = None,
) -> bool:
    """Score full- or partial-RoPE paged keys with precomputed query phases."""
    if k_cache.device.type != "npu":
        return False
    if not _triton_available():
        raise RuntimeError("triton-ascend is unavailable or failed to import")
    kv_heads, queries_per_kv, frequency_count = q_real.shape
    effective_rotary_dim = 2 * frequency_count if rotary_dim is None else rotary_dim
    head_dim = k_cache.shape[-1]
    if effective_rotary_dim != 2 * frequency_count or not (
        0 < effective_rotary_dim <= head_dim
    ):
        raise ValueError("rotary_dim must match the calibrated frequency count")
    pass_dim = head_dim - effective_rotary_dim
    if pass_dim:
        if q_pass_mean is None or q_pass_mean.shape != (
            kv_heads,
            queries_per_kv,
            pass_dim,
        ):
            raise ValueError("partial-RoPE scoring requires matching q_pass_mean")
        if not q_pass_mean.is_contiguous():
            raise ValueError("q_pass_mean must be contiguous")
    elif q_pass_mean is not None:
        raise ValueError("full-RoPE scoring must not pass q_pass_mean")
    block_frequency = triton.next_power_of_2(frequency_count)
    _score_paged_keys_mean_kernel[
        (kv_heads * queries_per_kv, triton.cdiv(num_tokens, 16))
    ](
        k_cache,
        source_block_ids,
        q_real,
        q_imag,
        q_pass_mean if q_pass_mean is not None else q_real,
        frequency_scale,
        extra_coefficient,
        phase_cos,
        phase_sin,
        output,
        num_tokens,
        queries_per_kv,
        frequency_count,
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        output.stride(1),
        BLOCK_SIZE=block_size,
        ROPE_STYLE=0 if rope_style == "interleaved" else 1,
        ROTARY_DIM=effective_rotary_dim,
        PASS_DIM=pass_dim,
        BLOCK_N=16,
        BLOCK_F=block_frequency,
        BLOCK_P=triton.next_power_of_2(max(pass_dim, 1)),
    )
    return True


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
    *,
    q_pass_mean: torch.Tensor | None = None,
    rotary_dim: int | None = None,
) -> bool:
    """Compatibility wrapper that prepares phases for one scoring layer."""
    if k_cache.device.type != "npu":
        return False
    phase = omega * float(round_start)
    round_cos = torch.cos(phase)
    round_sin = torch.sin(phase)
    phase_cos = round_cos * offset_cos_mean - round_sin * offset_sin_mean
    phase_sin = round_sin * offset_cos_mean + round_cos * offset_sin_mean
    return score_paged_keys_mean_precomputed(
        k_cache,
        source_block_ids,
        q_real,
        q_imag,
        frequency_scale,
        extra_coefficient,
        phase_cos,
        phase_sin,
        num_tokens,
        output,
        block_size,
        rope_style,
        q_pass_mean=q_pass_mean,
        rotary_dim=rotary_dim,
    )


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
    normalized = (scores - mean[None, :]) * tl.rsqrt(
        tl.maximum(variance[None, :], 1e-6)
    )
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
