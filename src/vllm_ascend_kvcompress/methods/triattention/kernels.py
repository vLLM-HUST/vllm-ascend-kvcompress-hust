# SPDX-License-Identifier: Apache-2.0
"""TriAttention 热路径使用的 Ascend Triton 内核。"""

from __future__ import annotations

import torch

# 本文件含三类热路径：位置偏移、mean 模式分页评分、标准化与聚合。
# 前者当前用 torch 算子；后两者是 Triton-Ascend kernel，
# CPU 调用返回 False 供上层回退。
# 装饰器替身仅允许 CPU 环境导入源码，不是 NPU 执行实现；NPU 缺 Triton 会显式报错。
try:
    import triton
    import triton.language as tl
except (ImportError, ModuleNotFoundError):
    # 静态清单发现及纯 CPU 校验不得导入加速器工具链。
    # 下方 NPU 调用在依赖缺失时仍会报错。
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
    """为每层、每个频率预计算一个平均偏移 RoPE 系数。

    评分网格为每个 query 头与 token 分块启动一个程序，若在其中计算
    三角函数，会重复相同工作数千次。此小型内核改为对每个采样层与
    频率组合只计算一次。
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
    # 每个程序处理最多 256 个系数，尾部用 mask 屏蔽越界项。
    # 网格只覆盖 layer×frequency，不随 Query 头数和 token 数重复计算三角函数。
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
    # 二维 grid：[本地 Query 头,token tile]；query_head/queries_per_kv 定位 KV 头。
    # 直接按块表读取已旋转 K，避免先 gather 整段 cache 到临时张量。
    # 统计按 [KV 头,Query 组,frequency] 连续排布，故可把前两维视为 query_head。
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

    # 页编号来自 source_block_ids；cache_base 使用实际页/槽/头 stride。
    # 最后一维已由 provider 验证连续，所以频率维可直接加元素下标，支持页间 padding。
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
    # partial RoPE 的非旋转部分直接加 E[Q_pass]·K_pass，无相位项。
    # PASS_DIM 是编译期常量，完整 RoPE 不会执行这段载入与点积。
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
    """使用预计算的 query 相位，对全 RoPE 或部分 RoPE 分页 key 评分。"""
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
    # 二维网格分别覆盖本地 Query 头与 token 分块；每块一次读取一组旋转频率。
    # 显式传入 cache stride，使页间带 padding 的存储也可直接评分。
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
    # mean/variance 由上层对完整 token 维计算；本 kernel 只做归一化及头间 max。
    # MODE 控制层间累积是 sum 还是 max；mean 的最终除层数在 method 中完成。
    # FIRST_LAYER 也用于 TP 暂存每层独立向量，避免误读 workspace 的旧值。
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
    """融合归一化、query 头最大值归约与跨层累加。"""
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
