# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
# Derived from flash-linear-attention (MIT), Copyright (c) 2023-2025,
# Songlin Yang and Yu Zhang. See NOTICE for the attribution boundary.
"""由插件提供缺失的 Qwen GDN 输出算子的 Triton 实现。

同步后的 Ascend Python 路径调用 _C_ascend.chunk_fwd_o_vllm，
但部分部署中的 Ascend 二进制尚未注册它。插件仅补齐此算子，
首次使用时通过 Triton Ascend 编译内核，已有上游实现时优先保留。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as functional
from vllm.triton_utils import tl, triton

# 此处实现 hybrid 模型运行所需的 GDN 运算兼容，和 TriAttention 压缩独立。
# _LIBRARY 持有 torch.library 注册对象，避免对象析构导致自定义算子注册被撤销。
_LIBRARY: torch.library.Library | None = None
_SCHEMA = (
    "chunk_fwd_o_vllm(Tensor q, Tensor k, Tensor v, Tensor h, float scale, "
    "*, Tensor? g=None, Tensor? g_gamma=None, int[]? cu_seqlens=None, "
    "int[]? chunk_indices=None, int? chunk_size=None, "
    "bool? transpose_state_layout=False) -> Tensor"
)


def causal_conv1d_spec_reference(
    output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    bias: torch.Tensor | None,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    activation_mode: int,
    pad_slot_id: int,
) -> torch.Tensor:
    """模拟宿主 MTP 卷积扩展后的滚动状态语义。"""
    if cache_indices.ndim == 2:
        if cache_indices.shape[1] < 1:
            raise RuntimeError("Qwen GDN MTP cache block table is empty")
        # 与宿主 Qwen GDN 参考实现保持一致：卷积更新目标状态，
        # 额外草稿状态由循环 MTP 路径负责。
        cache_indices = cache_indices[:, 0]
    if activation_mode not in (0, 1):
        raise RuntimeError("unsupported Qwen GDN activation mode")
    if output.shape != x.shape or output.dtype != x.dtype or output.device != x.device:
        raise RuntimeError("Qwen GDN output does not match input")
    if x.ndim != 2 or query_start_loc.ndim != 1 or cache_indices.ndim != 1:
        raise RuntimeError("Qwen GDN speculative input or metadata shape is invalid")
    if (
        num_accepted_tokens.ndim != 1
        or query_start_loc.numel() != cache_indices.numel() + 1
    ):
        raise RuntimeError("Qwen GDN speculative row counts disagree")
    if num_accepted_tokens.numel() != cache_indices.numel():
        raise RuntimeError("Qwen GDN accepted-token row count disagrees")
    feature_dim = x.shape[1]
    if weight.shape[0] != feature_dim and weight.shape[1] == feature_dim:
        weight = weight.T
    if weight.ndim != 2 or weight.shape[0] != feature_dim:
        raise RuntimeError("Qwen GDN speculative weight dimensions disagree")
    width = weight.shape[1]
    if conv_state.shape[-2] != feature_dim and conv_state.shape[-1] == feature_dim:
        conv_state = conv_state.transpose(-1, -2)
    if conv_state.shape[-2] != feature_dim or conv_state.shape[-1] < width - 1:
        raise RuntimeError("Qwen GDN speculative state dimensions disagree")
    # 这是兼容参考路径，CPU 读取各请求边界/接受数会同步设备；不属于纯设备压缩热路径。
    # accepted-1 定位有效历史窗口，用于回滚后重新拼接已接受前缀与本次候选 token。
    boundaries = [int(value) for value in query_start_loc.detach().cpu().tolist()]
    indices = [int(value) for value in cache_indices.detach().cpu().tolist()]
    accepted = [int(value) for value in num_accepted_tokens.detach().cpu().tolist()]
    if boundaries[0] != 0 or boundaries[-1] != x.shape[0]:
        raise RuntimeError("Qwen GDN speculative token bounds disagree")
    weight = weight.contiguous()
    output.copy_(x)
    for row, cache_index in enumerate(indices):
        start, end = boundaries[row : row + 2]
        if cache_index == pad_slot_id or end == start:
            continue
        # 接受数定位当前有效的历史状态窗口；草稿被拒绝后，不能直接沿用乐观尾部。
        # 先复制有效历史，再拼接本步候选 token，避免原地更新覆盖仍需读取的状态。
        offset = accepted[row] - 1
        if (
            start < 0
            or end < start
            or cache_index < 0
            or cache_index >= conv_state.shape[0]
            or offset < 0
            or offset + width - 1 > conv_state.shape[-1]
            or width - 2 + end - start > conv_state.shape[-1]
        ):
            raise RuntimeError("Qwen GDN speculative cache index or span is invalid")
        state = conv_state[cache_index]
        prior = state[:, offset : offset + width - 1].clone()
        tokens = x[start:end].T.to(weight.dtype)
        joined = torch.cat((prior.to(weight.dtype), tokens), dim=-1)
        result = functional.conv1d(
            joined.unsqueeze(0), weight.unsqueeze(1), bias, groups=feature_dim
        ).squeeze(0)
        if activation_mode:
            result = functional.silu(result)
        output[start:end].copy_(result.T.to(x.dtype))
        rolling_state = torch.cat((prior[:, 1:], tokens), dim=-1)
        state[:, : rolling_state.shape[-1]].copy_(rolling_state.to(state.dtype))
    return output


@triton.jit
def _safe_exp(value):
    return tl.exp(tl.where(value <= 0, value, float("-inf")))


@triton.jit(do_not_specialize=["scale", "length", "start", "chunk_base"])
def _chunk_fwd_o_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    h_ptr,
    g_ptr,
    o_ptr,
    scale,
    length,
    start,
    chunk_base,
    q_sb,
    q_sh,
    q_st,
    q_sk,
    k_sb,
    k_sh,
    k_st,
    k_sk,
    v_sb,
    v_sh,
    v_st,
    v_sv,
    h_sb,
    h_sh,
    h_sc,
    h_sk,
    h_sv,
    g_sb,
    g_sh,
    g_st,
    o_sb,
    o_sh,
    o_st,
    o_sv,
    batch,
    heads: tl.constexpr,
    key_heads: tl.constexpr,
    key_dim: tl.constexpr,
    value_dim: tl.constexpr,
    use_gate: tl.constexpr,
    block_t: tl.constexpr,
    block_k: tl.constexpr,
    block_v: tl.constexpr,
):
    value_tile = tl.program_id(0)
    head = tl.program_id(1)
    key_head = head // (heads // key_heads)
    q_base = q_ptr + batch * q_sb + key_head * q_sh + start * q_st
    k_base = k_ptr + batch * k_sb + key_head * k_sh + start * k_st
    v_base = v_ptr + batch * v_sb + head * v_sh + start * v_st
    o_base = o_ptr + batch * o_sb + head * o_sh + start * o_st
    g_base = g_ptr + batch * g_sb + head * g_sh + start * g_st

    # GDN chunk 输出由跨 chunk 状态项 Q·H 与当前 chunk 的因果 QKᵀ·V 两项组成。
    # 可选 gate 修正历史衰减，三角 mask 禁止读未来 token；这里没有 KV token 淘汰。
    for chunk in range(tl.cdiv(length, block_t)):
        h_base = h_ptr + batch * h_sb + head * h_sh + (chunk_base + chunk) * h_sc
        output = tl.zeros((block_t, block_v), dtype=tl.float32)
        scores = tl.zeros((block_t, block_t), dtype=tl.float32)
        for key_tile in range(tl.cdiv(key_dim, block_k)):
            q_block = tl.make_block_ptr(
                q_base,
                (length, key_dim),
                (q_st, q_sk),
                (chunk * block_t, key_tile * block_k),
                (block_t, block_k),
                (1, 0),
            )
            k_block = tl.make_block_ptr(
                k_base,
                (key_dim, length),
                (k_sk, k_st),
                (key_tile * block_k, chunk * block_t),
                (block_k, block_t),
                (0, 1),
            )
            h_block = tl.make_block_ptr(
                h_base,
                (key_dim, value_dim),
                (h_sk, h_sv),
                (key_tile * block_k, value_tile * block_v),
                (block_k, block_v),
                (1, 0),
            )
            q_value = tl.load(q_block, boundary_check=(0, 1))
            k_value = tl.load(k_block, boundary_check=(0, 1))
            h_value = tl.load(h_block, boundary_check=(0, 1))
            output += tl.dot(q_value, h_value)
            scores += tl.dot(q_value, k_value)

        if use_gate:
            rows = chunk * block_t + tl.arange(0, block_t)
            gate = tl.load(g_base + rows * g_st, mask=rows < length, other=0)
            output *= tl.exp(gate)[:, None]
            scores *= _safe_exp(gate[:, None] - gate[None, :])
        rows = tl.arange(0, block_t)
        scores = tl.where(rows[:, None] >= rows[None, :], scores, 0)
        v_block = tl.make_block_ptr(
            v_base,
            (length, value_dim),
            (v_st, v_sv),
            (chunk * block_t, value_tile * block_v),
            (block_t, block_v),
            (1, 0),
        )
        o_block = tl.make_block_ptr(
            o_base,
            (length, value_dim),
            (o_st, o_sv),
            (chunk * block_t, value_tile * block_v),
            (block_t, block_v),
            (1, 0),
        )
        values = tl.load(v_block, boundary_check=(0, 1))
        output = output * scale + tl.dot(scores.to(values.dtype), values) * scale
        tl.store(o_block, output.to(o_block.dtype.element_ty), boundary_check=(0, 1))


# 把普通 batch 或 packed 变长输入转换为 (batch,start,length,chunk_base)。
# chunk_base 指向 h 的 chunk 轴；零长序列不占 chunk，需与宿主标签规则一致。
def _segments(
    q: torch.Tensor,
    block_t: int,
    cu_seqlens: Sequence[int] | None,
    chunk_indices: Sequence[int] | None,
) -> list[tuple[int, int, int, int]]:
    batch, _, total, _ = q.shape
    if cu_seqlens is None:
        if chunk_indices is not None:
            raise ValueError("chunk_indices require cu_seqlens")
        return [(b, 0, total, 0) for b in range(batch)]
    if batch != 1:
        raise ValueError("packed variable-length GDN requires batch size one")
    offsets = [int(value) for value in cu_seqlens]
    if len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != total:
        raise ValueError("cu_seqlens must cover the packed input exactly")
    if any(left > right for left, right in zip(offsets, offsets[1:], strict=False)):
        raise ValueError("cu_seqlens must be nondecreasing")
    result = []
    chunk_base = 0
    for start, end in zip(offsets, offsets[1:], strict=False):
        result.append((0, start, end - start, chunk_base))
        chunk_base += math.ceil((end - start) / block_t)
    if chunk_indices is not None:
        flat = tuple(int(value) for value in chunk_indices)
        if len(flat) != 2 * chunk_base:
            raise ValueError("chunk_indices must contain one pair per chunk")
        # Ascend 的 prepare_chunk_indices 对非空序列连续编号；
        # 零长度片段不占用分块，也不占用序列编号。
        expected = tuple(
            value
            for seq, (_, _, length, _) in enumerate(
                segment for segment in result if segment[2] > 0
            )
            for chunk in range(math.ceil(length / block_t))
            for value in (seq, chunk)
        )
        if flat != expected:
            raise ValueError("chunk_indices do not match packed segment order")
    return result


def _validate(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None,
    g_gamma: torch.Tensor | None,
    cu_seqlens: Sequence[int] | None,
    chunk_indices: Sequence[int] | None,
    chunk_size: int | None,
    transpose_state_layout: bool | None,
) -> tuple[int, list[tuple[int, int, int, int]]]:
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4 or h.ndim != 5:
        raise ValueError("expected q/k [B,Hg,T,K], v [B,H,T,V], h [B,H,C,K,V]")
    batch, key_heads, total, key_dim = q.shape
    if v.shape[0] != batch or v.shape[2] != total or h.shape[0] != batch:
        raise ValueError("GDN batch or token dimensions do not match")
    heads, value_dim = v.shape[1], v.shape[3]
    # GDN 的 Q/K 头数可能少于 V/输出头数，用整除分组关系将其广播到值头。
    # h 按 [batch,值头,chunk,key_dim,value_dim] 存储跨 chunk 的循环状态。
    if heads % key_heads or h.shape[1] != heads or h.shape[-2:] != (key_dim, value_dim):
        raise ValueError("GDN head or feature dimensions do not match")
    if g is not None and g.shape != (batch, heads, total):
        raise ValueError("gate must have shape [B,H,T]")
    if g_gamma is not None or transpose_state_layout:
        raise ValueError("g_gamma and transposed state layout are unsupported")
    block_t = 64 if chunk_size is None else int(chunk_size)
    if block_t != 64:
        raise ValueError("Qwen GDN output kernel requires chunk_size=64")
    segments = _segments(q, block_t, cu_seqlens, chunk_indices)
    needed_chunks = max(
        (base + math.ceil(length / block_t) for _, _, length, base in segments),
        default=0,
    )
    if h.shape[2] < needed_chunks:
        raise ValueError("GDN state has too few chunks")
    return block_t, segments


def _run(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    scale: float,
    *,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    cu_seqlens: Sequence[int] | None = None,
    chunk_indices: Sequence[int] | None = None,
    chunk_size: int | None = None,
    transpose_state_layout: bool | None = False,
) -> torch.Tensor:
    block_t, segments = _validate(
        q,
        k,
        v,
        h,
        g,
        g_gamma,
        cu_seqlens,
        chunk_indices,
        chunk_size,
        transpose_state_layout,
    )
    output = torch.empty_like(v)
    heads = v.shape[1]
    key_heads = q.shape[1]
    value_dim = v.shape[3]
    key_dim = q.shape[3]
    # packed 输入的每个真实序列分别启动 kernel，避免因果计算跨越请求边界。
    # base 指向该序列的首个状态 chunk，不是 token 下标。
    for batch, start, length, base in segments:
        if length == 0:
            continue
        _chunk_fwd_o_kernel[(triton.cdiv(value_dim, 128), heads)](
            q,
            k,
            v,
            h,
            g if g is not None else q,
            output,
            scale,
            length,
            start,
            base,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *h.stride(),
            *(g.stride() if g is not None else (0, 0, 0)),
            *output.stride(),
            batch,
            heads,
            key_heads,
            key_dim,
            value_dim,
            g is not None,
            block_t,
            128,
            128,
            num_warps=4,
            num_stages=2,
        )
    return output


# Meta 实现只描述输出形状/类型供 tracing 使用，不执行数值计算。
# 真实 NPU 调用走 PrivateUse1 的 _run；安装前先检查宿主是否已有原生注册。
def _meta(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    scale: float,
    **kwargs: Any,
) -> torch.Tensor:
    del q, k, h, scale, kwargs
    return torch.empty_like(v)


def install_missing_chunk_output_op() -> bool:
    """仅在 Ascend 缺少同名 schema 时注册插件 JIT 内核。"""
    global _LIBRARY
    if hasattr(torch.ops._C_ascend, "chunk_fwd_o_vllm"):
        return False
    if _LIBRARY is not None:
        return True
    # FRAGMENT 为已有命名空间补充单个缺失算子。
    # PrivateUse1 对应 NPU 数值实现，Meta 只向追踪器提供输出形状与类型。
    library = torch.library.Library("_C_ascend", "FRAGMENT")
    library.define(_SCHEMA)
    library.impl("chunk_fwd_o_vllm", _run, "PrivateUse1")
    library.impl("chunk_fwd_o_vllm", _meta, "Meta")
    _LIBRARY = library
    return True
