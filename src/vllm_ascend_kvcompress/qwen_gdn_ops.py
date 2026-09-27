# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
# Derived from flash-linear-attention (MIT), Copyright (c) 2023-2025,
# Songlin Yang and Yu Zhang. See NOTICE for the attribution boundary.
"""Plugin-owned Triton implementation of the missing Qwen GDN output op.

The synced Ascend Python path calls ``_C_ascend.chunk_fwd_o_vllm`` while some
deployed Ascend binaries predate that registration. The plugin defines exactly
this missing op, compiles its kernel through Triton Ascend on first use, and
leaves an upstream implementation untouched when one is present.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as functional
from vllm.triton_utils import tl, triton

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
    """Emulate the host MTP convolution's extended rolling-state semantics."""
    if cache_indices.ndim == 2:
        if cache_indices.shape[1] < 1:
            raise RuntimeError("Qwen GDN MTP cache block table is empty")
        # Match the host Qwen GDN reference: convolution updates the target
        # state; the recurrent MTP path owns the additional draft states.
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
        # Ascend's prepare_chunk_indices labels nonempty sequences densely;
        # zero-length segments consume no chunk and no sequence label.
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
    """Register the plugin JIT kernel only when Ascend lacks the same schema."""
    global _LIBRARY
    if hasattr(torch.ops._C_ascend, "chunk_fwd_o_vllm"):
        return False
    if _LIBRARY is not None:
        return True
    library = torch.library.Library("_C_ascend", "FRAGMENT")
    library.define(_SCHEMA)
    library.impl("chunk_fwd_o_vllm", _run, "PrivateUse1")
    library.impl("chunk_fwd_o_vllm", _meta, "Meta")
    _LIBRARY = library
    return True
