#!/usr/bin/env python3
"""Bounded BF16 numerical smoke for the plugin-owned Qwen GDN output op."""

from __future__ import annotations

import math

import torch
import torch_npu  # noqa: F401

from vllm_ascend_kvcompress.plugin import _prepare_current_triton_runtime
from vllm_ascend_kvcompress.qwen_gdn_ops import _run as run_plugin_fallback
from vllm_ascend_kvcompress.qwen_gdn_ops import install_missing_chunk_output_op


def reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor,
    offsets: tuple[int, ...],
    scale: float,
) -> torch.Tensor:
    output = torch.empty_like(v)
    chunk_base = 0
    for start, end in zip(offsets, offsets[1:], strict=False):
        for chunk in range(math.ceil((end - start) / 64)):
            left = start + chunk * 64
            right = min(left + 64, end)
            count = right - left
            causal = torch.tril(torch.ones(count, count, dtype=torch.bool))
            for head in range(v.shape[1]):
                key_head = head // (v.shape[1] // q.shape[1])
                query = q[0, key_head, left:right].float()
                key = k[0, key_head, left:right].float()
                value = v[0, head, left:right].float()
                state = h[0, head, chunk_base + chunk].float()
                gate = g[0, head, left:right].float()
                weights = query @ key.T
                gate_diff = gate[:, None] - gate[None, :]
                weights *= torch.exp(
                    torch.where(gate_diff <= 0, gate_diff, float("-inf"))
                )
                weights = torch.where(causal, weights, 0)
                result = (
                    (query @ state) * torch.exp(gate[:, None]) + weights @ value
                ) * scale
                output[0, head, left:right] = result.to(output.dtype)
        chunk_base += math.ceil((end - start) / 64)
    return output


def main() -> None:
    torch.manual_seed(7)
    import vllm_ascend.vllm_ascend_C  # noqa: F401
    from vllm_ascend.utils import bootstrap_custom_op_env

    # A direct op test bypasses the normal platform bootstrap performed by
    # serving, so expose the host's installed vendor kernels explicitly.
    bootstrap_custom_op_env(include_vendor_lib=True)

    installed = install_missing_chunk_output_op()
    batch, key_heads, heads, tokens, key_dim, value_dim = 1, 2, 4, 97, 128, 128
    q = torch.randn(batch, key_heads, tokens, key_dim).mul_(0.1).to(torch.bfloat16)
    k = torch.randn_like(q).mul_(0.1)
    v = torch.randn(batch, heads, tokens, value_dim).mul_(0.1).to(torch.bfloat16)
    h = torch.randn(batch, heads, 2, key_dim, value_dim).mul_(0.1).to(torch.bfloat16)
    g = -torch.rand(batch, heads, tokens).mul_(0.01).cumsum(dim=2)
    offsets = (0, 64, 97)
    chunks = (0, 0, 1, 0)
    scale = key_dim**-0.5
    expected = reference(q, k, v, h, g, offsets, scale)
    npu_q, npu_k, npu_v, npu_h, npu_g = (tensor.npu() for tensor in (q, k, v, h, g))
    actual = torch.ops._C_ascend.chunk_fwd_o_vllm(
        npu_q,
        npu_k,
        npu_v,
        npu_h,
        scale,
        g=npu_g,
        cu_seqlens=offsets,
        chunk_indices=chunks,
        chunk_size=64,
        transpose_state_layout=False,
    ).cpu()
    torch.npu.synchronize()
    max_abs = (actual.float() - expected.float()).abs().max().item()
    assert torch.allclose(actual.float(), expected.float(), atol=0.015, rtol=0.05), (
        max_abs
    )
    source = "plugin fallback" if installed else "host operator"
    print(f"PASS {source} GDN output BF16 max_abs={max_abs:.6f}")
    if not installed:
        # The new host registers the same schema, so verify the plugin's own
        # Triton fallback directly without replacing the host registration.
        _prepare_current_triton_runtime()
        fallback = run_plugin_fallback(
            npu_q,
            npu_k,
            npu_v,
            npu_h,
            scale,
            g=npu_g,
            cu_seqlens=offsets,
            chunk_indices=chunks,
            chunk_size=64,
            transpose_state_layout=False,
        ).cpu()
        torch.npu.synchronize()
        fallback_max_abs = (fallback.float() - expected.float()).abs().max().item()
        assert torch.allclose(
            fallback.float(), expected.float(), atol=0.015, rtol=0.05
        ), fallback_max_abs
        print(f"PASS plugin fallback GDN output BF16 max_abs={fallback_max_abs:.6f}")


if __name__ == "__main__":
    main()
