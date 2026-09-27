# SPDX-License-Identifier: Apache-2.0
"""Manual Qwen3.5-shaped partial-RoPE scorer comparison on one Ascend NPU."""

from __future__ import annotations

import time

import torch
import torch_npu  # noqa: F401

from vllm_ascend_kvcompress.methods.triattention.cache import gather_paged_range
from vllm_ascend_kvcompress.methods.triattention.kernels import (
    score_paged_keys_mean_precomputed,
)
from vllm_ascend_kvcompress.methods.triattention.scoring import score_post_rope_keys
from vllm_ascend_kvcompress.methods.triattention.stats import (
    DeviceLayerCalibrationStats,
)


def elapsed_ms(function, iterations: int = 3) -> float:
    function()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        function()
    torch.npu.synchronize()
    return 1000 * (time.perf_counter() - start) / iterations


def main() -> None:
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    torch.manual_seed(20260927)
    block_size = 128
    num_tokens = 32768
    rotary_dim = 64
    head_dim = 256
    kv_heads = 1  # Qwen3.5 TP=2: two KV heads split over two ranks.
    queries_per_kv = 8
    frequency_count = rotary_dim // 2
    cache = torch.randn(
        num_tokens // block_size,
        block_size,
        kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    source_blocks = torch.arange(
        num_tokens // block_size, dtype=torch.int64, device=device
    )
    stat_shape = (kv_heads, queries_per_kv, frequency_count)
    q_real = torch.randn(stat_shape, device=device)
    q_imag = torch.randn(stat_shape, device=device)
    q_abs = torch.sqrt(q_real.square() + q_imag.square()) + 0.1
    q_pass = torch.randn(kv_heads, queries_per_kv, head_dim - rotary_dim, device=device)
    scale = torch.ones(stat_shape, device=device)
    omega = torch.logspace(0, -4, frequency_count, device=device)
    offsets = torch.arange(1, 17, dtype=torch.float32, device=device)
    phase = (offsets[:, None] + 65536) * omega[None, :]
    phase_cos = torch.cos(phase).mean(dim=0)
    phase_sin = torch.sin(phase).mean(dim=0)
    extra = q_abs - torch.sqrt(q_real.square() + q_imag.square() + 1e-8)
    stats = DeviceLayerCalibrationStats(
        q_mean_real=q_real,
        q_mean_imag=q_imag,
        q_abs_mean=q_abs,
        freq_scale_sq=scale.square(),
        omega=omega,
        rope_style="half",
        rotary_dim=rotary_dim,
        q_pass_mean=q_pass,
    )
    reference = torch.empty(
        kv_heads, queries_per_kv, num_tokens, dtype=torch.float32, device=device
    )
    fused = torch.empty_like(reference)

    def generic() -> None:
        for start in range(0, num_tokens, 8192):
            count = min(8192, num_tokens - start)
            keys = gather_paged_range(
                cache, source_blocks, start=start, count=count, block_size=block_size
            )
            reference[..., start : start + count].copy_(
                score_post_rope_keys(
                    keys,
                    stats,
                    round_start=65536,
                    offsets=offsets,
                    aggregation="mean",
                )
            )

    def kernel() -> None:
        assert score_paged_keys_mean_precomputed(
            cache,
            source_blocks,
            q_real,
            q_imag,
            scale,
            extra,
            phase_cos,
            phase_sin,
            num_tokens,
            fused,
            block_size,
            "half",
            q_pass_mean=q_pass,
            rotary_dim=rotary_dim,
        )

    generic_ms = elapsed_ms(generic)
    kernel_ms = elapsed_ms(kernel)
    torch.testing.assert_close(fused.cpu(), reference.cpu(), atol=0.02, rtol=0.002)
    print(
        f"Qwen3.5-shaped partial-RoPE scorer, {num_tokens} tokens: "
        f"fallback={generic_ms:.2f} ms, fused={kernel_ms:.2f} ms, "
        f"speedup={generic_ms / kernel_ms:.2f}x"
    )


if __name__ == "__main__":
    main()
