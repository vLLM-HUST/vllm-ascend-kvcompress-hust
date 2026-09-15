# SPDX-License-Identifier: Apache-2.0
"""Manual microbenchmark for Ascend-specific compression kernels."""

from __future__ import annotations

import time

import torch
import torch_npu  # noqa: F401

from vllm_ascend_kvcompress.methods.triattention.cache import (
    gather_paged_range,
    materialize_selected_tokens,
    materialize_token_slots,
    token_slots,
)
from vllm_ascend_kvcompress.methods.triattention.kernels import (
    aggregate_normalized_scores,
    prepare_mean_phase_coefficients,
    score_paged_keys_mean,
    score_paged_keys_mean_precomputed,
    shift_positions,
)
from vllm_ascend_kvcompress.methods.triattention.scoring import score_post_rope_keys
from vllm_ascend_kvcompress.methods.triattention.stats import (
    DeviceLayerCalibrationStats,
)


def elapsed_ms(function, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        function()
    torch.npu.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        function()
    torch.npu.synchronize()
    return (time.perf_counter() - started) * 1000 / iterations


def main() -> None:
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    block_size = 128
    num_tokens = 2176
    budget = 2048
    kv_heads = 8
    queries_per_kv = 5
    head_dim = 128
    num_blocks = 32

    k_cache = torch.randn(
        num_blocks,
        block_size,
        kv_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    v_cache = torch.randn_like(k_cache)
    source = torch.arange(
        (num_tokens + block_size - 1) // block_size,
        device=device,
        dtype=torch.int64,
    )
    destination = source[: budget // block_size]
    keep = torch.arange(budget, device=device, dtype=torch.int64) + (
        num_tokens - budget
    )
    k_workspace = torch.empty(
        budget, kv_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    v_workspace = torch.empty_like(k_workspace)

    def generic_copy() -> None:
        materialize_selected_tokens(
            k_cache, v_cache, source, destination, keep, block_size
        )

    source_slots = token_slots(source, keep, block_size)
    destination_slots = token_slots(
        destination,
        torch.arange(budget, device=device, dtype=torch.int64),
        block_size,
    )

    def kernel_copy() -> None:
        materialize_token_slots(
            k_cache,
            v_cache,
            source_slots,
            destination_slots,
            k_workspace,
            v_workspace,
        )

    generic_copy_ms = elapsed_ms(generic_copy, 2, 10)
    kernel_copy_ms = elapsed_ms(kernel_copy, 2, 10)

    frequency_count = head_dim // 2
    q_shape = (kv_heads, queries_per_kv, frequency_count)
    q_real = torch.randn(q_shape, device=device)
    q_imag = torch.randn(q_shape, device=device)
    q_abs = torch.sqrt(q_real.square() + q_imag.square()) + 0.2
    frequency_scale = torch.ones(q_shape, device=device)
    omega = torch.logspace(0, -4, frequency_count, device=device)
    offsets = torch.tensor(
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
        device=device,
        dtype=torch.float32,
    )
    extra_coefficient = q_abs - torch.sqrt(q_real.square() + q_imag.square() + 1e-8)
    output = torch.empty(
        kv_heads, queries_per_kv, num_tokens, device=device, dtype=torch.float32
    )
    stats = DeviceLayerCalibrationStats(
        q_mean_real=q_real,
        q_mean_imag=q_imag,
        q_abs_mean=q_abs,
        freq_scale_sq=frequency_scale.square(),
        omega=omega,
        rope_style="half",
    )

    def generic_score() -> None:
        keys = gather_paged_range(
            k_cache,
            source,
            start=0,
            count=num_tokens,
            block_size=block_size,
        )
        score_post_rope_keys(
            keys,
            stats,
            round_start=8192,
            offsets=offsets,
            aggregation="mean",
        )

    def kernel_score() -> None:
        score_paged_keys_mean(
            k_cache,
            source,
            q_real,
            q_imag,
            frequency_scale,
            extra_coefficient,
            omega,
            torch.cos(offsets.unsqueeze(1) * omega).mean(dim=0),
            torch.sin(offsets.unsqueeze(1) * omega).mean(dim=0),
            8192,
            num_tokens,
            output,
            block_size,
            "half",
        )

    generic_score_ms = elapsed_ms(generic_score, 2, 5)
    kernel_score_ms = elapsed_ms(kernel_score, 2, 5)

    # Qwen2.5-14B has 48 attention layers; the validated default stride of 8
    # scores six uniformly sampled layers per compression transaction.
    sampled_layers = 6
    layer_omega = omega.unsqueeze(0).expand(sampled_layers, -1).contiguous()
    layer_offset_cos = (
        torch.cos(offsets.unsqueeze(1) * omega)
        .mean(dim=0)
        .unsqueeze(0)
        .expand(sampled_layers, -1)
        .contiguous()
    )
    layer_offset_sin = (
        torch.sin(offsets.unsqueeze(1) * omega)
        .mean(dim=0)
        .unsqueeze(0)
        .expand(sampled_layers, -1)
        .contiguous()
    )
    layer_phase_cos = torch.empty_like(layer_omega)
    layer_phase_sin = torch.empty_like(layer_omega)

    def phase_prepare() -> None:
        prepare_mean_phase_coefficients(
            layer_omega,
            layer_offset_cos,
            layer_offset_sin,
            8192,
            layer_phase_cos,
            layer_phase_sin,
        )

    phase_prepare()

    def precomputed_score() -> None:
        score_paged_keys_mean_precomputed(
            k_cache,
            source,
            q_real,
            q_imag,
            frequency_scale,
            extra_coefficient,
            layer_phase_cos[0],
            layer_phase_sin[0],
            num_tokens,
            output,
            block_size,
            "half",
        )

    phase_prepare_ms = elapsed_ms(phase_prepare, 5, 50)
    precomputed_score_ms = elapsed_ms(precomputed_score, 2, 10)

    head_mean = output.mean(dim=-1)
    head_variance = output.var(dim=-1, correction=0)
    aggregate = torch.empty(num_tokens, device=device, dtype=torch.float32)

    def generic_aggregate() -> None:
        normalized = (output - head_mean.unsqueeze(-1)) * torch.rsqrt(
            head_variance.unsqueeze(-1) + 1e-6
        )
        torch.amax(normalized, dim=(0, 1), out=aggregate)

    def kernel_aggregate() -> None:
        aggregate_normalized_scores(
            output,
            head_mean,
            head_variance,
            aggregate,
            num_tokens,
            layer_aggregation="mean",
            first_layer=True,
        )

    generic_aggregate_ms = elapsed_ms(generic_aggregate, 5, 50)
    kernel_aggregate_ms = elapsed_ms(kernel_aggregate, 5, 50)

    positions = torch.arange(4096, device=device, dtype=torch.int64)
    request_indices = torch.arange(4096, device=device, dtype=torch.int64) % 16
    request_offsets = torch.arange(16, device=device, dtype=torch.int64) * 128
    physical_positions = torch.empty_like(positions)

    def generic_shift() -> None:
        torch.sub(
            positions,
            request_offsets.index_select(0, request_indices),
            out=physical_positions,
        )

    def kernel_shift() -> None:
        shift_positions(
            positions,
            request_indices,
            request_offsets,
            physical_positions,
        )

    generic_shift_ms = elapsed_ms(generic_shift, 5, 100)
    kernel_shift_ms = elapsed_ms(kernel_shift, 5, 100)

    print(
        f"copy generic={generic_copy_ms:.3f}ms kernel={kernel_copy_ms:.3f}ms "
        f"speedup={generic_copy_ms / kernel_copy_ms:.2f}x"
    )
    print(
        f"score generic={generic_score_ms:.3f}ms kernel={kernel_score_ms:.3f}ms "
        f"speedup={generic_score_ms / kernel_score_ms:.2f}x"
    )
    print(
        f"score precomputed={precomputed_score_ms:.3f}ms "
        f"phase6={phase_prepare_ms:.3f}ms "
        f"amortized={precomputed_score_ms + phase_prepare_ms / sampled_layers:.3f}ms"
    )
    print(
        f"aggregate generic={generic_aggregate_ms:.3f}ms "
        f"kernel={kernel_aggregate_ms:.3f}ms "
        f"speedup={generic_aggregate_ms / kernel_aggregate_ms:.2f}x"
    )
    print(
        f"offset generic={generic_shift_ms:.3f}ms kernel={kernel_shift_ms:.3f}ms "
        f"speedup={generic_shift_ms / kernel_shift_ms:.2f}x"
    )


if __name__ == "__main__":
    main()
