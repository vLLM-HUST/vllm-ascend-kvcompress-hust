# SPDX-License-Identifier: Apache-2.0
"""Manual Ascend correctness/compilation smoke test for plugin kernels."""

from __future__ import annotations

import torch
import torch_npu  # noqa: F401

from vllm_ascend_kvcompress.methods.triattention.cache import (
    gather_paged_range,
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


def main() -> None:
    device = torch.device("npu:0")
    torch.npu.set_device(device)

    positions = torch.tensor([300, 20, 301], device=device, dtype=torch.int64)
    request_indices = torch.tensor([0, 1, 0], device=device, dtype=torch.int64)
    offsets = torch.tensor([172, 0], device=device, dtype=torch.int64)
    shifted = torch.empty_like(positions)
    shift_positions(positions, request_indices, offsets, shifted)
    torch.testing.assert_close(
        shifted.cpu(), torch.tensor([128, 20, 129], dtype=torch.int64)
    )

    block_size = 128
    cpu_cache = torch.arange(4 * block_size * 2 * 8, dtype=torch.float32).view(
        4, block_size, 2, 8
    )
    k_cache = cpu_cache.to(dtype=torch.bfloat16, device=device)
    v_cache = (cpu_cache + 10000).to(dtype=torch.bfloat16, device=device)
    source = torch.tensor([2, 0, 3], device=device, dtype=torch.int64)
    destination = torch.tensor([2], device=device, dtype=torch.int64)
    keep = torch.arange(0, block_size, device=device, dtype=torch.int64) + 1
    expected_k = gather_paged_range(
        k_cache,
        source,
        start=1,
        count=block_size,
        block_size=block_size,
    ).clone()
    expected_v = gather_paged_range(
        v_cache,
        source,
        start=1,
        count=block_size,
        block_size=block_size,
    ).clone()
    k_workspace = torch.empty_like(expected_k)
    v_workspace = torch.empty_like(expected_v)
    source_slots = token_slots(source, keep, block_size)
    destination_slots = token_slots(
        destination,
        torch.arange(block_size, device=device, dtype=torch.int64),
        block_size,
    )
    materialize_token_slots(
        k_cache,
        v_cache,
        source_slots,
        destination_slots,
        k_workspace,
        v_workspace,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(k_cache[2].cpu(), expected_k.cpu())
    torch.testing.assert_close(v_cache[2].cpu(), expected_v.cpu())

    score_cache = (
        torch.linspace(-1.0, 1.0, 3 * block_size * 2 * 8, dtype=torch.float32)
        .view(3, block_size, 2, 8)
        .to(dtype=torch.bfloat16, device=device)
    )
    score_source = torch.tensor([2, 0], device=device, dtype=torch.int64)
    q_real = torch.linspace(0.1, 0.8, 16, device=device).view(2, 2, 4)
    q_imag = torch.linspace(-0.4, 0.3, 16, device=device).view(2, 2, 4)
    q_abs = torch.sqrt(q_real.square() + q_imag.square()) + 0.2
    frequency_scale = torch.linspace(0.8, 1.2, 16, device=device).view(2, 2, 4)
    omega = torch.tensor([1.0, 0.1, 0.01, 0.001], device=device)
    future_offsets = torch.tensor([1.0, 2.0, 4.0], device=device)
    extra_coefficient = q_abs - torch.sqrt(q_real.square() + q_imag.square() + 1e-8)
    output = torch.empty((2, 2, 256), dtype=torch.float32, device=device)
    assert score_paged_keys_mean(
        score_cache,
        score_source,
        q_real,
        q_imag,
        frequency_scale,
        extra_coefficient,
        omega,
        torch.cos(future_offsets.unsqueeze(1) * omega).mean(dim=0),
        torch.sin(future_offsets.unsqueeze(1) * omega).mean(dim=0),
        300,
        256,
        output,
        block_size,
        "half",
    )
    keys = gather_paged_range(
        score_cache,
        score_source,
        start=0,
        count=256,
        block_size=block_size,
    )
    stats = DeviceLayerCalibrationStats(
        q_mean_real=q_real,
        q_mean_imag=q_imag,
        q_abs_mean=q_abs,
        freq_scale_sq=frequency_scale.square(),
        omega=omega,
        rope_style="half",
    )
    expected_scores = score_post_rope_keys(
        keys,
        stats,
        round_start=300,
        offsets=future_offsets,
        aggregation="mean",
    )
    torch.npu.synchronize()
    torch.testing.assert_close(
        output.cpu(), expected_scores.cpu(), rtol=2e-3, atol=2e-3
    )

    layer_count = 3
    layer_omega = omega.unsqueeze(0).expand(layer_count, -1).contiguous()
    offset_cos = torch.cos(future_offsets.unsqueeze(1) * omega).mean(dim=0)
    offset_sin = torch.sin(future_offsets.unsqueeze(1) * omega).mean(dim=0)
    layer_offset_cos = offset_cos.unsqueeze(0).expand(layer_count, -1).contiguous()
    layer_offset_sin = offset_sin.unsqueeze(0).expand(layer_count, -1).contiguous()
    phase_cos = torch.empty_like(layer_omega)
    phase_sin = torch.empty_like(layer_omega)
    assert prepare_mean_phase_coefficients(
        layer_omega,
        layer_offset_cos,
        layer_offset_sin,
        300,
        phase_cos,
        phase_sin,
    )
    precomputed_output = torch.empty_like(output)
    assert score_paged_keys_mean_precomputed(
        score_cache,
        score_source,
        q_real,
        q_imag,
        frequency_scale,
        extra_coefficient,
        phase_cos[1],
        phase_sin[1],
        256,
        precomputed_output,
        block_size,
        "half",
    )
    torch.npu.synchronize()
    torch.testing.assert_close(
        precomputed_output.cpu(), expected_scores.cpu(), rtol=2e-3, atol=2e-3
    )

    head_mean = output.mean(dim=-1)
    head_variance = output.var(dim=-1, correction=0)
    aggregate = torch.empty(256, dtype=torch.float32, device=device)
    assert aggregate_normalized_scores(
        output,
        head_mean,
        head_variance,
        aggregate,
        256,
        layer_aggregation="mean",
        first_layer=True,
    )
    expected_aggregate = (
        (output - head_mean.unsqueeze(-1))
        * torch.rsqrt(head_variance.unsqueeze(-1) + 1e-6)
    ).amax(dim=(0, 1))
    torch.npu.synchronize()
    torch.testing.assert_close(
        aggregate.cpu(), expected_aggregate.cpu(), rtol=2e-3, atol=2e-3
    )
    print("Ascend kernel smoke: PASS")


if __name__ == "__main__":
    main()
