# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_ascend_kvcompress.methods.triattention.scoring import (
    score_post_rope_keys,
)
from vllm_ascend_kvcompress.methods.triattention.stats import (
    DeviceLayerCalibrationStats,
)


def _stats() -> DeviceLayerCalibrationStats:
    return DeviceLayerCalibrationStats(
        q_mean_real=torch.tensor([[[1.0, 0.5], [0.25, 0.75]]]),
        q_mean_imag=torch.tensor([[[0.0, 0.5], [0.5, 0.25]]]),
        q_abs_mean=torch.ones(1, 2, 2),
        freq_scale_sq=torch.ones(1, 2, 2),
        omega=torch.tensor([1.0, 0.1]),
        rope_style="half",
    )


def _slow_reference(
    keys: torch.Tensor,
    stats: DeviceLayerCalibrationStats,
    round_start: int,
    offsets: torch.Tensor,
) -> torch.Tensor:
    key_real = keys[..., :2].permute(1, 0, 2).unsqueeze(1)
    key_imag = keys[..., 2:].permute(1, 0, 2).unsqueeze(1)
    q_real = stats.q_mean_real.unsqueeze(2)
    q_imag = stats.q_mean_imag.unsqueeze(2)
    product_real = q_real * key_real + q_imag * key_imag
    product_imag = q_imag * key_real - q_real * key_imag
    candidates = []
    for offset in offsets:
        phase = (float(round_start) + offset) * stats.omega
        candidates.append(
            (product_real * torch.cos(phase) - product_imag * torch.sin(phase)).sum(
                dim=-1
            )
        )
    position = torch.stack(candidates).mean(dim=0)
    key_abs = torch.sqrt(key_real.square() + key_imag.square())
    q_mean_abs = torch.sqrt(
        stats.q_mean_real.square() + stats.q_mean_imag.square() + 1e-8
    )
    extra = (key_abs * (stats.q_abs_mean - q_mean_abs).unsqueeze(2)).sum(dim=-1)
    return position + extra


def test_vectorized_mean_matches_reference() -> None:
    keys = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0]],
            [[0.5, 1.5, 2.5, 3.5]],
            [[2.0, 1.0, 4.0, 3.0]],
        ]
    )
    offsets = torch.tensor([1.0, 2.0, 4.0])
    actual = score_post_rope_keys(
        keys,
        _stats(),
        round_start=3,
        offsets=offsets,
        aggregation="mean",
    )
    expected = _slow_reference(keys, _stats(), 3, offsets)
    torch.testing.assert_close(actual, expected)


def test_max_aggregation_shape() -> None:
    keys = torch.randn(7, 1, 4)
    scores = score_post_rope_keys(
        keys,
        _stats(),
        round_start=7,
        offsets=torch.tensor([1.0, 2.0]),
        aggregation="max",
    )
    assert scores.shape == (1, 2, 7)


def test_post_rope_keys_use_one_query_side_scaling_factor() -> None:
    stats = DeviceLayerCalibrationStats(
        q_mean_real=torch.ones(1, 1, 1),
        q_mean_imag=torch.zeros(1, 1, 1),
        q_abs_mean=torch.ones(1, 1, 1),
        freq_scale_sq=torch.full((1, 1, 1), 4.0),
        omega=torch.zeros(1),
        rope_style="half",
    )
    # The post-RoPE key already carries one scale factor of two. The scorer
    # supplies the query-side factor of two, for a final dot product of four.
    keys = torch.tensor([[[2.0, 0.0]]])
    scores = score_post_rope_keys(
        keys,
        stats,
        round_start=1,
        offsets=torch.tensor([1.0]),
        aggregation="mean",
    )
    torch.testing.assert_close(scores, torch.tensor([[[4.0]]]))


def test_partial_rope_score_includes_unrotated_content_dimensions() -> None:
    stats = DeviceLayerCalibrationStats(
        q_mean_real=torch.zeros(1, 1, 1),
        q_mean_imag=torch.zeros(1, 1, 1),
        q_abs_mean=torch.zeros(1, 1, 1),
        freq_scale_sq=torch.ones(1, 1, 1),
        omega=torch.zeros(1),
        rope_style="half",
        rotary_dim=2,
        q_pass_mean=torch.tensor([[[2.0, -1.0]]]),
    )
    keys = torch.tensor(
        [
            [[0.0, 0.0, 3.0, 4.0]],
            [[0.0, 0.0, -2.0, 1.0]],
        ]
    )

    scores = score_post_rope_keys(
        keys,
        stats,
        round_start=1,
        offsets=torch.tensor([1.0]),
        aggregation="mean",
    )

    torch.testing.assert_close(scores, torch.tensor([[[2.0, -5.0]]]))
