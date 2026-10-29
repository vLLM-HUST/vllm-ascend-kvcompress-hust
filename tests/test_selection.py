# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend_kvcompress.methods.base import (
    CompressionRequest,
    LayerCache,
    ModelShape,
)
from vllm_ascend_kvcompress.methods.triattention.config import TriAttentionConfig
from vllm_ascend_kvcompress.methods.triattention.method import TriAttentionMethod
from vllm_ascend_kvcompress.methods.triattention.scoring import score_post_rope_keys
from vllm_ascend_kvcompress.methods.triattention.selection import (
    _select_v3_batched,
    select_keep_indices,
)


def test_v3_protects_prefix_and_recent_and_spreads_eviction() -> None:
    scores = torch.arange(16, dtype=torch.float32)

    keep = select_keep_indices(
        scores,
        budget=12,
        protected_prefix=2,
        protected_recent=2,
        segments=4,
        policy="v3",
    )

    torch.testing.assert_close(keep[:2], torch.tensor([0, 1]))
    torch.testing.assert_close(keep[-2:], torch.tensor([14, 15]))
    evicted = set(range(16)) - set(keep.tolist())
    assert evicted == {2, 5, 8, 11}


def test_v3_uses_exact_proportional_quota_with_uneven_segments() -> None:
    scores = torch.zeros(19)

    keep = select_keep_indices(
        scores,
        budget=13,
        protected_prefix=2,
        protected_recent=2,
        segments=8,
        policy="v3",
    )

    assert keep.numel() == 13
    assert {0, 1, 17, 18}.issubset(set(keep.tolist()))


@pytest.mark.parametrize(
    ("tokens", "budget", "prefix", "recent", "segments"),
    [
        (32768, 8192, 128, 512, 8),
        (32, 17, 0, 0, 4),
        (32, 18, 2, 2, 4),
        (32, 31, 0, 0, 4),
    ],
)
def test_batched_v3_matches_per_segment_selection(
    tokens: int, budget: int, prefix: int, recent: int, segments: int
) -> None:
    torch.manual_seed(28)
    scores = torch.randperm(tokens, dtype=torch.long).float()
    expected = select_keep_indices(
        scores,
        budget=budget,
        protected_prefix=prefix,
        protected_recent=recent,
        segments=segments,
        policy="v3",
    )
    actual = _select_v3_batched(
        scores,
        budget=budget,
        prefix=prefix,
        recent=recent,
        segments=segments,
    )
    assert actual is not None
    torch.testing.assert_close(actual, expected)


def test_batched_v3_defers_uneven_segments_to_existing_path() -> None:
    assert (
        _select_v3_batched(
            torch.arange(19).float(), budget=13, prefix=2, recent=2, segments=8
        )
        is None
    )


def test_global_policy_preserves_legacy_topk_behavior() -> None:
    keep = select_keep_indices(
        torch.tensor([10.0, 9.0, 8.0, 7.0, -5.0, -6.0]),
        budget=4,
        protected_prefix=0,
        protected_recent=2,
        segments=8,
        policy="global",
    )

    torch.testing.assert_close(keep, torch.tensor([0, 1, 4, 5]))


def _bound_method(
    tmp_path, *, layers, stride=1, tp_size=1, auxiliary=False, partial_rope=False
):
    """使用真实统计文件和 CPU 缓存完成绑定，只由 TP 测试替换跨 rank 通信。"""
    generator = torch.Generator().manual_seed(28)
    shape = ModelShape(
        "qwen3_5_moe_text" if auxiliary else "qwen2",
        layers,
        8 * tp_size,
        tp_size,
        4,
        10000.0,
        False,
        rotary_dim=2 if partial_rope else None,
    )
    stats = {}
    for index in range(layers):
        width = shape.effective_rotary_dim // 2
        real = torch.randn(shape.num_attention_heads, width, generator=generator)
        imag = torch.randn(shape.num_attention_heads, width, generator=generator)
        stats[str(index)] = {
            "q_mean_real": real,
            "q_mean_imag": imag,
            "q_abs_mean": torch.sqrt(real.square() + imag.square()) + 0.2,
        }
        if partial_rope:
            stats[str(index)]["q_pass_mean"] = torch.randn(
                shape.num_attention_heads, 2, generator=generator
            )
    path = tmp_path / "stats.pt"
    torch.save({"metadata": {}, "layer_stats": stats}, path)
    config = TriAttentionConfig.from_method_config(
        {
            "stats_path": str(path),
            "kv_budget": 128,
            "protected_recent_window": 0,
            "score_chunk_size": 128,
            "score_layer_stride": stride,
        }
    )
    host = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=tp_size),
        model_config=SimpleNamespace(max_model_len=512),
        speculative_config=(
            SimpleNamespace(method="mtp", num_speculative_tokens=2)
            if auxiliary
            else None
        ),
    )
    caches = [
        LayerCache(
            f"model.layers.{index}.self_attn",
            index,
            torch.randn(4, 128, 1, 4, generator=generator, dtype=torch.float16),
            torch.randn(4, 128, 1, 4, generator=generator, dtype=torch.float16),
        )
        for index in range(layers)
    ]
    if auxiliary:
        caches.append(
            LayerCache(
                "mtp.layers.0.self_attn.attn",
                0,
                torch.randn(4, 128, 1, 4, generator=generator, dtype=torch.float16),
                torch.randn(4, 128, 1, 4, generator=generator, dtype=torch.float16),
            )
        )
    method = TriAttentionMethod(config, host, shape)
    runner = SimpleNamespace(device=torch.device("cpu"))
    assert method.compatibility_reasons(runner) == ()
    method.bind_model_runner(runner, tuple(caches))
    return method, caches


def _reference_layer_scores(method, caches):
    """按完整物理序列归一化，独立于运行时分块和工作区的组织方式。"""
    stats = method.calibration.to_device(
        method.model_shape,
        torch.device("cpu"),
        tensor_parallel_size=method.vllm_config.parallel_config.tensor_parallel_size,
    )
    scores = []
    for cache in caches:
        keys = torch.cat((cache.k_cache[2], cache.k_cache[0]))
        raw = score_post_rope_keys(
            keys,
            stats[cache.layer_index],
            round_start=300,
            offsets=2.0 ** torch.arange(10),
            aggregation="mean",
        )
        normalized = (raw - raw.mean(dim=-1, keepdim=True)) / torch.sqrt(
            raw.var(dim=-1, correction=0, keepdim=True).clamp_min(1e-6)
        )
        scores.append(normalized.amax(dim=(0, 1)))
    return torch.stack(scores)


def _assert_compressed_cache_contents(method, caches, expected_scores):
    keep = torch.sort(torch.topk(expected_scores, 128, sorted=False).indices).values
    originals = [(cache.k_cache.clone(), cache.v_cache.clone()) for cache in caches]
    result = method.compress(
        CompressionRequest("r", 300, 256, torch.tensor([2, 0]), torch.tensor([3]))
    )
    assert result.physical_num_tokens == 128
    for cache, (keys, values) in zip(caches, originals, strict=True):
        expected_k = torch.cat((keys[2], keys[0]))[keep]
        expected_v = torch.cat((values[2], values[0]))[keep]
        torch.testing.assert_close(cache.k_cache[3], expected_k)
        torch.testing.assert_close(cache.v_cache[3], expected_v)
        torch.testing.assert_close(cache.k_cache[:3], keys[:3])
        torch.testing.assert_close(cache.v_cache[:3], values[:3])


@pytest.mark.parametrize("partial_rope", [False, True])
def test_compression_scores_sampled_layers_and_materializes_every_cache(
    tmp_path, partial_rope
):
    method, caches = _bound_method(
        tmp_path, layers=6, stride=2, auxiliary=True, partial_rope=partial_rope
    )
    # 只用目标层 0、2、4 评分；未抽样层和层号同为 0 的 MTP 缓存仍须搬运。
    # 源序列跨两个非连续页，评分分成两块，但标准化必须覆盖完整 256 token。
    scores = _reference_layer_scores(method, [caches[0], caches[2], caches[4]])
    _assert_compressed_cache_contents(method, caches, scores.mean(dim=0))


def test_tp_compression_reduces_each_layer_before_averaging(tmp_path, monkeypatch):
    import torch.distributed as dist
    import vllm.distributed as vllm_distributed

    monkeypatch.setattr(
        vllm_distributed,
        "get_tp_group",
        lambda: SimpleNamespace(rank_in_group=0, device_group="test-group"),
    )
    method, caches = _bound_method(tmp_path, layers=2, tp_size=2)
    local = _reference_layer_scores(method, caches)
    peer = local.roll(47, dims=-1)
    peer_offset = 0

    def all_reduce(scores, *, op, group):
        nonlocal peer_offset
        assert op == dist.ReduceOp.MAX
        # 允许逐层或合并通信；仅模拟另一个 rank 的逐层分数，不约束调用次数。
        contribution = peer.flatten()[peer_offset : peer_offset + scores.numel()]
        peer_offset += scores.numel()
        torch.maximum(scores, contribution.view_as(scores), out=scores)

    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    # 真实绑定和评分也保护 TP=2 下的 GQA 组宽，避免用本地 KV 头数扩大 Query 组。
    _assert_compressed_cache_contents(
        method, caches, torch.maximum(local, peer).mean(dim=0)
    )
