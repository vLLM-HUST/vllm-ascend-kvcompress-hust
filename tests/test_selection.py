# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import torch

from vllm_ascend_kvcompress.methods.triattention import (
    TriAttentionConfig,
    TriAttentionLayerCache,
    TriAttentionMethod,
)
from vllm_ascend_kvcompress.methods.triattention import method as triattention_method


def test_protected_window_and_sorted_topk_contract() -> None:
    scores = torch.tensor([10.0, 9.0, 8.0, 7.0, -5.0, -6.0])
    budget = 4
    protected = 2
    scores[-protected:] = float("inf")
    keep = torch.sort(
        torch.topk(scores, k=budget, largest=True, sorted=False).indices
    ).values
    torch.testing.assert_close(keep, torch.tensor([0, 1, 4, 5]))


def test_selection_reuses_first_pass_score_chunks(monkeypatch) -> None:
    method = object.__new__(TriAttentionMethod)
    method.config = TriAttentionConfig(
        stats_path=Path("unused.pt"),
        kv_budget=128,
        recompute_window=128,
        protected_recent_window=0,
        score_chunk_size=128,
    )
    stats = SimpleNamespace(q_mean_real=torch.zeros((1, 1, 1)), omega=torch.ones(1))
    method.layer_caches = (
        TriAttentionLayerCache(
            name="layer.0",
            k_cache=torch.empty(0),
            v_cache=torch.empty(0),
            stats=stats,
        ),
    )
    method.offsets = torch.tensor([1.0])
    method.score_workspace = torch.empty((1, 1, 256), dtype=torch.float32)
    method.aggregate_workspace = torch.empty(256, dtype=torch.float32)
    score_calls = 0

    def fake_gather(*args, count: int, **kwargs) -> torch.Tensor:
        del args, kwargs
        return torch.zeros((count, 1, 2))

    def fake_score(keys: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        nonlocal score_calls
        del args, kwargs
        score_calls += 1
        return torch.zeros((1, 1, keys.shape[0]))

    monkeypatch.setattr(triattention_method, "gather_paged_range", fake_gather)
    monkeypatch.setattr(triattention_method, "score_post_rope_keys", fake_score)

    keep = method._select_keep_indices(torch.tensor([0, 1]), 256, 256)

    assert score_calls == 2
    assert keep.shape == (128,)


def test_selection_scores_uniform_layer_sample(monkeypatch) -> None:
    method = object.__new__(TriAttentionMethod)
    method.config = TriAttentionConfig(
        stats_path=Path("unused.pt"),
        kv_budget=128,
        recompute_window=128,
        protected_recent_window=0,
        score_chunk_size=128,
        score_layer_stride=2,
    )
    stats = SimpleNamespace(q_mean_real=torch.zeros((1, 1, 1)), omega=torch.ones(1))
    method.layer_caches = tuple(
        TriAttentionLayerCache(
            name=f"layer.{index}",
            k_cache=torch.empty(0),
            v_cache=torch.empty(0),
            stats=stats,
        )
        for index in range(5)
    )
    method.offsets = torch.tensor([1.0])
    method.score_workspace = torch.empty((1, 1, 256), dtype=torch.float32)
    method.aggregate_workspace = torch.empty(256, dtype=torch.float32)
    scored_layers = 0

    def fake_gather(*args, count: int, **kwargs) -> torch.Tensor:
        del args, kwargs
        return torch.zeros((count, 1, 2))

    def fake_score(keys: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        nonlocal scored_layers
        del args, kwargs
        scored_layers += 1
        return torch.zeros((1, 1, keys.shape[0]))

    monkeypatch.setattr(triattention_method, "gather_paged_range", fake_gather)
    monkeypatch.setattr(triattention_method, "score_post_rope_keys", fake_score)

    method._select_keep_indices(torch.tensor([0, 1]), 256, 256)

    assert scored_layers == 6


def test_selection_does_not_add_unsampled_layers(monkeypatch) -> None:
    method = object.__new__(TriAttentionMethod)
    method.config = TriAttentionConfig(
        stats_path=Path("unused.pt"),
        kv_budget=128,
        recompute_window=128,
        protected_recent_window=0,
        score_chunk_size=128,
        score_layer_stride=2,
    )
    stats = SimpleNamespace(q_mean_real=torch.zeros((1, 1, 1)), omega=torch.ones(1))
    method.layer_caches = tuple(
        TriAttentionLayerCache(
            name=f"layer.{index}",
            k_cache=torch.empty(0),
            v_cache=torch.empty(0),
            stats=stats,
        )
        for index in range(6)
    )
    method.offsets = torch.tensor([1.0])
    method.score_workspace = torch.empty((1, 1, 256), dtype=torch.float32)
    method.aggregate_workspace = torch.empty(256, dtype=torch.float32)
    scored_layers = 0

    def fake_gather(*args, count: int, **kwargs) -> torch.Tensor:
        del args, kwargs
        return torch.zeros((count, 1, 2))

    def fake_score(keys: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        nonlocal scored_layers
        del args, kwargs
        scored_layers += 1
        return torch.zeros((1, 1, keys.shape[0]))

    monkeypatch.setattr(triattention_method, "gather_paged_range", fake_gather)
    monkeypatch.setattr(triattention_method, "score_post_rope_keys", fake_score)

    method._select_keep_indices(torch.tensor([0, 1]), 256, 256)

    assert scored_layers == 6


def test_selection_prepares_phases_once_for_all_scoring_layers(monkeypatch) -> None:
    method = object.__new__(TriAttentionMethod)
    method.config = TriAttentionConfig(
        stats_path=Path("unused.pt"),
        kv_budget=128,
        recompute_window=128,
        protected_recent_window=0,
        score_chunk_size=128,
        score_layer_stride=2,
    )
    stats = SimpleNamespace(
        q_mean_real=torch.zeros((1, 1, 1)),
        q_mean_imag=torch.zeros((1, 1, 1)),
        omega=torch.ones(1),
        rope_style="half",
    )
    method.layer_caches = tuple(
        TriAttentionLayerCache(
            name=f"layer.{index}",
            k_cache=torch.empty(0),
            v_cache=torch.empty(0),
            stats=stats,
            frequency_scale=torch.ones((1, 1, 1)),
            extra_coefficient=torch.zeros((1, 1, 1)),
            offset_cos_mean=torch.ones(1),
            offset_sin_mean=torch.zeros(1),
        )
        for index in range(6)
    )
    method.offsets = torch.tensor([1.0])
    method.score_workspace = torch.empty((1, 1, 256), dtype=torch.float32)
    method.aggregate_workspace = torch.empty(256, dtype=torch.float32)
    method.scoring_omega = torch.ones((3, 1))
    method.scoring_offset_cos = torch.ones((3, 1))
    method.scoring_offset_sin = torch.zeros((3, 1))
    method.phase_cos_workspace = torch.empty((3, 1))
    method.phase_sin_workspace = torch.empty((3, 1))
    phase_calls = 0
    score_calls = 0

    def fake_prepare(*args, **kwargs) -> bool:
        nonlocal phase_calls
        del args, kwargs
        phase_calls += 1
        return True

    def fake_score(*args, output: torch.Tensor | None = None, **kwargs) -> bool:
        nonlocal score_calls
        del kwargs
        score_calls += 1
        target = output if output is not None else args[9]
        target.zero_()
        return True

    def fake_aggregate(
        _scores,
        _mean,
        _variance,
        aggregate,
        _num_tokens,
        *,
        layer_aggregation,
        first_layer,
    ) -> bool:
        del layer_aggregation
        if first_layer:
            aggregate.zero_()
        return True

    monkeypatch.setattr(
        triattention_method, "prepare_mean_phase_coefficients", fake_prepare
    )
    monkeypatch.setattr(
        triattention_method, "score_paged_keys_mean_precomputed", fake_score
    )
    monkeypatch.setattr(
        triattention_method, "aggregate_normalized_scores", fake_aggregate
    )
    monkeypatch.setattr(
        triattention_method,
        "gather_paged_range",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("fallback used")),
    )

    keep = method._select_keep_indices(torch.tensor([0, 1]), 256, 256)

    assert phase_calls == 1
    assert score_calls == 3
    assert keep.shape == (128,)
