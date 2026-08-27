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
    stats = SimpleNamespace(q_mean_real=torch.zeros((1, 1, 1)))
    method.layer_caches = (
        TriAttentionLayerCache(
            name="layer.0",
            k_cache=torch.empty(0),
            v_cache=torch.empty(0),
            stats=stats,
        ),
    )
    method.offsets = torch.tensor([1.0])
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

    keep = method._select_keep_indices(torch.tensor([0, 1]), 256)

    assert score_calls == 2
    assert keep.shape == (128,)
