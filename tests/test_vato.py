# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend_kvcompress.methods import available_methods, create_method
from vllm_ascend_kvcompress.methods.base import (
    CompressionRequest,
    LayerCache,
    ModelShape,
)


def _shape():
    return ModelShape("qwen2", 1, 4, 2, 2, 10_000.0, False)


def _config(*, eager=True, tp=1, dbo=False):
    return SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=eager, max_model_len=1024),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp, enable_dbo=dbo),
    )


def _method(**options):
    return create_method(
        "vato",
        {"kv_budget": 128, "window_size": 4, "sink_size": 4, **options},
        _config(),
        _shape(),
    )


def _bound_method():
    method = _method()
    # Source logical order is blocks 2, 0. Destination overlaps block 2.
    keys = torch.zeros(4, 128, 2, 2)
    values = torch.zeros_like(keys)
    positions = torch.arange(256, dtype=torch.float32)
    for block, part in zip((2, 0), positions.split(128), strict=True):
        keys[block, :, :, 0] = part[:, None]
        keys[block, :, :, 1] = part[:, None] + 1000
        values[block, :, 0, 0] = part
        values[block, :, 1, 0] = -part
    layer = LayerCache("layer", 0, keys, values)
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        compilation_config=SimpleNamespace(
            static_forward_context={"layer": torch.nn.Identity()}
        ),
        input_batch=SimpleNamespace(num_reqs=1, req_ids=["r"]),
        query_start_loc=SimpleNamespace(cpu=torch.tensor([0, 4])),
        requests={"r": SimpleNamespace(num_computed_tokens=252)},
    )
    method.bind_model_runner(runner, (layer,))
    return method, runner, layer


def _request(semantic=256, physical=256):
    return CompressionRequest(
        "r",
        semantic,
        physical,
        ((2, 0),),
        ((2,),),
        torch.tensor([2, 0]),
        torch.tensor([2]),
    )


def test_vato_registers_without_loading_calibration():
    assert "vato" in available_methods()
    method = _method()
    assert method.runtime_spec.compression_threshold_tokens == 256
    assert method.runtime_spec.max_physical_num_tokens == 128


@pytest.mark.parametrize(
    "options",
    [
        {"stats_path": "unused.pt"},
        {"kv_budget": 129},
        {"recompute_window": 0},
        {"window_size": 0},
        {"window_size": True},
        {"sink_size": -1},
        {"window_size": 125},
        {"kernel_size": 2},
        {"variant": "gram"},
        {"score_chunk_size": 0},
        {"min_output_tokens_for_compression": -1},
    ],
)
def test_vato_rejects_invalid_options(options):
    with pytest.raises(ValueError):
        _method(**options)


def test_vato_requires_eager_and_no_microbatching():
    method = create_method("vato", {}, _config(eager=False, dbo=True), _shape())
    reasons = " ".join(method.compatibility_reasons(SimpleNamespace()))
    assert "enforce-eager" in reasons
    assert "DBO" in reasons


@pytest.mark.parametrize(
    "variant", ["dot", "abs", "cosine", "centered", "centered_norm"]
)
@pytest.mark.parametrize("chunk_size", [1, 3, 128])
def test_paged_scores_match_dense_formula(variant, chunk_size):
    from vllm_ascend_kvcompress.methods.vato.scoring import score_paged_values

    generator = torch.Generator().manual_seed(51)
    cache = torch.randn(4, 128, 2, 4, generator=generator)
    cache[2, 3] = 0  # Norm variants must handle a zero value.
    observation = torch.randn(2, 4, generator=generator)
    values = torch.cat([cache[2], cache[0]])[:173].transpose(0, 1)
    centered = values - values.mean(dim=1, keepdim=True)
    dot = torch.einsum("htd,hd->ht", values, observation)
    norm = values.norm(dim=-1).clamp_min(1e-8)
    expected = {
        "dot": dot,
        "abs": dot.abs(),
        "cosine": dot / norm,
        "centered": torch.einsum("htd,hd->ht", centered, observation),
        "centered_norm": torch.einsum("htd,hd->ht", centered, observation) * norm,
    }[variant]
    actual = score_paged_values(
        cache,
        torch.tensor([2, 0]),
        observation,
        num_tokens=173,
        chunk_size=chunk_size,
        variant=variant,
    )
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def test_selection_preserves_sink_recent_and_independent_heads():
    from vllm_ascend_kvcompress.methods.vato.scoring import select_keep_indices

    scores = torch.tensor(
        [[0.0, 2.0, 4.0, 1.0, 3.0, 0.0], [0.0, 4.0, 2.0, 3.0, 1.0, 0.0]]
    )
    keep = select_keep_indices(
        scores, budget=4, sink_size=1, window_size=1, kernel_size=1
    )
    assert keep.tolist() == [[0, 2, 4, 5], [0, 1, 3, 5]]


def test_output_windows_follow_request_order_and_semantic_continuity():
    from vllm_ascend_kvcompress.methods.vato.observation import OutputObserver

    runner = SimpleNamespace(
        input_batch=SimpleNamespace(num_reqs=2, req_ids=["b", "a"]),
        query_start_loc=SimpleNamespace(cpu=torch.tensor([0, 2, 5, 8])),
        requests={
            "a": SimpleNamespace(num_computed_tokens=10),
            "b": SimpleNamespace(num_computed_tokens=20),
        },
    )
    observer = OutputObserver(
        runner, window_size=4, num_kv_heads=2, head_dim=2, num_query_heads=4
    )
    output = torch.arange(64.0).view(8, 4, 2)
    observer.observe("layer", output.flatten(1))
    # a owns rows 2..4, GQA sums head pairs; padded rows 5..7 are ignored.
    torch.testing.assert_close(
        observer.mean("layer", "a", 13), torch.tensor([[50.0, 52.0], [58.0, 60.0]])
    )
    runner.input_batch.req_ids = ["a", "b"]
    runner.query_start_loc.cpu = torch.tensor([0, 2, 3])
    runner.requests["a"].num_computed_tokens = 13
    runner.requests["b"].num_computed_tokens = 22
    observer.observe("layer", torch.ones(3, 8))
    torch.testing.assert_close(
        observer.mean("layer", "a", 15), torch.tensor([[30.0, 31.0], [34.0, 35.0]])
    )
    observer.reset_requests({"a"})
    with pytest.raises(RuntimeError, match="observation"):
        observer.mean("layer", "a", 15)
    with pytest.raises(RuntimeError, match="observation"):
        observer.mean("layer", "b", 24)


def test_vato_materializes_each_head_with_overlapping_source_and_destination():
    method, runner, layer = _bound_method()
    method.observer.observe("layer", torch.ones(4, 8))
    result = method.compress(_request())
    assert result.physical_num_tokens == 128
    assert result.per_layer_physical_num_tokens == (("layer", 128),)
    expected = torch.tensor(
        [
            list(range(4)) + list(range(132, 256)),
            list(range(124)) + list(range(252, 256)),
        ],
        dtype=torch.float32,
    ).T
    torch.testing.assert_close(layer.k_cache[2, :, :, 0], expected)
    torch.testing.assert_close(layer.v_cache[2, :, 0, 0], expected[:, 0])
    torch.testing.assert_close(layer.v_cache[2, :, 1, 0], -expected[:, 1])


def test_missing_or_stale_observation_fails_before_cache_write():
    method, runner, layer = _bound_method()
    before = layer.k_cache.clone()
    with pytest.raises(RuntimeError, match="observation"):
        method.compress(_request())
    method.observer.observe("layer", torch.ones(4, 8))
    with pytest.raises(RuntimeError, match="observation"):
        method.compress(_request(semantic=257))
    torch.testing.assert_close(layer.k_cache, before)
