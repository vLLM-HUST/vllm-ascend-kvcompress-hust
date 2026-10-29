# SPDX-License-Identifier: Apache-2.0
"""覆盖显式选择的已停止维护适配器的迁移行为。"""

import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import pytest
import torch
from runtime_fakes import (
    FakeBackend,
    make_provider,
)
from runtime_fakes import (
    RecordingBlockTable as _RecordingBlockTable,
)
from runtime_fakes import (
    RecordingMethod as _RecordingMethod,
)
from runtime_fakes import (
    output_with_plan as _output_with_plan,
)

from vllm_ascend_kvcompress.extensions.legacy import (
    ATTENTION_QUERY_PROVIDER_ATTRIBUTE,
    LayerActiveCompression,
    LayerCompressionRequest,
    LayerCompressionResult,
    LegacyExtensionBackend,
    LegacyExtensionProvider,
    QueryObservation,
)
from vllm_ascend_kvcompress.extensions.legacy_hooks import (
    install_attention_query_hook,
    install_layer_aware_fia_graph_replay,
)
from vllm_ascend_kvcompress.methods.base import LayerCache
from vllm_ascend_kvcompress.runtime.transaction import PLAN_ATTRIBUTE


# 仅这组测试显式选择迁移适配器；默认事务测试使用通用 provider。
# 因此逐层类型和 Query 状态的覆盖不会变成核心 API 的兼容要求。
class _Backend(FakeBackend, LegacyExtensionBackend):
    def create_method(self):
        return _RecordingMethod()


def _provider():
    return make_provider(_Backend(), LegacyExtensionProvider)


def test_per_layer_result_commits_only_on_acknowledged_request_step() -> None:
    provider = _provider()
    provider.layer_caches += (
        LayerCache("model.layers.1.self_attn", 1, torch.empty(0), torch.empty(0)),
    )
    provider.runner = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(name="NONE")),
        device=torch.device("cpu"),
        requests={
            "r": SimpleNamespace(
                num_computed_tokens=256,
                num_prompt_tokens=300,
                max_tokens=128,
                block_ids=([2, 0, 3],),
            )
        },
        input_batch=SimpleNamespace(
            req_id_to_index={}, block_table=_RecordingBlockTable()
        ),
    )
    provider.method.compress = lambda request: LayerCompressionResult(
        128,
        (("model.layers.1.self_attn", 64), ("model.layers.0.self_attn", 128)),
    )
    provider.compress_scheduled_requests(_output_with_plan(300, (2, 0, 3), (50,), 44))
    expected = (
        ("model.layers.0.self_attn", 128),
        ("model.layers.1.self_attn", 64),
    )
    assert provider.pending["r"].per_layer_physical_num_tokens == expected
    other_step = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
        num_scheduled_tokens={"other": 1},
    )
    provider.before_update_states(other_step)
    assert "r" in provider.pending and "r" not in provider.active

    other_step.num_scheduled_tokens = {"r": 1}
    provider.before_update_states(other_step)
    assert provider.active["r"].per_layer_physical_num_tokens == expected
    assert "r" not in provider.pending

    other_step.num_scheduled_tokens = {}
    other_step.preempted_req_ids = {"r"}
    provider.before_update_states(other_step)
    assert "r" not in provider.active


def test_uniform_per_layer_result_keeps_shared_metadata_path() -> None:
    provider = _provider()
    provider.layer_caches += (
        LayerCache("model.layers.1.self_attn", 1, torch.empty(0), torch.empty(0)),
    )
    assert (
        provider._validate_per_layer_lengths(
            (("model.layers.1.self_attn", 128), ("model.layers.0.self_attn", 128)),
            128,
        )
        is None
    )


def test_per_layer_eager_metadata_uses_distinct_write_slots_and_lengths() -> None:
    @dataclass
    class AscendMetadata:
        seq_lens: torch.Tensor
        seq_lens_cpu: torch.Tensor
        seq_lens_list: list[int]
        slot_mapping: torch.Tensor

    AscendMetadata.__module__ = "vllm_ascend.attention.attention_v1"

    class BlockTable:
        def __init__(self):
            self.slot_mapping = SimpleNamespace(gpu=torch.tensor([12928, 2580]))

        def native(self, num_reqs, query_start_loc, positions):
            assert num_reqs == 2
            assert query_start_loc.tolist() == [0, 1, 2]
            # 请求 0 使用两个私有的 128-token 页，请求 1 使用一个。
            blocks = ((100, 101), (20,))
            for token, position in enumerate(positions.tolist()):
                request = token
                self.slot_mapping.gpu[token] = (
                    blocks[request][position // 128] * 128 + position % 128
                )

    BlockTable._ascend_kvcompress_patch_v4_slot_mapping_original = BlockTable.native
    table = BlockTable()
    provider = _provider()
    provider.layer_caches += (
        LayerCache("model.layers.1.self_attn", 1, torch.empty(0), torch.empty(0)),
    )
    provider.active["compressed"] = LayerActiveCompression(
        300,
        128,
        (("model.layers.0.self_attn", 128), ("model.layers.1.self_attn", 64)),
    )
    provider._has_per_layer_rows = True
    provider._physical_lengths_applied = True
    provider._request_offsets_cpu = torch.tensor([172, 0])
    provider.runner = SimpleNamespace(
        max_num_tokens=4,
        compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(name="NONE")),
        input_batch=SimpleNamespace(
            req_ids=["compressed", "plain"],
            block_table=SimpleNamespace(block_tables=[table]),
        ),
        positions=torch.tensor([300, 20]),
        req_indices=SimpleNamespace(gpu=torch.tensor([0, 1])),
        query_start_loc=SimpleNamespace(gpu=torch.tensor([0, 1, 2])),
    )
    shared = AscendMetadata(
        torch.tensor([129, 21]),
        torch.tensor([129, 21]),
        [129, 21],
        table.slot_mapping.gpu,
    )
    metadata, common = provider.per_layer_attention_metadata(
        ({layer.name: shared for layer in provider.layer_caches}, "common"), 2, 2
    )
    first, second = (metadata[layer.name] for layer in provider.layer_caches)
    assert common == "common"
    assert first is not shared
    assert first.seq_lens_list == [129, 21]
    first_slot_address = first.slot_mapping.data_ptr()
    second_slot_address = second.slot_mapping.data_ptr()
    assert second.seq_lens_list == [65, 21]
    assert second.seq_lens.tolist() == [65, 21]
    assert second.slot_mapping.tolist() == [100 * 128 + 64, 20 * 128 + 20]
    assert table.slot_mapping.gpu.tolist() == [12928, 2580]
    for slots in provider._per_layer_slot_buffers.values():
        assert slots[2:].tolist() == [-1, -1]
        slots[2:].fill_(777)

    provider.runner.positions[0] = 305
    table.slot_mapping.gpu[0] = 12933
    next_shared = AscendMetadata(
        torch.tensor([134, 21]),
        torch.tensor([134, 21]),
        [134, 21],
        table.slot_mapping.gpu,
    )
    next_metadata, _ = provider.per_layer_attention_metadata(
        ({layer.name: next_shared for layer in provider.layer_caches}, None), 2, 2
    )
    assert (
        next_metadata["model.layers.0.self_attn"].slot_mapping.data_ptr()
        == first_slot_address
    )
    assert (
        next_metadata["model.layers.1.self_attn"].slot_mapping.data_ptr()
        == second_slot_address
    )
    assert next_metadata["model.layers.1.self_attn"].seq_lens_list == [70, 21]
    assert next_metadata["model.layers.1.self_attn"].slot_mapping.tolist() == [
        100 * 128 + 69,
        20 * 128 + 20,
    ]
    assert table.slot_mapping.gpu.tolist() == [12933, 2580]
    for slots in provider._per_layer_slot_buffers.values():
        assert slots[2:].tolist() == [-1, -1]


def test_undeclared_nonuniform_lengths_fail_closed_under_graph_replay() -> None:
    provider = _provider()
    provider.layer_caches += (
        LayerCache("model.layers.1.self_attn", 1, torch.empty(0), torch.empty(0)),
    )
    provider.runner = SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(name="FULL_AND_PIECEWISE")
        )
    )
    with pytest.raises(RuntimeError, match="declare per-layer physical state"):
        provider._validate_per_layer_lengths(
            (("model.layers.0.self_attn", 128), ("model.layers.1.self_attn", 64)),
            128,
        )


def test_declared_per_layer_state_accepts_layer_aware_full_graph(
    monkeypatch,
) -> None:
    class AscendAttentionBackendImpl:
        _ascend_kvcompress_layer_aware_graph_v1 = True

    ascend = ModuleType("vllm_ascend")
    attention = ModuleType("vllm_ascend.attention")
    attention_v1 = ModuleType("vllm_ascend.attention.attention_v1")
    attention_v1.AscendAttentionBackendImpl = AscendAttentionBackendImpl
    monkeypatch.setitem(sys.modules, "vllm_ascend", ascend)
    monkeypatch.setitem(sys.modules, "vllm_ascend.attention", attention)
    monkeypatch.setitem(sys.modules, "vllm_ascend.attention.attention_v1", attention_v1)

    provider = _provider()
    provider.requires_per_layer_physical_state = True
    provider.runner = SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(name="FULL_AND_PIECEWISE")
        ),
        use_dcp=False,
        pcp_enabled=False,
        use_async_spec_decode=True,
    )
    provider.vllm_config.speculative_config = None

    provider._validate_per_layer_host()


def test_per_layer_full_graph_capture_accepts_synthetic_empty_batch() -> None:
    @dataclass
    class Metadata:
        slot_mapping: torch.Tensor

    provider = _provider()
    provider.requires_per_layer_physical_state = True
    provider.runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=[], num_reqs=0),
    )
    provider.layer_caches += (
        LayerCache("mtp.layers.0.self_attn", 0, torch.empty(0), torch.empty(0)),
    )
    provider.speculative_cache_layer = provider.layer_caches[-1]
    provider._per_layer_slot_buffers = {
        layer.name: torch.empty(48, dtype=torch.int64)
        for layer in provider.layer_caches
    }
    shared = Metadata(torch.arange(48))
    metadata, common = provider.per_layer_attention_metadata(
        ({layer.name: shared for layer in provider.layer_caches}, shared), 48, 48
    )
    captured_slots = [
        metadata[layer.name].slot_mapping for layer in provider.layer_caches
    ]
    assert all(
        slots.data_ptr() != shared.slot_mapping.data_ptr() for slots in captured_slots
    )
    assert captured_slots[0].data_ptr() != captured_slots[1].data_ptr()
    assert common.slot_mapping.data_ptr() == captured_slots[1].data_ptr()
    # 已捕获的缓存写入内核持有此地址；回放时必须独立更新其内容，
    # 以支持各层物理长度不同的场景。
    for index, layer in enumerate(provider.layer_caches):
        provider._per_layer_slot_buffers[layer.name][0] = 64 + index
    assert [int(slots[0]) for slots in captured_slots] == [64, 65]
    assert int(shared.slot_mapping[0]) == 0
    assert provider._per_layer_metadata_seen


def test_query_observation_can_exclude_auxiliary_mtp_cache() -> None:
    provider = _provider()
    provider.layer_caches += (
        LayerCache("mtp.layers.0.self_attn.attn", 0, torch.empty(0), torch.empty(0)),
        LayerCache("model.layers.3.self_attn", 3, torch.empty(0), torch.empty(0)),
    )
    provider.backend.query_layer_indices = (3,)
    provider.query_window_tokens = 2

    selected = provider._resolve_query_layer_caches()

    assert tuple(cache.name for cache in selected) == ("model.layers.3.self_attn",)


def test_mtp_rejected_draft_reuses_layer_specific_physical_slots() -> None:
    @dataclass
    class AscendMetadata:
        seq_lens: torch.Tensor
        seq_lens_cpu: torch.Tensor
        seq_lens_list: list[int]
        slot_mapping: torch.Tensor

    AscendMetadata.__module__ = "vllm_ascend.attention.attention_v1"

    @dataclass
    class CommonMetadata:
        seq_lens: torch.Tensor
        _seq_lens_cpu: torch.Tensor
        seq_lens_cpu_upper_bound: torch.Tensor
        slot_mapping: torch.Tensor
        seq_lens_cpu: torch.Tensor | None = None
        num_computed_tokens_cpu: torch.Tensor | None = None

    class BlockTable:
        def __init__(self) -> None:
            self.slot_mapping = SimpleNamespace(
                gpu=torch.tensor([128, 129, 130], dtype=torch.int64)
            )

        def native(self, num_reqs, query_start_loc, positions):
            assert num_reqs == 1
            assert int(query_start_loc[-1]) == len(positions)
            for token, position in enumerate(positions.tolist()):
                self.slot_mapping.gpu[token] = position

    BlockTable._ascend_kvcompress_patch_v4_slot_mapping_original = BlockTable.native
    table = BlockTable()
    provider = _provider()
    provider.layer_caches += (
        LayerCache("model.layers.1.self_attn", 1, torch.empty(0), torch.empty(0)),
    )
    provider.active["r"] = LayerActiveCompression(
        300,
        128,
        (("model.layers.0.self_attn", 128), ("model.layers.1.self_attn", 64)),
    )
    provider._has_per_layer_rows = True
    provider._physical_lengths_applied = True
    provider._request_offsets_cpu = torch.tensor([172])
    provider.speculative_cache_layer = provider.layer_caches[1]
    provider.runner = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(name="NONE")),
        input_batch=SimpleNamespace(
            req_ids=["r"],
            block_table=SimpleNamespace(block_tables=[table]),
        ),
        positions=torch.tensor([300, 301, 302]),
        req_indices=SimpleNamespace(gpu=torch.tensor([0, 0, 0])),
        query_start_loc=SimpleNamespace(gpu=torch.tensor([0, 3])),
    )
    first = AscendMetadata(
        torch.tensor([131]),
        torch.tensor([131]),
        [131],
        table.slot_mapping.gpu,
    )
    first_common = CommonMetadata(
        torch.tensor([131]),
        torch.tensor([131]),
        torch.tensor([131]),
        table.slot_mapping.gpu,
        torch.tensor([131]),
        torch.tensor([300]),
    )
    metadata, common = provider.per_layer_attention_metadata(
        ({layer.name: first for layer in provider.layer_caches}, first_common),
        3,
        1,
    )
    assert metadata["model.layers.1.self_attn"].slot_mapping.tolist() == [64, 65, 66]
    assert common.seq_lens.tolist() == [67]
    assert common._seq_lens_cpu.tolist() == [67]
    assert common.num_computed_tokens_cpu.tolist() == [64]
    assert common.slot_mapping.tolist() == [64, 65, 66]

    # 仅第一个推测 token 被接受。下一轮会再次写入语义位置 301，
    # 必须覆盖被拒绝的草稿槽位。
    provider.runner.positions = torch.tensor([301])
    provider.runner.req_indices.gpu = torch.tensor([0])
    provider.runner.query_start_loc.gpu = torch.tensor([0, 1])
    table.slot_mapping.gpu[0] = 129
    second = AscendMetadata(
        torch.tensor([130]),
        torch.tensor([130]),
        [130],
        table.slot_mapping.gpu[:1],
    )
    second_common = CommonMetadata(
        torch.tensor([130]),
        torch.tensor([130]),
        torch.tensor([130]),
        table.slot_mapping.gpu[:1],
        torch.tensor([130]),
        torch.tensor([301]),
    )
    metadata, common = provider.per_layer_attention_metadata(
        ({layer.name: second for layer in provider.layer_caches}, second_common),
        1,
        1,
    )
    assert metadata["model.layers.1.self_attn"].slot_mapping.tolist() == [65]
    assert metadata["model.layers.1.self_attn"].seq_lens_list == [66]
    assert common.seq_lens.tolist() == [66]
    assert common.num_computed_tokens_cpu.tolist() == [65]
    assert common.slot_mapping.tolist() == [65]


def test_per_layer_metadata_bypass_fails_closed() -> None:
    provider = _provider()
    provider._has_per_layer_rows = True
    provider.begin_per_layer_step(SimpleNamespace(num_scheduled_tokens={"r": 1}))
    with pytest.raises(RuntimeError, match="metadata hook was bypassed"):
        provider.finish_per_layer_step()


def test_per_layer_metadata_allows_async_empty_execution_step() -> None:
    provider = _provider()
    provider._has_per_layer_rows = True

    provider.begin_per_layer_step(SimpleNamespace(num_scheduled_tokens={}))
    provider.finish_per_layer_step()

    assert not provider._per_layer_step_requires_metadata


class _ObservingMethod(_RecordingMethod):
    def __init__(self) -> None:
        super().__init__()
        self.buffers: dict[str, torch.Tensor] = {}
        self.counts: dict[str, int] = {}
        self.observation: QueryObservation | None = None
        self.observed_rows: torch.Tensor | None = None
        self.discarded: list[str] = []

    def capture_query(self, layer, query, spans) -> None:
        assert layer.layer_index == 0
        for span in spans:
            buffer = self.buffers.setdefault(span.request_id, torch.empty(2, 2))
            rows = query[span.start : span.end]
            count = self.counts.get(span.request_id, 0)
            if count < 2:
                copy = min(2 - count, len(rows))
                buffer[count : count + copy].copy_(rows[:copy])
                count += copy
                rows = rows[copy:]
            for row in rows:
                buffer[:-1].copy_(buffer[1:].clone())
                buffer[-1].copy_(row)
            self.counts[span.request_id] = count

    def complete_query_observation(self, observation: QueryObservation) -> None:
        assert observation.request_id == "r"
        assert observation.window_tokens == 2
        assert observation.layer_indices == (0,)
        assert self.counts["r"] == 2
        self.observation = observation
        self.observed_rows = self.buffers["r"].clone()

    def compress(self, request: LayerCompressionRequest) -> LayerCompressionResult:
        assert self.observation is not None
        assert request.plan is self.observation.plan
        self.last_request = request
        return LayerCompressionResult(128)

    def discard_query_observation(self, request_id: str) -> None:
        self.discarded.append(request_id)
        self.buffers.pop(request_id, None)
        self.counts.pop(request_id, None)


def _observing_provider() -> LegacyExtensionProvider:
    provider = _provider()
    provider.method = _ObservingMethod()
    provider.query_window_tokens = 2
    provider.query_layer_caches = provider.layer_caches
    request = SimpleNamespace(
        num_computed_tokens=298,
        num_prompt_tokens=300,
        max_tokens=128,
        block_ids=([2, 0, 3],),
    )
    provider.runner = SimpleNamespace(
        device=torch.device("cpu"),
        requests={"r": request},
        input_batch=SimpleNamespace(req_id_to_index={"r": 0}),
        query_start_loc=SimpleNamespace(cpu=torch.tensor([0, 1])),
    )
    return provider


def test_query_observation_spans_chunked_prefill_and_matches_plan() -> None:
    provider = _observing_provider()
    first = SimpleNamespace(num_scheduled_tokens={"r": 1})

    provider.begin_query_step(first)
    provider.capture_attention_query(
        "model.layers.0.self_attn", torch.tensor([[1.0, 2.0]])
    )
    provider.finish_query_step()
    stable_address = provider.method.buffers["r"].data_ptr()

    provider.runner.requests["r"].num_computed_tokens = 299
    final = _output_with_plan(300, (2, 0, 3), (50,), 1)
    provider.begin_query_step(final)
    provider.capture_attention_query(
        "model.layers.0.self_attn", torch.tensor([[3.0, 4.0]])
    )
    provider.finish_query_step()
    assert provider.method.buffers["r"].data_ptr() == stable_address
    provider.compress_scheduled_requests(final)

    torch.testing.assert_close(
        provider.method.observed_rows,
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )
    assert provider.method.observation.plan is getattr(final, PLAN_ATTRIBUTE)["r"]
    assert provider.method.discarded == ["r"]
    assert provider.method.buffers == {}


def test_query_observation_fails_closed_when_forward_hook_is_bypassed() -> None:
    provider = _observing_provider()
    output = SimpleNamespace(num_scheduled_tokens={"r": 1})

    provider.begin_query_step(output)
    with pytest.raises(RuntimeError, match="graph replay path is unvalidated"):
        provider.finish_query_step()
    provider.abort_query_step(output)

    assert not provider._query_forward_complete
    assert provider.method.discarded == ["r"]


def test_query_observation_is_discarded_on_preemption() -> None:
    provider = _observing_provider()
    output = SimpleNamespace(num_scheduled_tokens={"r": 1})
    provider.begin_query_step(output)
    provider.capture_attention_query(
        "model.layers.0.self_attn", torch.tensor([[1.0, 2.0]])
    )
    provider.finish_query_step()

    provider.before_update_states(
        SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids={"r"},
            scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
            num_scheduled_tokens={},
        )
    )

    assert provider.method.buffers == {}
    assert provider.method.discarded == ["r"]


def test_layer_aware_graph_replay_uses_validated_host_seam(monkeypatch) -> None:
    class Impl:
        def _graph_metadata_layer_name(self):
            return "layer"

        @staticmethod
        def update_graph_params():
            layer_name = "layer"
            metadata_key = layer_name
            attn_metadata = {metadata_key: object()}
            return attn_metadata

        def full_graph_fia(self):
            return None

    utils = ModuleType("vllm_ascend.attention.utils")
    attention_v1 = ModuleType("vllm_ascend.attention.attention_v1")

    def original_probe():
        return False

    utils.needs_layer_aware_fia_graph_replay = original_probe
    attention_v1.needs_layer_aware_fia_graph_replay = original_probe
    attention_v1.AscendAttentionBackendImpl = Impl
    attention = ModuleType("vllm_ascend.attention")
    attention.attention_v1 = attention_v1
    attention.utils = utils
    ascend = ModuleType("vllm_ascend")
    ascend.attention = attention
    monkeypatch.setitem(sys.modules, "vllm_ascend", ascend)
    monkeypatch.setitem(sys.modules, "vllm_ascend.attention", attention)
    monkeypatch.setitem(sys.modules, "vllm_ascend.attention.attention_v1", attention_v1)
    monkeypatch.setitem(sys.modules, "vllm_ascend.attention.utils", utils)
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(name="FULL_AND_PIECEWISE")
        )
    )

    assert install_layer_aware_fia_graph_replay(config)
    assert utils.needs_layer_aware_fia_graph_replay()
    assert attention_v1.needs_layer_aware_fia_graph_replay()
    assert Impl._ascend_kvcompress_layer_aware_graph_v1
    assert not install_layer_aware_fia_graph_replay(config)


def test_attention_query_hook_runs_at_custom_op_backend_boundary(
    monkeypatch,
) -> None:
    calls = []

    class Impl:
        def forward(self, layer, query, suffix):
            calls.append(("native", layer.layer_name, query, suffix))
            return "output"

    attention_v1 = ModuleType("vllm_ascend.attention.attention_v1")
    attention_v1.AscendAttentionBackendImpl = Impl
    monkeypatch.setitem(sys.modules, "vllm_ascend.attention.attention_v1", attention_v1)
    provider = SimpleNamespace(
        capture_attention_query=lambda name, query: calls.append(
            ("observe", name, query)
        )
    )
    layer = SimpleNamespace(layer_name="model.layers.3.self_attn")
    setattr(layer, ATTENTION_QUERY_PROVIDER_ATTRIBUTE, provider)

    install_attention_query_hook()
    result = Impl().forward(layer, "query", "suffix")

    assert result == "output"
    assert calls == [
        ("observe", "model.layers.3.self_attn", "query"),
        ("native", "model.layers.3.self_attn", "query", "suffix"),
    ]
