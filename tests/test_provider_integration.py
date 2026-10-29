# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from runtime_fakes import (
    RecordingBlockTable as _RecordingBlockTable,
)
from runtime_fakes import (
    make_provider as _provider,
)
from runtime_fakes import (
    output_with_plan as _output_with_plan,
)

from vllm_ascend_kvcompress.methods.base import MethodRuntimeSpec
from vllm_ascend_kvcompress.runtime.provider import PendingCompression
from vllm_ascend_kvcompress.runtime.transaction import CompressionAnchor


def test_gdn_metadata_retains_semantic_lengths_after_attention_compaction() -> None:
    @dataclass
    class Metadata:
        num_reqs: int
        seq_lens: torch.Tensor
        _seq_lens_cpu: torch.Tensor
        seq_lens_cpu_upper_bound: torch.Tensor

    provider = _provider()
    seq_lens = torch.tensor([300], dtype=torch.int64)
    cpu_lens = torch.tensor([300], dtype=torch.int64)
    provider.runner = SimpleNamespace(
        seq_lens=seq_lens,
        optimistic_seq_lens_cpu=cpu_lens,
        input_batch=SimpleNamespace(num_reqs=1),
    )
    provider._request_offsets_cpu = torch.tensor([172], dtype=torch.int64)
    provider._request_offsets_device = torch.tensor([172], dtype=torch.int64)
    provider._semantic_seq_lens_device = torch.empty_like(seq_lens)
    provider._semantic_seq_lens_cpu = torch.empty_like(cpu_lens)
    provider._has_active_rows = True

    provider.apply_physical_attention_lengths()
    gdn = provider.semantic_gdn_metadata(Metadata(1, seq_lens, cpu_lens, cpu_lens))

    assert seq_lens.tolist() == [128]
    assert cpu_lens.tolist() == [128]
    assert gdn.seq_lens.tolist() == [300]
    assert gdn._seq_lens_cpu.tolist() == [300]
    assert gdn.seq_lens_cpu_upper_bound.tolist() == [300]


def test_final_prefill_compresses_and_arms_worker_commit() -> None:
    provider = _provider()
    provider.vllm_config.cache_config.enable_prefix_caching = True
    request = SimpleNamespace(
        num_computed_tokens=256,
        num_prompt_tokens=300,
        max_tokens=128,
        block_ids=([2, 0, 3],),
    )
    provider.runner = SimpleNamespace(
        device=torch.device("cpu"), requests={"r": request}
    )

    provider.compress_scheduled_requests(_output_with_plan(300, (2, 0, 3), (50,), 44))

    assert provider.pending["r"] == PendingCompression(300, 128, (50,))
    assert provider.method.last_request is not None
    assert provider.method.last_request.physical_num_tokens == 300
    assert provider.method.last_request.source_block_ids_device.tolist() == [2, 0, 3]
    assert provider.method.last_request.destination_block_ids_device.tolist() == [50]


def test_worker_requires_scheduler_authorization() -> None:
    provider = _provider()
    provider.vllm_config.cache_config.enable_prefix_caching = True
    provider.runner = SimpleNamespace(
        device=torch.device("cpu"),
        requests={
            "r": SimpleNamespace(
                num_computed_tokens=256,
                num_prompt_tokens=300,
                max_tokens=128,
                block_ids=([2, 0, 3],),
            )
        },
    )
    output = SimpleNamespace(num_scheduled_tokens={"r": 44})

    provider.compress_scheduled_requests(output)

    assert not provider.pending
    assert provider.method.last_request is None


def test_worker_commit_preserves_non_attention_group_tables() -> None:
    provider = _provider()
    provider.attention_group_index = 1
    block_table = _RecordingBlockTable()
    recurrent_state_blocks = [11]
    request = SimpleNamespace(block_ids=(recurrent_state_blocks, [2, 0, 3]))
    provider.runner = SimpleNamespace(
        requests={"r": request},
        input_batch=SimpleNamespace(req_id_to_index={"r": 0}, block_table=block_table),
    )
    provider.pending["r"] = PendingCompression(300, 128, (50,))
    output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
        num_scheduled_tokens={"r": 1},
    )

    provider.before_update_states(output)

    assert request.block_ids == (recurrent_state_blocks, [50])
    assert block_table.added == ((recurrent_state_blocks, [50]), 0)
    assert provider.active["r"] == CompressionAnchor(300, 128)
    assert "r" not in provider.pending


def test_worker_waits_for_request_reschedule_before_async_table_switch() -> None:
    provider = _provider()
    block_table = _RecordingBlockTable()
    request = SimpleNamespace(block_ids=([2, 0, 3],))
    provider.runner = SimpleNamespace(
        requests={"r": request},
        input_batch=SimpleNamespace(req_id_to_index={"r": 0}, block_table=block_table),
    )
    provider.pending["r"] = PendingCompression(300, 128, (50,))

    unrelated = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
        num_scheduled_tokens={"other": 1},
    )
    provider.before_update_states(unrelated)
    assert request.block_ids == ([2, 0, 3],)
    assert provider.pending["r"] == PendingCompression(300, 128, (50,))
    assert block_table.added is None

    provider.before_update_states(
        SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=set(),
            scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
            num_scheduled_tokens={"r": 1},
        )
    )
    assert request.block_ids == ([50],)
    assert "r" not in provider.pending


def test_worker_discards_preempted_compression_state() -> None:
    provider = _provider()
    provider.runner = SimpleNamespace(
        requests={},
        input_batch=SimpleNamespace(req_id_to_index={}),
    )
    provider.pending["r"] = PendingCompression(300, 128, (2,))
    provider.active["r"] = CompressionAnchor(300, 128)
    output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids={"r"},
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
        num_scheduled_tokens={},
    )

    provider.before_update_states(output)

    assert "r" not in provider.pending
    assert "r" not in provider.active


def test_physical_positions_and_lengths_preserve_semantic_rope() -> None:
    provider = _provider()
    provider.active["compressed"] = CompressionAnchor(300, 128)
    provider._request_offsets_cpu = torch.tensor([172, 0], dtype=torch.int64)
    provider._request_offsets_device = torch.tensor([172, 0], dtype=torch.int64)
    provider._physical_positions = torch.empty(2, dtype=torch.int64)
    provider._semantic_seq_lens_device = torch.empty(2, dtype=torch.int64)
    provider._semantic_seq_lens_cpu = torch.empty(2, dtype=torch.int64)
    provider._has_active_rows = True
    positions = torch.tensor([300, 20], dtype=torch.int64)
    provider.runner = SimpleNamespace(
        req_indices=SimpleNamespace(gpu=torch.tensor([0, 1], dtype=torch.long)),
        input_batch=SimpleNamespace(num_reqs=2),
        seq_lens=torch.tensor([301, 21], dtype=torch.int64),
        optimistic_seq_lens_cpu=torch.tensor([301, 21], dtype=torch.int64),
    )

    physical = provider.physical_positions_for_slot_mapping(positions)
    provider.apply_physical_attention_lengths()

    torch.testing.assert_close(positions, torch.tensor([300, 20]))
    torch.testing.assert_close(physical, torch.tensor([128, 20]))
    torch.testing.assert_close(provider.runner.seq_lens, torch.tensor([129, 21]))
    torch.testing.assert_close(
        provider.runner.optimistic_seq_lens_cpu, torch.tensor([129, 21])
    )


def test_repeated_compression_uses_current_physical_window() -> None:
    provider = _provider()
    request = SimpleNamespace(
        num_computed_tokens=427,
        num_prompt_tokens=300,
        max_tokens=128,
        block_ids=([5, 9],),
    )
    provider.runner = SimpleNamespace(
        device=torch.device("cpu"), requests={"r": request}
    )
    provider.active["r"] = CompressionAnchor(300, 128)

    provider.compress_scheduled_requests(_output_with_plan(428, (5, 9), (50,), 1))

    assert provider.pending["r"] == PendingCompression(428, 128, (50,))
    assert provider.method.last_request is not None
    assert provider.method.last_request.physical_num_tokens == 256


@pytest.mark.parametrize("cached_sampling_params", [False, True])
def test_worker_output_limit_gates_authorized_compression(
    cached_sampling_params,
) -> None:
    provider = _provider()
    provider.runtime_spec = MethodRuntimeSpec(256, 128, 128, 64)
    request = SimpleNamespace(
        num_computed_tokens=256,
        num_prompt_tokens=300,
        block_ids=([2, 0, 3],),
    )
    # 宿主调度请求直接保存 max_tokens，worker 缓存请求可能只保存 sampling_params。
    limit = SimpleNamespace(max_tokens=63)
    if cached_sampling_params:
        request.sampling_params = limit
    else:
        request.max_tokens = limit.max_tokens
    provider.runner = SimpleNamespace(
        device=torch.device("cpu"), requests={"r": request}
    )

    output = _output_with_plan(300, (2, 0, 3), (50,), 44)
    provider.compress_scheduled_requests(output)

    assert not provider.pending
    assert provider.method.last_request is None
    # 到达门槛后，同一份有效计划应正常执行，防止无授权计划导致测试提前返回。
    if cached_sampling_params:
        limit.max_tokens = 64
    else:
        request.max_tokens = 64
    provider.compress_scheduled_requests(output)
    assert provider.pending["r"] == PendingCompression(300, 128, (50,))
