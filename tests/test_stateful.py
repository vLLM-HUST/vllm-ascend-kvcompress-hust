# SPDX-License-Identifier: Apache-2.0

import pickle
from types import SimpleNamespace

import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    MambaSpec,
)

from vllm_ascend_kvcompress.config import ProviderSelection
from vllm_ascend_kvcompress.stateful import (
    SchedulerActiveCompression,
    SchedulerCompressionState,
)
from vllm_ascend_kvcompress.transaction import PLAN_ATTRIBUTE, CompressionPlan


def _selection() -> ProviderSelection:
    return ProviderSelection.from_mapping(
        {
            "method": "triattention",
            "method_config": {
                "stats_path": "/tmp/stats.pt",
                "kv_budget": 128,
                "recompute_window": 128,
                "score_chunk_size": 128,
            },
        }
    )


def _scheduler(
    *,
    scheduler_module="vllm.v1.core.sched.scheduler",
    balance=False,
    block_size=128,
    hybrid_model_type=None,
    attention_block_size=None,
    prefix_caching=False,
    async_scheduling=False,
    mamba_cache_mode="none",
):
    parallel = SimpleNamespace(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
    )
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            enable_prefix_caching=prefix_caching,
            mamba_cache_mode=mamba_cache_mode,
        ),
        model_config=SimpleNamespace(
            is_hybrid=hybrid_model_type is not None,
            hf_text_config=SimpleNamespace(model_type=hybrid_model_type),
        ),
        speculative_config=None,
        kv_transfer_config=None,
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        parallel_config=parallel,
    )
    blocks = [SimpleNamespace(block_id=value) for value in (3, 4, 5)]
    if attention_block_size is None:
        attention_block_size = block_size
    attention_group = KVCacheGroupSpec(
        layer_names=["model.layers.0.self_attn"],
        kv_cache_spec=FullAttentionSpec(
            block_size=attention_block_size,
            num_kv_heads=2,
            head_size=64,
            dtype=torch.float16,
        ),
    )
    cached = []
    single_manager = SimpleNamespace(
        block_size=attention_block_size,
        req_to_blocks={"r": blocks},
        num_cached_block={"r": 0},
        cached=cached,
        cache_blocks=lambda *args, **kwargs: cached.append((args, kwargs)),
    )
    groups = [attention_group]
    single_managers = [single_manager]
    if hybrid_model_type is not None:
        groups.append(
            KVCacheGroupSpec(
                layer_names=["model.layers.1.linear_attn"],
                kv_cache_spec=MambaSpec(
                    block_size=32768,
                    shapes=((1,),),
                    dtypes=(torch.float16,),
                    mamba_cache_mode=mamba_cache_mode,
                ),
            )
        )
        single_managers.append(
            SimpleNamespace(block_size=32768, req_to_blocks={}, num_cached_block={})
        )
    freed = []
    next_block_id = [50]

    def get_new_blocks(count):
        start = next_block_id[0]
        next_block_id[0] += count
        return [
            SimpleNamespace(block_id=value) for value in range(start, start + count)
        ]

    manager = SimpleNamespace(
        coordinator=SimpleNamespace(single_type_managers=tuple(single_managers)),
        block_pool=SimpleNamespace(
            free_blocks=lambda values: freed.extend(values),
            get_num_free_blocks=lambda: 100,
            get_new_blocks=get_new_blocks,
        ),
    )
    request = SimpleNamespace(
        request_id="r",
        num_computed_tokens=300,
        num_prompt_tokens=300,
        max_tokens=128,
        next_decode_eligible_step=7,
    )
    scheduler_name = (
        "AsyncScheduler"
        if scheduler_module.endswith(".async_scheduler")
        else "BalanceScheduler"
        if "vllm_ascend" in scheduler_module
        else "Scheduler"
    )
    scheduler_type = type(scheduler_name, (SimpleNamespace,), {})
    scheduler_type.__module__ = scheduler_module
    scheduler = scheduler_type(
        vllm_config=vllm_config,
        block_size=block_size,
        kv_cache_config=SimpleNamespace(kv_cache_groups=groups),
        kv_cache_manager=manager,
        requests={"r": request},
        finished_req_ids=set(),
        reset_preempted_req_ids=set(),
        _balance_enabled=balance,
    )
    return scheduler, single_manager, freed


def test_compression_plan_survives_worker_message_pickle() -> None:
    from vllm.v1.core.sched.output import SchedulerOutput

    output = SchedulerOutput.make_empty()
    plan = CompressionPlan(300, 128, (3, 4, 5), (50,))
    setattr(output, PLAN_ATTRIBUTE, {"r": plan})

    copied = pickle.loads(pickle.dumps(output))

    assert getattr(copied, PLAN_ATTRIBUTE)["r"] == plan


def test_qwen35_scheduler_separates_alignment_from_attention_pages() -> None:
    scheduler, _, _ = _scheduler(
        block_size=32768,
        hybrid_model_type="qwen3_5_moe_text",
        attention_block_size=2048,
    )
    selection = ProviderSelection.from_mapping(
        {
            "method": "triattention",
            "method_config": {
                "stats_path": "/tmp/stats.pt",
                "kv_budget": 4096,
                "recompute_window": 128,
                "score_chunk_size": 128,
            },
        }
    )

    state = SchedulerCompressionState(scheduler, selection)

    assert state.scheduler_alignment_size == 32768
    assert state.scheduler_block_size == 2048


def test_qwen35_scheduler_rejects_wrong_group_lcm_alignment() -> None:
    scheduler, _, _ = _scheduler(
        block_size=16384,
        hybrid_model_type="qwen3_5_moe_text",
        attention_block_size=2048,
    )
    selection = ProviderSelection.from_mapping(
        {
            "method": "triattention",
            "method_config": {
                "stats_path": "/tmp/stats.pt",
                "kv_budget": 4096,
                "recompute_window": 128,
                "score_chunk_size": 128,
            },
        }
    )

    try:
        SchedulerCompressionState(scheduler, selection)
    except RuntimeError as error:
        assert "LCM of cache-group block sizes" in str(error)
    else:
        raise AssertionError("mismatched scheduler alignment should be rejected")


def test_scheduler_commits_private_blocks_only_after_output_ack() -> None:
    scheduler, manager, freed = _scheduler()
    state = SchedulerCompressionState(scheduler, _selection())
    output = SimpleNamespace(
        num_scheduled_tokens={"r": 44},
        finished_req_ids=set(),
        preempted_req_ids=set(),
    )

    state.after_schedule(output)
    state.before_schedule()
    assert [block.block_id for block in manager.req_to_blocks["r"]] == [3, 4, 5]
    assert not freed
    assert getattr(output, PLAN_ATTRIBUTE)["r"].destination_block_ids == (50,)

    state.after_output(output)
    state.before_schedule()

    assert [block.block_id for block in manager.req_to_blocks["r"]] == [50]
    assert [block.block_id for block in freed] == [5, 4, 3]
    assert state.active["r"] == SchedulerActiveCompression(300, 128)


def test_prefix_cache_keeps_original_blocks_and_never_hashes_private_kv() -> None:
    scheduler, manager, freed = _scheduler(prefix_caching=True)
    shared_prefix = manager.req_to_blocks["r"][0]
    manager.req_to_blocks["other"] = [shared_prefix]
    state = SchedulerCompressionState(scheduler, _selection())
    request = scheduler.requests["r"]
    output = SimpleNamespace(
        num_scheduled_tokens={"r": 44},
        finished_req_ids=set(),
        preempted_req_ids=set(),
    )

    manager.cache_blocks(request, 300)
    assert len(manager.cached) == 1
    state.after_schedule(output)
    state.after_output(output)
    state.before_schedule()
    manager.cache_blocks(request, 301)

    assert [block.block_id for block in manager.req_to_blocks["r"]] == [50]
    assert manager.req_to_blocks["other"] == [shared_prefix]
    assert manager.req_to_blocks["other"][0] is shared_prefix
    assert [block.block_id for block in freed] == [5, 4, 3]
    assert len(manager.cached) == 1


def test_async_request_is_frozen_until_compression_output_ack() -> None:
    scheduler, manager, _ = _scheduler(
        scheduler_module="vllm.v1.core.sched.async_scheduler",
        async_scheduling=True,
    )
    state = SchedulerCompressionState(scheduler, _selection())
    output = SimpleNamespace(
        num_scheduled_tokens={"r": 44},
        finished_req_ids=set(),
        preempted_req_ids=set(),
    )

    state.after_schedule(output)
    assert scheduler.requests["r"].next_decode_eligible_step == 1 << 62
    state.before_schedule()
    assert [block.block_id for block in manager.req_to_blocks["r"]] == [3, 4, 5]

    unrelated = SimpleNamespace(num_scheduled_tokens={"other": 1})
    state.after_output(unrelated)
    state.before_schedule()
    assert [block.block_id for block in manager.req_to_blocks["r"]] == [3, 4, 5]
    assert scheduler.requests["r"].next_decode_eligible_step == 1 << 62

    state.after_output(output)
    state.before_schedule()
    assert [block.block_id for block in manager.req_to_blocks["r"]] == [50]
    assert scheduler.requests["r"].next_decode_eligible_step == 7


def test_hybrid_cache_groups_keep_distinct_length_spaces() -> None:
    scheduler, attention, _ = _scheduler(
        block_size=32768,
        hybrid_model_type="qwen3_5_moe_text",
        attention_block_size=2048,
        prefix_caching=True,
        mamba_cache_mode="align",
    )
    mamba = scheduler.kv_cache_manager.coordinator.single_type_managers[1]
    seen = []

    def record(group, operation):
        def call(*args, **kwargs):
            seen.append((group, operation, args, kwargs))
            return 0 if operation == "count" else []

        return call

    for group, manager in (("attention", attention), ("mamba", mamba)):
        manager.get_num_blocks_to_allocate = record(group, "count")
        manager.allocate_new_blocks = record(group, "allocate")
        manager.remove_skipped_blocks = record(group, "skip")
    state = SchedulerCompressionState(
        scheduler,
        ProviderSelection.from_mapping(
            {
                "method": "triattention",
                "method_config": {
                    "stats_path": "/tmp/stats.pt",
                    "kv_budget": 4096,
                    "recompute_window": 128,
                    "score_chunk_size": 128,
                },
            }
        ),
    )
    state.active["r"] = SchedulerActiveCompression(300, 128)
    for manager in (attention, mamba):
        manager.get_num_blocks_to_allocate("r", 301, [], 300, 300, 301)
        manager.allocate_new_blocks("r", 301, 301)
        manager.remove_skipped_blocks("r", 300, 300)

    assert seen[0][2] == ("r", 129, [], 128, 128, 129)
    assert seen[1][2] == ("r", 129, 129)
    assert seen[2][2] == ("r", 128, 300)
    assert seen[3][2] == ("r", 301, [], 300, 300, 301)
    assert seen[4][2] == ("r", 301, 301)
    assert seen[5][2] == ("r", 300, 300)
    assert scheduler.requests["r"].num_computed_tokens == 300


def test_finished_request_discards_pending_state() -> None:
    scheduler, _, freed = _scheduler()
    state = SchedulerCompressionState(scheduler, _selection())
    state.after_schedule(
        SimpleNamespace(
            num_scheduled_tokens={"r": 44},
            finished_req_ids=set(),
            preempted_req_ids=set(),
        )
    )
    scheduler.finished_req_ids.add("r")

    state.before_schedule()

    assert not state.pending
    assert [block.block_id for block in freed] == [50]


def test_short_decode_request_is_not_armed_for_compression() -> None:
    scheduler, _, _ = _scheduler()
    selection = ProviderSelection.from_mapping(
        {
            "method": "triattention",
            "method_config": {
                "stats_path": "/tmp/stats.pt",
                "kv_budget": 128,
                "recompute_window": 128,
                "score_chunk_size": 128,
                "min_output_tokens_for_compression": 64,
            },
        }
    )
    scheduler.requests["r"].max_tokens = 32
    state = SchedulerCompressionState(scheduler, selection)

    state.after_schedule(
        SimpleNamespace(
            num_scheduled_tokens={"r": 44},
            finished_req_ids=set(),
            preempted_req_ids=set(),
        )
    )

    assert not state.pending


def test_preempted_request_discards_scheduler_state() -> None:
    scheduler, _, _ = _scheduler()
    state = SchedulerCompressionState(scheduler, _selection())
    state.active["r"] = SchedulerActiveCompression(300, 128)

    state.after_schedule(
        SimpleNamespace(
            num_scheduled_tokens={},
            finished_req_ids=set(),
            preempted_req_ids={"r"},
        )
    )

    assert not state.active


def test_current_ascend_balance_scheduler_wrapper_is_supported_when_inert() -> None:
    scheduler, _, _ = _scheduler(
        scheduler_module="vllm_ascend.patch.platform.patch_balance_schedule"
    )

    SchedulerCompressionState(scheduler, _selection())


def test_active_ascend_balance_scheduling_is_rejected() -> None:
    scheduler, _, _ = _scheduler(
        scheduler_module="vllm_ascend.patch.platform.patch_balance_schedule",
        balance=True,
    )

    try:
        SchedulerCompressionState(scheduler, _selection())
    except RuntimeError as error:
        assert "balance scheduling must be disabled" in str(error)
    else:
        raise AssertionError("active balance scheduling should be rejected")
