# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from vllm_ascend_kvcompress.stateful import (
    _install_scheduler_hook,
    _validate_repeated_plan,
)


def test_scheduler_arms_repeated_compression_at_physical_threshold() -> None:
    output = SimpleNamespace(
        num_scheduled_tokens={"r": 1},
        kv_cache_compression_transaction_ids=None,
        kv_cache_compression_destination_block_ids=None,
    )

    class Scheduler:
        def schedule(self):
            return output

        def _arm_kv_cache_compression_transaction(self, request_id, transactions):
            transactions[request_id] = 9
            self._inflight_kv_cache_compression_transactions[request_id] = 9

    _install_scheduler_hook(Scheduler)
    scheduler = Scheduler()
    scheduler.kv_cache_compression_runtime_spec = SimpleNamespace(
        compression_threshold_tokens=256
    )
    scheduler.kv_cache_manager = SimpleNamespace(
        enable_caching=False,
        get_compressed_physical_num_tokens=lambda request_id: 256,
    )
    scheduler._inflight_kv_cache_compression_transactions = {}
    scheduler.requests = {"r": object()}

    result = scheduler.schedule()

    assert result.kv_cache_compression_transaction_ids == {"r": 9}


def test_repeated_plan_validates_live_physical_block_table() -> None:
    validations = []
    coordinator = SimpleNamespace(
        validate_request_tail_truncation=lambda *args: validations.append(args)
    )
    manager = SimpleNamespace(
        kv_cache_compression_config=SimpleNamespace(
            schema_version=1, provider="ascend_kvcompress"
        ),
        kv_cache_compression_runtime_spec=SimpleNamespace(
            compression_threshold_tokens=256,
            max_physical_num_tokens=128,
        ),
        _compressed_request_physical_tokens={"r": 256},
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(layer_names=["layer.0"])]
        ),
        enable_caching=False,
        coordinator=coordinator,
    )
    request = SimpleNamespace(request_id="r", num_computed_tokens=428)
    plan = SimpleNamespace(
        schema_version=1,
        provider="ascend_kvcompress",
        request_id="r",
        semantic_num_tokens=428,
        physical_num_tokens=128,
        per_layer_physical_num_tokens=(("layer.0", 128),),
        expected_block_ids=((5, 9),),
    )

    assert _validate_repeated_plan(manager, request, plan) == 1
    assert validations == [("r", 1, ((5, 9),))]
