# SPDX-License-Identifier: Apache-2.0

from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace

import pytest
import torch

import vllm_ascend_kvcompress.plugin as plugin
from vllm_ascend_kvcompress.host_compat import (
    QWEN_GDN_LIST_COMPAT_ENV,
    install_dcp_length_alias,
    install_qwen_gdn_list_compat,
    install_removed_async_output_routing_field,
    install_removed_model_routing_flag,
    install_removed_output_routing_field,
    install_removed_routed_experts_types,
    install_step3p5_fp32_linear_alias,
    qwen_mtp2_cache_binding_bridge,
)
from vllm_ascend_kvcompress.plugin import (
    _configure_message_queue_defaults,
    _install_runner_hooks,
    _install_runtime_slot_mapping_hooks,
    _install_slot_mapping_hook,
    _prepare_current_triton_runtime,
    _RunnerPatchLoader,
)


class _Runner:
    def load_model(self):
        return "loaded"

    def initialize_kv_cache(self, config):
        return config

    def _update_states(self, output):
        return output

    def _build_attention_metadata(self, *args, **kwargs):
        return args, kwargs

    def sample_tokens(self, grammar):
        return grammar


class _BlockTable:
    def compute_slot_mapping(self, num_reqs, query_start_loc, positions):
        self.arguments = num_reqs, query_start_loc, positions


class _WrappedLoader:
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.NPUModelRunner = _Runner


def test_runner_hook_installation_is_idempotent() -> None:
    selection = object()
    _install_runner_hooks(_Runner, selection)
    first_update = _Runner._update_states
    first_sample = _Runner.sample_tokens

    _install_runner_hooks(_Runner, selection)

    assert _Runner._update_states is first_update
    assert _Runner.sample_tokens is first_sample


def test_runner_hook_supplies_removed_xdrope_compatibility_attribute() -> None:
    class Runner:
        def load_model(self):
            return "loaded"

        def initialize_kv_cache(self, config):
            return config

        def _update_states(self, output):
            return output

        def _build_attention_metadata(self, *args, **kwargs):
            return args, kwargs

        def sample_tokens(self, grammar):
            return grammar

    assert not hasattr(Runner, "uses_xdrope_dim")
    _install_runner_hooks(Runner, object())
    assert Runner.uses_xdrope_dim == 0


def test_current_triton_runtime_preloads_gluon_descriptor_namespace(
    monkeypatch,
) -> None:
    imported = []
    modules = (
        "test_triton.experimental",
        "test_triton.experimental.gluon",
        "test_triton.experimental.gluon.language",
        "test_triton.experimental.gluon.nvidia",
    )
    monkeypatch.setattr(plugin, "_TRITON_GLUON_MODULES", modules)
    monkeypatch.setitem(
        plugin.sys.modules, "triton", SimpleNamespace(__version__="3.6.0")
    )
    monkeypatch.setattr(plugin, "import_module", imported.append)

    _prepare_current_triton_runtime()

    assert imported == list(modules)


def test_triton_32_skips_unavailable_gluon_namespace(monkeypatch) -> None:
    imported = []
    monkeypatch.setitem(
        plugin.sys.modules, "triton", SimpleNamespace(__version__="3.2.0")
    )
    monkeypatch.setattr(plugin, "import_module", imported.append)

    _prepare_current_triton_runtime()

    assert imported == []


def test_message_queue_env_override_covers_omitted_worker_default(
    monkeypatch,
) -> None:
    class MessageQueue:
        def __init__(
            self,
            n_reader,
            n_local_reader,
            local_reader_ranks=None,
            max_chunk_bytes=24 * 1024 * 1024,
            max_chunks=10,
            connect_ip=None,
        ):
            pass

    module = ModuleType("fake_shm_broadcast")
    module.MessageQueue = MessageQueue
    monkeypatch.setenv("VLLM_MQ_MAX_CHUNK_BYTES_MB", "4")
    monkeypatch.setattr(plugin, "import_module", lambda name: module)

    _configure_message_queue_defaults()

    assert MessageQueue.__init__.__defaults__ == (
        None,
        4 * 1024 * 1024,
        10,
        None,
    )
    assert MessageQueue.__dict__[plugin._MESSAGE_QUEUE_PATCH_MARKER]


def test_message_queue_env_override_rejects_nonpositive_size(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_MQ_MAX_CHUNK_BYTES_MB", "0")

    with pytest.raises(RuntimeError, match="must be positive"):
        _configure_message_queue_defaults()


def test_qwen_gdn_list_compat_is_explicitly_opt_in(monkeypatch) -> None:
    monkeypatch.delenv(QWEN_GDN_LIST_COMPAT_ENV, raising=False)
    namespace = SimpleNamespace()

    assert not install_qwen_gdn_list_compat(namespace)


def test_qwen_gdn_list_compat_converts_tensor_metadata(monkeypatch) -> None:
    calls = []

    class Operation:
        default = SimpleNamespace(
            _schema=(
                "_C_ascend::npu_causal_conv1d_custom(Tensor output, Tensor x, "
                "Tensor weight, Tensor conv_state, Tensor? bias_opt, "
                "int[] query_start_loc_opt, int[] cache_indices_opt, "
                "int[] initial_state_mode_opt, int[] num_accepted_tokens_opt, "
                "int activation_mode, int pad_slot_id, int run_mode) -> Tensor"
            )
        )

        def __call__(self, *args, **kwargs):
            calls.append((args, kwargs))
            return "output"

    operation = Operation()
    namespace = SimpleNamespace(npu_causal_conv1d_custom=operation)
    monkeypatch.setenv(QWEN_GDN_LIST_COMPAT_ENV, "1")

    assert install_qwen_gdn_list_compat(namespace)
    with torch.inference_mode():
        result = namespace.npu_causal_conv1d_custom(
            "out",
            "x",
            "weight",
            "state",
            None,
            torch.tensor([0, 4], dtype=torch.int32),
            cache_indices_opt=torch.tensor([[7, 8], [9, 10]], dtype=torch.int64),
            initial_state_mode_opt=None,
            num_accepted_tokens_opt=torch.tensor([1], dtype=torch.int32),
            activation_mode=1,
            pad_slot_id=-1,
            run_mode=0,
        )

    assert result == "output"
    assert calls[0][0][5] == [0, 4]
    assert calls[0][1]["cache_indices_opt"] == [7, 8, 9, 10]
    assert calls[0][1]["initial_state_mode_opt"] == []
    assert calls[0][1]["num_accepted_tokens_opt"] == [1]
    assert install_qwen_gdn_list_compat(namespace)


def test_qwen_gdn_mtp2_reference_is_scoped_to_spec_decode(monkeypatch) -> None:
    from vllm_ascend_kvcompress import qwen_gdn_ops

    calls = []

    class Operation:
        default = SimpleNamespace(
            _schema=(
                "_C_ascend::npu_causal_conv1d_custom(Tensor output, Tensor x, "
                "Tensor weight, Tensor conv_state, Tensor? bias_opt, "
                "int[] query_start_loc_opt, int[] cache_indices_opt, "
                "int[] initial_state_mode_opt, int[] num_accepted_tokens_opt, "
                "int activation_mode, int pad_slot_id, int run_mode) -> Tensor"
            )
        )

        def __call__(self, *args, **kwargs):
            calls.append(("native", args, kwargs))
            return "native"

    monkeypatch.setenv(QWEN_GDN_LIST_COMPAT_ENV, "1")
    monkeypatch.setattr(
        qwen_gdn_ops,
        "causal_conv1d_spec_reference",
        lambda *args: calls.append(("reference", args)) or "reference",
    )
    namespace = SimpleNamespace(npu_causal_conv1d_custom=Operation())
    assert install_qwen_gdn_list_compat(namespace, mtp2_reference_fallback=True)
    metadata = torch.tensor([1], dtype=torch.int32)
    kwargs = {
        "output": "out",
        "x": "x",
        "weight": "weight",
        "conv_state": "state",
        "bias_opt": None,
        "query_start_loc_opt": torch.tensor([0, 1]),
        "cache_indices_opt": torch.tensor([0]),
        "initial_state_mode_opt": None,
        "num_accepted_tokens_opt": metadata,
        "activation_mode": 1,
        "pad_slot_id": -1,
        "run_mode": 1,
    }
    assert namespace.npu_causal_conv1d_custom(**kwargs) == "reference"
    kwargs["run_mode"] = 0
    assert namespace.npu_causal_conv1d_custom(**kwargs) == "native"
    assert [call[0] for call in calls] == ["reference", "native"]


def test_qwen_gdn_mtp2_prefill_uses_target_state_column(monkeypatch) -> None:
    seen = []

    class Operation:
        default = SimpleNamespace(
            _schema=(
                "_C_ascend::npu_causal_conv1d_custom(Tensor output, Tensor x, "
                "Tensor weight, Tensor conv_state, Tensor? bias_opt, "
                "int[] query_start_loc_opt, int[] cache_indices_opt, "
                "int[] initial_state_mode_opt, int[] num_accepted_tokens_opt, "
                "int activation_mode, int pad_slot_id, int run_mode) -> Tensor"
            )
        )

        def __call__(self, *args, **kwargs):
            seen.append(kwargs["cache_indices_opt"])
            return "native"

    monkeypatch.setenv(QWEN_GDN_LIST_COMPAT_ENV, "1")
    namespace = SimpleNamespace(npu_causal_conv1d_custom=Operation())
    assert install_qwen_gdn_list_compat(namespace, mtp2_reference_fallback=True)
    assert (
        namespace.npu_causal_conv1d_custom(
            output="out",
            x="x",
            weight="weight",
            conv_state="state",
            bias_opt=None,
            query_start_loc_opt=torch.tensor([0, 8192]),
            cache_indices_opt=torch.tensor([[6, 7, 8]]),
            initial_state_mode_opt=torch.tensor([False]),
            num_accepted_tokens_opt=None,
            activation_mode=1,
            pad_slot_id=-1,
            run_mode=0,
        )
        == "native"
    )
    assert seen == [[6]]


def test_qwen_gdn_list_compat_leaves_tensor_abi_untouched(monkeypatch) -> None:
    operation = SimpleNamespace(
        default=SimpleNamespace(
            _schema=(
                "op(Tensor? query_start_loc_opt, Tensor? cache_indices_opt) -> Tensor"
            )
        )
    )
    namespace = SimpleNamespace(npu_causal_conv1d_custom=operation)
    monkeypatch.setenv(QWEN_GDN_LIST_COMPAT_ENV, "true")

    assert not install_qwen_gdn_list_compat(namespace)
    assert namespace.npu_causal_conv1d_custom is operation


def test_removed_routed_expert_import_aliases_are_idempotent() -> None:
    namespace = SimpleNamespace()

    assert install_removed_routed_experts_types(namespace)
    values = namespace.RoutedExpertsLists("routes", "slots")
    assert values.routing_data == "routes"
    assert values.slot_mapping == "slots"
    assert not install_removed_routed_experts_types(namespace)

    partial = SimpleNamespace(RoutedExpertsLists=namespace.RoutedExpertsLists)
    with pytest.raises(RuntimeError, match="partial routed-experts"):
        install_removed_routed_experts_types(partial)


def test_removed_output_routing_field_accepts_only_disabled_value() -> None:
    class Output:
        def __init__(self, req_ids):
            self.req_ids = req_ids

    empty = Output([])
    namespace = SimpleNamespace(
        ModelRunnerOutput=Output, EMPTY_MODEL_RUNNER_OUTPUT=empty
    )

    assert install_removed_output_routing_field(namespace)
    result = Output(["req"], routed_experts=None)
    assert result.req_ids == ["req"]
    assert result.routed_experts is None
    assert empty.routed_experts is None
    assert not install_removed_output_routing_field(namespace)
    with pytest.raises(RuntimeError, match="routed-expert output"):
        Output(["req"], routed_experts=[1])

    class AlignedOutput:
        def __init__(self, routed_experts=None):
            self.routed_experts = routed_experts

    assert not install_removed_output_routing_field(
        SimpleNamespace(ModelRunnerOutput=AlignedOutput)
    )


def test_removed_async_output_routing_field_accepts_only_disabled_value() -> None:
    class AsyncOutput:
        def __init__(self, sampled_token_ids):
            self.sampled_token_ids = sampled_token_ids

    assert install_removed_async_output_routing_field(AsyncOutput)
    assert AsyncOutput([1], routed_experts=None).sampled_token_ids == [1]
    assert not install_removed_async_output_routing_field(AsyncOutput)
    with pytest.raises(RuntimeError, match="routed-expert output"):
        AsyncOutput([1], routed_experts=[1])

    class AlignedAsyncOutput:
        def __init__(self, routed_experts=None):
            self.routed_experts = routed_experts

    assert not install_removed_async_output_routing_field(AlignedAsyncOutput)


def test_qwen_mtp2_cache_binding_bridge_is_exact_and_restored(monkeypatch) -> None:
    from vllm.v1.worker import utils as worker_utils

    original = worker_utils.bind_kv_cache
    bound = []
    monkeypatch.setattr(
        worker_utils,
        "bind_kv_cache_to_layers",
        lambda *args: bound.append(args),
        raising=False,
    )
    caches = {
        "language_model.model.layers.0.linear_attn": object(),
        "mtp.layers.0.self_attn.attn": object(),
        "language_model.model.layers.1.linear_attn": object(),
    }
    runner = []
    with qwen_mtp2_cache_binding_bridge():
        assert worker_utils.bind_kv_cache is not original
        worker_utils.bind_kv_cache(caches, {}, runner)
        with pytest.raises(RuntimeError, match="unsupported duplicate KV"):
            worker_utils.bind_kv_cache(
                {**caches, "other.layers.1.self_attn": object()}, {}, []
            )
    assert worker_utils.bind_kv_cache is original
    assert runner == list(caches.values())
    assert len(bound) == 1


def test_qwen_mtp2_cache_binding_bridge_supports_legacy_inline_binding(
    monkeypatch,
) -> None:
    from vllm.v1.worker import utils as worker_utils

    monkeypatch.delattr(worker_utils, "bind_kv_cache_to_layers", raising=False)
    caches = {
        "language_model.model.layers.0.linear_attn": object(),
        "mtp.layers.0.self_attn.attn": object(),
        "language_model.model.layers.1.linear_attn": object(),
    }
    contexts = {
        name: SimpleNamespace(kv_cache=None) for name in caches
    }
    runner = []

    with qwen_mtp2_cache_binding_bridge():
        worker_utils.bind_kv_cache(caches, contexts, runner)
        with pytest.raises(RuntimeError, match="explicit cache groups"):
            worker_utils.bind_kv_cache(caches, contexts, [], kv_cache_groups=[0])

    assert runner == list(caches.values())
    assert all(contexts[name].kv_cache is cache for name, cache in caches.items())


def test_step3p5_import_alias_is_idempotent() -> None:
    step3p5 = SimpleNamespace()
    class_sentinel = object()

    assert install_step3p5_fp32_linear_alias(step3p5, class_sentinel)
    assert step3p5.FP32ReplicatedLinear is class_sentinel
    assert not install_step3p5_fp32_linear_alias(step3p5, object())


def test_dcp_length_alias_is_optional_and_idempotent() -> None:
    calls = []
    cp_utils = SimpleNamespace(
        prepare_dcp_local_seq_lens=lambda *args, **kwargs: (
            calls.append((args, kwargs)) or "prepared"
        )
    )

    assert install_dcp_length_alias(cp_utils)
    alias = cp_utils.maybe_prepare_dcp_local_seq_lens
    assert alias("buffer", "lengths", 2, 1, 0, 1) is None
    assert not calls
    assert alias("buffer", "lengths", 2, 2, 0, 1, num_reqs_padded=4) == "prepared"
    assert calls == [(("buffer", "lengths", 2, 2, 0, 1), {"num_reqs_padded": 4})]
    assert not install_dcp_length_alias(cp_utils)


def test_removed_model_routing_flag_uses_disabled_aux_output() -> None:
    model_config = SimpleNamespace()
    config = SimpleNamespace(
        model_config=model_config,
        aux_output_config=SimpleNamespace(enable_return_routed_experts=False),
    )

    assert install_removed_model_routing_flag(config)
    assert model_config.enable_return_routed_experts is False
    assert not install_removed_model_routing_flag(config)

    enabled = SimpleNamespace(
        model_config=SimpleNamespace(),
        aux_output_config=SimpleNamespace(enable_return_routed_experts=True),
    )
    with pytest.raises(RuntimeError, match="routed-expert output"):
        install_removed_model_routing_flag(enabled)


def test_slot_mapping_hook_is_idempotent() -> None:
    _install_slot_mapping_hook(_BlockTable)
    first = _BlockTable.compute_slot_mapping
    _install_slot_mapping_hook(_BlockTable)
    assert _BlockTable.compute_slot_mapping is first


def test_runtime_slot_mapping_hook_uses_concrete_ascend_table() -> None:
    class ConcreteBlockTable:
        def compute_slot_mapping(self, num_reqs, query_start_loc, positions):
            self.arguments = num_reqs, query_start_loc, positions

    concrete = ConcreteBlockTable()
    runner = type(
        "Runner",
        (),
        {
            "input_batch": type(
                "InputBatch",
                (),
                {"block_table": type("MultiGroup", (), {"block_tables": [concrete]})()},
            )()
        },
    )()

    _install_runtime_slot_mapping_hooks(runner)

    assert ConcreteBlockTable.__dict__["_ascend_kvcompress_patch_v3_slot_mapping"]


def test_runtime_slot_mapping_hook_rejects_changed_group_layout() -> None:
    runner = type(
        "Runner",
        (),
        {
            "input_batch": type(
                "InputBatch",
                (),
                {"block_table": type("MultiGroup", (), {"block_tables": []})()},
            )()
        },
    )()

    with pytest.raises(RuntimeError, match="full-attention block table is unavailable"):
        _install_runtime_slot_mapping_hooks(runner)


def test_lazy_loader_patches_runner_after_module_execution(monkeypatch) -> None:
    calls = []
    selection = object()
    monkeypatch.setattr(
        plugin,
        "_install_runner_hooks",
        lambda runner, config: calls.append((runner, config)),
    )
    loader = _RunnerPatchLoader(_WrappedLoader(), selection)
    module = ModuleType("fake_runner")
    spec = ModuleSpec(module.__name__, loader)

    assert loader.create_module(spec) is None
    loader.exec_module(module)

    assert calls == [(_Runner, selection)]


def test_runner_binds_the_normalized_ascend_cache_plan(monkeypatch) -> None:
    events = []

    class Provider:
        def __init__(self, vllm_config, selection):
            events.append(("create", vllm_config, selection))

        def validate_host(self, runner):
            events.append(("validate", runner))

        def bind_model_runner(self, runner, cache_config):
            events.append(("bind", runner, cache_config))

    class Runner:
        vllm_config = SimpleNamespace(model_config=SimpleNamespace())

        def load_model(self):
            return "loaded"

        def initialize_kv_cache(self, incoming, *, kv_cache_allocation_context=None):
            events.append(("allocation_context", kv_cache_allocation_context))
            self.kv_cache_config = "normalized-cache-plan"
            concrete = _BlockTable()
            self.input_batch = type(
                "InputBatch",
                (),
                {"block_table": type("MultiGroup", (), {"block_tables": [concrete]})()},
            )()
            return incoming

        def _update_states(self, output):
            return output

        def _build_attention_metadata(self, *args, **kwargs):
            return args, kwargs

        def sample_tokens(self, grammar):
            return grammar

    monkeypatch.setattr(plugin, "AscendKVCompressionProvider", Provider)
    selection = object()
    _install_runner_hooks(Runner, selection)

    assert (
        Runner().initialize_kv_cache(
            "incoming-plan", kv_cache_allocation_context="pool"
        )
        == "incoming-plan"
    )
    assert ("allocation_context", "pool") in events
    assert events[-1][0] == "bind"
    assert events[-1][2] == "normalized-cache-plan"


def test_runner_generates_calibration_before_loading_model(monkeypatch) -> None:
    events = []

    class Runner:
        def load_model(self):
            events.append("load")

        def initialize_kv_cache(self, config):
            return config

        def _update_states(self, output):
            return output

        def _build_attention_metadata(self, *args, **kwargs):
            return args, kwargs

        def sample_tokens(self, grammar):
            return grammar

    import vllm_ascend_kvcompress.calibration as calibration

    monkeypatch.setattr(
        calibration,
        "ensure_calibration_for_runner",
        lambda runner, selection: events.append("calibrate") or True,
    )
    _install_runner_hooks(Runner, object())

    Runner().load_model()

    assert events == ["calibrate", "load"]
    assert Runner.routed_experts_initialized is False
