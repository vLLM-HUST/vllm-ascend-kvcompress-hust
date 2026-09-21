# SPDX-License-Identifier: Apache-2.0

from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace

import pytest
import torch

import vllm_ascend_kvcompress.plugin as plugin
from vllm_ascend_kvcompress.host_compat import (
    QWEN_GDN_LIST_COMPAT_ENV,
    install_qwen_gdn_list_compat,
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
    monkeypatch.setattr(plugin, "import_module", imported.append)

    _prepare_current_triton_runtime()

    assert imported == list(modules)


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
        vllm_config = object()

        def load_model(self):
            return "loaded"

        def initialize_kv_cache(self, incoming):
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

    assert Runner().initialize_kv_cache("incoming-plan") == "incoming-plan"
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
