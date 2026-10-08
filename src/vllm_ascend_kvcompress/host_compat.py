# SPDX-License-Identifier: Apache-2.0
"""Explicit compatibility shims for validated host/runtime ABI mismatches."""

from __future__ import annotations

import inspect
import os
import weakref
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, NamedTuple

from vllm.logger import logger

QWEN_GDN_LIST_COMPAT_ENV = "VLLM_ASCEND_KVCOMPRESS_QWEN_GDN_LIST_COMPAT"
_GDN_OP_NAME = "npu_causal_conv1d_custom"
_GDN_PATCH_MARKER = "_ascend_kvcompress_qwen_gdn_list_compat_v1"
_GDN_METADATA_ARGUMENTS = (
    (5, "query_start_loc_opt"),
    (6, "cache_indices_opt"),
    (7, "initial_state_mode_opt"),
    (8, "num_accepted_tokens_opt"),
)
_MAX_METADATA_CACHE_ENTRIES = 64
_gdn_metadata_cache: OrderedDict[
    int, tuple[weakref.ReferenceType[Any], int | None, list[int]]
] = OrderedDict()


def install_grouped_topk_router_compat() -> bool:
    """Preserve the Ascend router's scaling check on newer vLLM hosts.

    The Ascend router still passes ``routed_scaling_factor`` to vLLM's
    classifier. Newer vLLM removed that argument, but the old classifier
    returned ``Unspecified`` for a biased, ungrouped sigmoid router with a
    non-unit scale. Keep that behavior without changing the host itself.
    """
    from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
    from vllm_ascend.ops.fused_moe.router import grouped_topk_router

    classify = grouped_topk_router.get_routing_method_type
    if getattr(classify, "_ascend_kvcompress_scaling_compat", False):
        return False
    if "routed_scaling_factor" in inspect.signature(classify).parameters:
        return False

    def classify_with_scaling(
        *,
        scoring_func: str,
        top_k: int,
        renormalize: bool,
        num_expert_group: int | None,
        has_e_score_bias: bool,
        routed_scaling_factor: float | None = 1.0,
    ) -> RoutingMethodType:
        if (
            has_e_score_bias
            and scoring_func == "sigmoid"
            and not num_expert_group
            and routed_scaling_factor not in (None, 1.0)
        ):
            return RoutingMethodType.Unspecified
        return classify(
            scoring_func=scoring_func,
            top_k=top_k,
            renormalize=renormalize,
            num_expert_group=num_expert_group,
            has_e_score_bias=has_e_score_bias,
        )

    classify_with_scaling._ascend_kvcompress_scaling_compat = True  # type: ignore[attr-defined]
    grouped_topk_router.get_routing_method_type = classify_with_scaling
    logger.warning("Installed Ascend grouped TopK router scaling compatibility")
    return True


def install_layer_aware_fia_graph_replay(vllm_config: Any) -> bool:
    """Enable the host's existing layer-keyed FULL graph update path.

    The validated Ascend host already records an attention layer name in each
    captured FIA task and can rebind that task from the current per-layer
    metadata.  It enables the path only for built-in mixed-attention model
    families.  External compression methods with unequal physical lengths
    need the same method-neutral mechanism, so opt into it only after checking
    that every expected host seam is present.
    """
    mode = getattr(vllm_config.compilation_config, "cudagraph_mode", None)
    mode_name = getattr(mode, "name", str(mode))
    if mode_name == "NONE":
        return False
    if mode_name != "FULL_AND_PIECEWISE":
        raise RuntimeError(
            "per-layer KV compression requires FULL_AND_PIECEWISE graph mode"
        )

    from vllm_ascend.attention import attention_v1
    from vllm_ascend.attention import utils as attention_utils

    impl = attention_v1.AscendAttentionBackendImpl
    marker = "_ascend_kvcompress_layer_aware_graph_v1"
    if getattr(impl, marker, False):
        return False
    required_impl_seams = (
        "_graph_metadata_layer_name",
        "update_graph_params",
        "full_graph_fia",
    )
    if any(not callable(getattr(impl, name, None)) for name in required_impl_seams):
        raise RuntimeError(
            "Ascend attention backend lacks layer-aware FIA graph replay seams"
        )
    original = getattr(attention_utils, "needs_layer_aware_fia_graph_replay", None)
    if not callable(original):
        raise RuntimeError("Ascend layer-aware graph capability probe is unavailable")
    if attention_v1.needs_layer_aware_fia_graph_replay is not original:
        raise RuntimeError("Ascend layer-aware graph capability probe changed")

    # Validate semantics rather than enabling a similarly named older hook.
    update_source = inspect.getsource(impl.update_graph_params)
    if not all(
        token in update_source
        for token in ("layer_name", "metadata_key", "attn_metadata")
    ):
        raise RuntimeError(
            "Ascend FIA graph updater cannot consume layer-distinct metadata"
        )

    def layer_aware_replay_enabled() -> bool:
        return True

    attention_utils.needs_layer_aware_fia_graph_replay = layer_aware_replay_enabled
    attention_v1.needs_layer_aware_fia_graph_replay = layer_aware_replay_enabled
    setattr(impl, marker, True)
    logger.warning(
        "Enabled layer-aware FIA graph replay for external per-layer KV compression"
    )
    return True


@contextmanager
def qwen_mtp2_cache_binding_bridge():
    """Bind Qwen3.5's MTP layer alongside target layer zero, only at startup.

    The current generic binder rejects duplicate numeric layer indices before
    binding, although the runner's KV list is iterated as a collection rather
    than indexed by that number. Retain the host's actual per-layer binding and
    ordering. Fail closed for every collision except the exact Qwen3.5 pair.
    """
    from vllm.model_executor.models.utils import extract_layer_index
    from vllm.v1.worker import utils as worker_utils

    original = worker_utils.bind_kv_cache

    def bind_kv_cache(
        kv_caches: Any,
        forward_context: Any,
        runner_kv_caches: Any,
        num_attn_module: int = 1,
        kv_cache_groups: Any = None,
    ) -> None:
        from collections import defaultdict

        index2names: dict[int, list[str]] = defaultdict(list)
        for name in kv_caches:
            index2names[extract_layer_index(name, num_attn_module)].append(name)
        collisions = {
            index: names for index, names in index2names.items() if len(names) > 1
        }
        if not collisions:
            return original(
                kv_caches,
                forward_context,
                runner_kv_caches,
                num_attn_module,
                kv_cache_groups,
            )
        expected = {
            "language_model.model.layers.0.linear_attn",
            "mtp.layers.0.self_attn.attn",
        }
        if (
            num_attn_module != 1
            or set(collisions) != {0}
            or set(collisions[0]) != expected
            or runner_kv_caches
        ):
            raise RuntimeError(
                "unsupported duplicate KV layer indices for Qwen3.5 MTP2: "
                f"{collisions!r}"
            )
        for index in sorted(index2names):
            runner_kv_caches.extend(kv_caches[name] for name in index2names[index])
        bind_to_layers = getattr(worker_utils, "bind_kv_cache_to_layers", None)
        if bind_to_layers is None:
            # vLLM 0.23 keeps the layer-binding loop inside ``bind_kv_cache``;
            # later hosts extracted it as a helper.  The collision bridge has
            # already performed only the runner-list half, so reproduce the
            # unchanged layer assignment on the older ABI.
            if kv_cache_groups is not None:
                raise RuntimeError(
                    "the legacy KV binder cannot accept explicit cache groups"
                )
            for layer_name, kv_cache in kv_caches.items():
                forward_context[layer_name].kv_cache = kv_cache
        else:
            bind_to_layers(kv_caches, forward_context, num_attn_module, kv_cache_groups)
        logger.warning("Bound exact Qwen3.5 target/MTP layer-zero KV pair")

    worker_utils.bind_kv_cache = bind_kv_cache
    try:
        yield
    finally:
        worker_utils.bind_kv_cache = original


class _RoutedExpertsLists(NamedTuple):
    routing_data: Any
    slot_mapping: Any


class _RoutedExpertsTensors(NamedTuple):
    routing_data: Any
    slot_mapping: Any


def install_removed_routed_experts_types(outputs_module: Any | None = None) -> bool:
    """Restore two import-only aliases removed by the newer vLLM host.

    Ascend imports these names even when routed-expert output is disabled.
    The plugin does not enable that optional feature; this only lets the
    worker module load without modifying either upstream repository.
    """
    if outputs_module is None:
        from vllm.v1 import outputs as outputs_module

    names = ("RoutedExpertsLists", "RoutedExpertsTensors")
    present = tuple(hasattr(outputs_module, name) for name in names)
    if all(present):
        return False
    if any(present):
        raise RuntimeError("partial routed-experts output ABI is unsupported")
    outputs_module.RoutedExpertsLists = _RoutedExpertsLists
    outputs_module.RoutedExpertsTensors = _RoutedExpertsTensors
    logger.warning("Installed import-only routed-experts aliases for Ascend worker")
    return True


def install_removed_output_routing_field(outputs_module: Any | None = None) -> bool:
    """Accept Ascend's disabled routed-expert field on a newer vLLM output.

    Ascend passes ``routed_experts=None`` unconditionally. The current vLLM
    output removed that constructor argument. Reject non-None values rather
    than silently discarding an enabled auxiliary output.
    """
    if outputs_module is None:
        from vllm.v1 import outputs as outputs_module

    output_cls = outputs_module.ModelRunnerOutput
    marker = "_ascend_kvcompress_routing_field_compat_v1"
    if getattr(output_cls, marker, False):
        return False
    if "routed_experts" in inspect.signature(output_cls).parameters:
        return False

    original_init = output_cls.__init__

    def init(self: Any, *args: Any, routed_experts: Any = None, **kwargs: Any) -> None:
        if routed_experts is not None:
            raise RuntimeError("routed-expert output requires an aligned vLLM host")
        original_init(self, *args, **kwargs)
        self.routed_experts = None

    output_cls.__init__ = init
    setattr(output_cls, marker, True)
    empty = getattr(outputs_module, "EMPTY_MODEL_RUNNER_OUTPUT", None)
    if empty is not None:
        empty.routed_experts = None
    logger.warning("Installed disabled routed-experts output compatibility")
    return True


def install_removed_async_output_routing_field(output_cls: Any | None = None) -> bool:
    """Accept Ascend's disabled routing field on the async output wrapper.

    The aligned vLLM async class no longer accepts ``routed_experts``. Keep
    non-None values fail-closed because this plugin cannot transport that
    auxiliary data through the current host's async output path.
    """
    if output_cls is None:
        from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput

        output_cls = AsyncGPUModelRunnerOutput

    marker = "_ascend_kvcompress_async_routing_compat_v1"
    if getattr(output_cls, marker, False):
        return False
    if "routed_experts" in inspect.signature(output_cls).parameters:
        return False

    original_init = output_cls.__init__

    def init(self: Any, *args: Any, routed_experts: Any = None, **kwargs: Any) -> None:
        if routed_experts is not None:
            raise RuntimeError("routed-expert output requires an aligned vLLM host")
        original_init(self, *args, **kwargs)

    output_cls.__init__ = init
    setattr(output_cls, marker, True)
    logger.warning("Installed disabled async routed-experts output compatibility")
    return True


def install_step3p5_fp32_linear_alias(
    step3p5_module: Any | None = None, linear_class: Any | None = None
) -> bool:
    """Supply a removed import used by Ascend's unrelated Step3p5 patch.

    The old class was a replicated linear initialized with FP32 weights. The
    current generic class accepts the same explicit ``params_dtype`` used at
    that call site. This shim does not enable Step3p5 for KV compression.
    """
    if step3p5_module is None:
        from vllm.model_executor.models import step3p5 as step3p5_module
    if hasattr(step3p5_module, "FP32ReplicatedLinear"):
        return False
    if linear_class is None:
        from vllm.model_executor.layers.linear import ReplicatedLinear

        linear_class = ReplicatedLinear
    step3p5_module.FP32ReplicatedLinear = linear_class
    logger.warning("Installed Step3p5 FP32 linear import alias for Ascend patch")
    return True


def install_dcp_length_alias(cp_utils_module: Any | None = None) -> bool:
    """Restore the optional DCP helper name expected by Ascend's v2 imports."""
    if cp_utils_module is None:
        from vllm.v1.worker.gpu import cp_utils as cp_utils_module
    if hasattr(cp_utils_module, "maybe_prepare_dcp_local_seq_lens"):
        return False
    prepare = getattr(cp_utils_module, "prepare_dcp_local_seq_lens", None)
    if prepare is None:
        raise RuntimeError("current vLLM DCP length helper is unavailable")

    def maybe_prepare_dcp_local_seq_lens(
        dcp_local_seq_lens: Any,
        seq_lens: Any,
        num_reqs: int,
        dcp_size: int,
        dcp_rank: int,
        cp_interleave: int,
        *,
        num_reqs_padded: int | None = None,
    ) -> Any:
        if dcp_size == 1:
            return None
        return prepare(
            dcp_local_seq_lens,
            seq_lens,
            num_reqs,
            dcp_size,
            dcp_rank,
            cp_interleave,
            num_reqs_padded=num_reqs_padded,
        )

    cp_utils_module.maybe_prepare_dcp_local_seq_lens = maybe_prepare_dcp_local_seq_lens
    logger.warning("Installed optional DCP length import alias for Ascend worker")
    return True


def install_removed_model_routing_flag(vllm_config: Any) -> bool:
    """Supply Ascend's old model-level flag only when aux output is disabled."""
    model_config = vllm_config.model_config
    if hasattr(model_config, "enable_return_routed_experts"):
        return False
    aux = getattr(vllm_config, "aux_output_config", None)
    if bool(getattr(aux, "enable_return_routed_experts", False)):
        raise RuntimeError("routed-expert output requires an aligned Ascend host")
    object.__setattr__(model_config, "enable_return_routed_experts", False)
    logger.warning("Installed removed model routing flag for Ascend worker")
    return True


def qwen_gdn_list_compat_enabled() -> bool:
    """Return whether the operator ABI workaround was explicitly requested."""
    return os.getenv(QWEN_GDN_LIST_COMPAT_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def clear_qwen_gdn_list_compat_cache() -> None:
    """Start a new model-execution step with fresh device metadata."""
    _gdn_metadata_cache.clear()


def install_qwen_gdn_list_compat(
    op_namespace: Any | None = None, *, mtp2_reference_fallback: bool = False
) -> bool:
    """Bridge a legacy ``int[]`` GDN operator ABI without editing the host.

    The current Python host passes device tensors for the four metadata inputs,
    while an older installed ``vllm_ascend_C`` binary declares those inputs as
    integer lists.  This opt-in wrapper is installed only for that exact schema.
    Aligned binaries exposing ``Tensor?`` inputs are left untouched.
    """
    if not qwen_gdn_list_compat_enabled():
        return False

    import torch

    if op_namespace is None:
        from vllm_ascend.utils import enable_custom_op

        if not enable_custom_op():
            raise RuntimeError("Ascend custom operators are unavailable")
        op_namespace = torch.ops._C_ascend

    operation = getattr(op_namespace, _GDN_OP_NAME, None)
    if operation is None:
        raise RuntimeError(f"Ascend custom operator {_GDN_OP_NAME!r} is unavailable")
    if getattr(operation, _GDN_PATCH_MARKER, False):
        return True

    overload = getattr(operation, "default", None)
    schema = str(getattr(overload, "_schema", ""))
    if "Tensor? query_start_loc_opt" in schema:
        logger.info("Qwen GDN custom operator already exposes the Tensor metadata ABI")
        return False
    if "int[] query_start_loc_opt" not in schema:
        raise RuntimeError(f"unsupported Qwen GDN custom operator schema: {schema!r}")

    def as_int_list(value: Any, name: str) -> Any:
        if value is None:
            # The legacy schema is non-optional ``int[]``.  Its C++ caller
            # represents absent optional metadata as an empty ArrayRef, so use
            # an empty Python list rather than forwarding ``None``.
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return [int(item) for item in value]
        if not torch.is_tensor(value):
            raise RuntimeError(f"{name} must be a tensor or integer list")
        # Current hybrid-cache metadata carries the complete per-request
        # Mamba block table as a 2-D cache_indices tensor.  The legacy C++ ABI
        # receives the same storage as a flat ArrayRef<int64_t>; preserve that
        # row-major layout when bridging it to Python's int[].  The remaining
        # metadata fields are semantically vectors and must stay one-dimensional.
        if name == "cache_indices_opt":
            value = value.reshape(-1)
        elif value.ndim != 1:
            raise RuntimeError(f"{name} must be one-dimensional")

        identity = id(value)
        try:
            version: int | None = int(value._version)
        except RuntimeError as error:
            if "do not track version counter" not in str(error):
                raise
            version = None
        cached = _gdn_metadata_cache.get(identity)
        if cached is not None and cached[0]() is value and cached[1] == version:
            _gdn_metadata_cache.move_to_end(identity)
            return cached[2]

        converted = [int(item) for item in value.detach().cpu().tolist()]
        _gdn_metadata_cache[identity] = (weakref.ref(value), version, converted)
        _gdn_metadata_cache.move_to_end(identity)
        while len(_gdn_metadata_cache) > _MAX_METADATA_CACHE_ENTRIES:
            _gdn_metadata_cache.popitem(last=False)
        return converted

    diagnostic_counts = {"reference": 0, "native": 0}

    def operation_argument(
        args: tuple[Any, ...], kwargs: dict[str, Any], index: int, name: str
    ) -> Any:
        return args[index] if len(args) > index else kwargs.get(name)

    def compatible_operation(*args: Any, **kwargs: Any) -> Any:
        run_mode = kwargs.get("run_mode", args[11] if len(args) > 11 else None)
        accepted = kwargs.get(
            "num_accepted_tokens_opt", args[8] if len(args) > 8 else None
        )
        if mtp2_reference_fallback and run_mode == 1 and accepted is not None:
            from .qwen_gdn_ops import causal_conv1d_spec_reference

            diagnostic_counts["reference"] += 1
            if diagnostic_counts["reference"] == 1:
                logger.warning("Using plugin-owned Qwen GDN MTP2 reference convolution")

            def argument(index: int, name: str) -> Any:
                return args[index] if len(args) > index else kwargs[name]

            return causal_conv1d_spec_reference(
                argument(0, "output"),
                argument(1, "x"),
                argument(2, "weight"),
                argument(3, "conv_state"),
                kwargs.get("bias_opt", args[4] if len(args) > 4 else None),
                argument(5, "query_start_loc_opt"),
                argument(6, "cache_indices_opt"),
                accepted,
                argument(9, "activation_mode"),
                argument(10, "pad_slot_id"),
            )
        positional = list(args)
        if mtp2_reference_fallback and run_mode == 0:
            cache_indices = operation_argument(args, kwargs, 6, "cache_indices_opt")
            if torch.is_tensor(cache_indices) and cache_indices.ndim == 2:
                if cache_indices.shape[1] < 1:
                    raise RuntimeError("Qwen GDN MTP2 cache block table is empty")
                target_indices = cache_indices[:, 0]
                if len(positional) > 6:
                    positional[6] = target_indices
                else:
                    kwargs["cache_indices_opt"] = target_indices
        for index, name in _GDN_METADATA_ARGUMENTS:
            if index < len(positional):
                positional[index] = as_int_list(positional[index], name)
            elif name in kwargs:
                kwargs[name] = as_int_list(kwargs[name], name)
        if not mtp2_reference_fallback:
            return operation(*positional, **kwargs)
        diagnostic_counts["native"] += 1
        if diagnostic_counts["native"] == 1:
            bounds = operation_argument(
                tuple(positional), kwargs, 5, "query_start_loc_opt"
            )
            indices = operation_argument(
                tuple(positional), kwargs, 6, "cache_indices_opt"
            )
            initial = operation_argument(
                tuple(positional), kwargs, 7, "initial_state_mode_opt"
            )
            input_tensor = operation_argument(tuple(positional), kwargs, 1, "x")
            state_tensor = operation_argument(
                tuple(positional), kwargs, 3, "conv_state"
            )
            logger.warning(
                "Qwen GDN MTP2 first native convolution: mode=%s input=%s "
                "state=%s bounds_len=%s bounds_end=%s indices_len=%s "
                "indices_first=%s initial_len=%s initial_first=%s",
                run_mode,
                tuple(getattr(input_tensor, "shape", ())),
                tuple(getattr(state_tensor, "shape", ())),
                len(bounds) if bounds is not None else None,
                bounds[-1] if bounds else None,
                len(indices) if indices is not None else None,
                indices[:4] if indices else None,
                len(initial) if initial is not None else None,
                initial[:4] if initial else None,
            )
        try:
            return operation(*positional, **kwargs)
        except RuntimeError:
            logger.exception(
                "Qwen GDN legacy operator failed in experimental MTP2: "
                "run_mode=%s accepted_present=%s input_shape=%s state_shape=%s "
                "native_call=%d reference_calls=%d",
                run_mode,
                accepted is not None,
                tuple(
                    getattr(args[1] if len(args) > 1 else kwargs.get("x"), "shape", ())
                ),
                tuple(
                    getattr(
                        args[3] if len(args) > 3 else kwargs.get("conv_state"),
                        "shape",
                        (),
                    )
                ),
                diagnostic_counts["native"],
                diagnostic_counts["reference"],
            )
            raise

    setattr(compatible_operation, _GDN_PATCH_MARKER, True)
    setattr(compatible_operation, f"{_GDN_PATCH_MARKER}_original", operation)
    setattr(op_namespace, _GDN_OP_NAME, compatible_operation)
    logger.warning(
        "Enabled opt-in Qwen GDN list-ABI compatibility for %s; eager execution "
        "is required",
        schema,
    )
    return True
