# SPDX-License-Identifier: Apache-2.0
"""针对已确认的宿主与运行时 ABI 差异提供显式兼容层。"""

from __future__ import annotations

import inspect
import os
import weakref
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, NamedTuple

from vllm.logger import logger

# 本文件处理宿主版本/算子 ABI，不计算 TriAttention 分数或选择 token。
# 大部分安装器按符号/签名检测后幂等安装；GDN list bridge 另需显式环境开关。
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
# 仅缓存旧 int[] ABI 的 tensor→Python list 转换；与 KV 稀疏索引无关。
# 键按对象身份，值同时检查弱引用及 version；execute_model 每步清空，防止跨步陈旧。
# 最大 64 项限制引用数量，list 化仍可能触发设备同步。
_gdn_metadata_cache: OrderedDict[
    int, tuple[weakref.ReferenceType[Any], int | None, list[int]]
] = OrderedDict()


def install_grouped_topk_router_compat() -> bool:
    """在新版 vLLM 宿主上保留 Ascend 路由器的缩放检查。

    Ascend 路由器仍向 vLLM 分类器传入 routed_scaling_factor，
    但新版 vLLM 已删除该参数。旧分类器对带偏置、未分组、缩放非 1
    的 sigmoid 路由返回 Unspecified。这里保留该行为，不修改宿主源码。
    """
    from vllm.model_executor.layers.fused_moe.config import (
        RoutingMethodType,
        get_routing_method_type,
    )

    # 受支持的旧版核心已处理缩放，配套 Ascend 版本尚无 router 包。
    # 因此先检查核心 ABI，再导入仅新版提供的模块。
    # 若核心确实需要此兼容层，
    # 则导入失败仍须向上传播。
    # 先看核心分类函数是否已支持旧参数；支持时无需加载较新的 Ascend router 包。
    # 需要桥接时只补参数语义，不重新实现 MoE 路由选择。
    if "routed_scaling_factor" in inspect.signature(get_routing_method_type).parameters:
        return False

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


@contextmanager
def qwen_mtp2_cache_binding_bridge():
    """仅在启动时，将 Qwen3.5 的 MTP 层与目标模型第零层一并绑定。

    当前通用绑定器在绑定前拒绝重复的数字层索引，但 runner 的 KV 列表
    实际按集合遍历，并不按该数字索引访问。保留宿主原有的逐层绑定
    与顺序，除 Qwen3.5 这组确切的索引冲突外，其余冲突均报错。
    """
    from vllm.model_executor.models.utils import extract_layer_index
    from vllm.v1.worker import utils as worker_utils

    # 这个桥只临时替换启动时 binder；contextmanager 的 finally 恢复原函数。
    # 持久类补丁和临时函数桥具有不同作用域，排查模型加载问题时应分别看待。
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
        # 正常层号布局直接委托原 binder；只有精确的目标层零/MTP 层零冲突才走桥。
        # 保留宿主对层对象的真实绑定，修正的是 runner 列表构造阶段的重复层号限制。
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
            # vLLM 0.23 将逐层绑定循环放在 bind_kv_cache 内部，
            # 后续宿主才将其提取为辅助函数。索引冲突兼容层
            # 已完成 runner 列表的绑定部分，
            # 因此在旧 ABI 上补做原有的逐层赋值。
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
    """恢复新版 vLLM 宿主删除的两个仅用于导入的类型别名。

    即使关闭路由专家输出，Ascend 也会导入这些名称。插件不启用该功能，
    这里只保证 worker 模块能够加载，无需修改两个上游仓库。
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
    """让新版 vLLM 输出类型接受 Ascend 已关闭的路由专家字段。

    Ascend 总会传入 routed_experts=None，而当前 vLLM 输出构造器
    已删除此参数。非 None 值仍须报错，避免静默丢弃已启用的辅助输出。
    """
    if outputs_module is None:
        from vllm.v1 import outputs as outputs_module

    # 部分宿主输出类已移除 routed_experts 参数，但 Ascend 仍传入 None。
    # 兼容层只接受关闭状态；不能静默吞掉真正启用的辅助路由输出。
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
    """让异步输出包装器接受 Ascend 已关闭的路由字段。

    配套 vLLM 的异步类不再接受 routed_experts。由于插件无法通过当前
    宿主异步输出路径传输该辅助数据，因此非 None 值仍须报错。
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
    """补齐 Ascend 的 Step3p5 补丁使用、但宿主已删除的导入名称。

    旧类是使用 FP32 权重初始化的复制线性层。当前通用类接受调用点
    显式传入的同一个 params_dtype 参数。该兼容层不为 KV 压缩启用
    Step3p5 模型支持。
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
    """恢复 Ascend v2 导入路径期望的可选 DCP 辅助函数名。"""
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
    """仅在辅助输出关闭时补齐 Ascend 旧版模型级标志。"""
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
    """返回是否显式启用了算子 ABI 兼容选项。"""
    return os.getenv(QWEN_GDN_LIST_COMPAT_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def clear_qwen_gdn_list_compat_cache() -> None:
    """清除上一步设备元数据，开始新的模型执行步骤。"""
    _gdn_metadata_cache.clear()


def install_qwen_gdn_list_compat(
    op_namespace: Any | None = None, *, mtp2_reference_fallback: bool = False
) -> bool:
    """在不修改宿主的前提下桥接旧版 int[] GDN 算子 ABI。

    当前 Python 宿主为四项元数据传入设备张量，但部分已安装的旧版
    vllm_ascend_C 二进制将其声明为整数列表。此可选包装仅对该确切
    schema 安装，已对齐并使用 Tensor? 输入的二进制保持原有调用路径。
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
    # 同名算子可能有不同 C++ schema，只有旧 int[] 版本需要桥接。
    # 已经接收 Tensor 元数据的版本保持原路径，不额外做设备到 CPU 的转换。
    schema = str(getattr(overload, "_schema", ""))
    if "Tensor? query_start_loc_opt" in schema:
        logger.info("Qwen GDN custom operator already exposes the Tensor metadata ABI")
        return False
    if "int[] query_start_loc_opt" not in schema:
        raise RuntimeError(f"unsupported Qwen GDN custom operator schema: {schema!r}")

    # 旧 C++ schema 明确要求 int[]，因此这里只桥接已识别的旧版本。
    # None 用空数组表达；二维 cache_indices 按行展平，其余字段必须是一维。
    # 若张量来自 inference mode 且无 version counter，仍依靠步级清理防止跨步复用。
    def as_int_list(value: Any, name: str) -> Any:
        if value is None:
            # 旧版 schema 使用非可选的 int[]。C++ 调用方
            # 以空 ArrayRef 表示缺失的可选元数据，
            # 因此这里使用空 Python 列表，而非传入 None。
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return [int(item) for item in value]
        if not torch.is_tensor(value):
            raise RuntimeError(f"{name} must be a tensor or integer list")
        # 当前混合缓存元数据以二维 cache_indices 张量保存
        # 每个请求完整的 Mamba 块表。旧版 C++ ABI
        # 将同一存储视为展平的 ArrayRef<int64_t>；
        # 转成 Python int[] 时须保留行优先布局。
        # 其余元数据字段在语义上是向量，必须保持一维。
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
        # 对象地址可能复用，张量也可能原地变化；因此同时核对弱引用和版本计数。
        # 对没有版本计数的 inference tensor，还依赖 execute_model 开头的逐步清空。
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
        # 仅实验 MTP2 的 speculative 卷积进入插件参考实现，
        # 以处理 accepted-token 滚动状态。
        # 普通 prefill/未启用回退仍调用原生算子；此分支是模型 ABI 适配，
        # 不是 KV 选取算法。
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


# 这部分属于宿主/模型 ABI 准备，不依赖选中的稀疏化算法。
# Qwen3.5 先加载原生算子注册，再只填补缺失项，最后准备通用路由兼容。
def prepare_model_load(vllm_config: Any) -> None:
    """安装宿主与模型 ABI 兼容支持，与算法准备流程分离。"""
    from importlib import import_module

    from .contracts import supports_mtp2

    install_qwen_gdn_list_compat(mtp2_reference_fallback=supports_mtp2(vllm_config))
    model_type = str(getattr(vllm_config.model_config.hf_config, "model_type", ""))
    if model_type.startswith("qwen3_5"):
        # 优先使用原生注册；JIT 算子仅补齐缺失的符号。
        import_module("vllm_ascend.vllm_ascend_C")
        from .qwen_gdn_ops import install_missing_chunk_output_op

        if install_missing_chunk_output_op():
            logger.info("Using plugin-owned Triton Qwen GDN output operator")
    install_grouped_topk_router_compat()
