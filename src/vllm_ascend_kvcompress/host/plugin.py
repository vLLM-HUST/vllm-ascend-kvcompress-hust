# SPDX-License-Identifier: Apache-2.0
"""通过显式启用的 vllm.general_plugins 入口注册当前 vLLM Ascend 插件。"""

from __future__ import annotations

import os
import sys
from contextlib import nullcontext
from contextvars import ContextVar
from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec, PathFinder
from types import ModuleType
from typing import Any

from vllm.logger import logger

from ..core.config import extension_enabled, load_runtime_selection
from ..methods import create_backend
from ..runtime.provider import RUNNER_PROVIDER_ATTRIBUTE, AscendKVCompressionProvider
from .contracts import supports_mtp2
from .model_config import model_shape_from_config

_PATCH_MARKER = "_ascend_kvcompress_patch_v4"
_RUNNER_MODULE = "vllm_ascend.worker.model_runner_v1"
_runner_patch_finder: _RunnerPatchFinder | None = None
_TRITON_GLUON_MODULES = (
    "triton.experimental",
    "triton.experimental.gluon",
    "triton.experimental.gluon.language",
    "triton.experimental.gluon.nvidia",
)
_MESSAGE_QUEUE_MODULE = "vllm.distributed.device_communicators.shm_broadcast"
_MESSAGE_QUEUE_PATCH_MARKER = "_ascend_kvcompress_env_chunk_default"
# 仅在一次 attention metadata 构建调用中传播当前 provider。
# GDN builder 与 full-attention builder 共用上游元数据，但两者需要不同的长度语义；
# 用调用作用域而非全局 runner 引用，确保嵌套调用后能恢复原上下文。
_METADATA_PROVIDER: ContextVar[AscendKVCompressionProvider | None] = ContextVar(
    "ascend_kvcompress_metadata_provider", default=None
)


# 阅读入口：pyproject.toml 的 vllm.general_plugins 指向这里。
# 安装包本身不会开启压缩；显式启用后才给 scheduler、cache manager 和 runner 打补丁。
# 算法实现由 selection.method 决定，
# 补丁主要负责把宿主生命周期转交给 stateful/provider。
def register() -> None:
    """仅在直接配置或 Extension Manager 明确启用插件时安装钩子。"""
    if not extension_enabled():
        return
    selection = load_runtime_selection()
    logger.info(
        "Ascend KV compression plugin activated provider=%s method=%s",
        selection.provider_name,
        selection.method,
    )
    _prepare_current_triton_runtime()
    _configure_message_queue_defaults()

    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.scheduler import Scheduler

    _install_lazy_runner_hook(selection)

    from ..runtime.scheduler import install_stateful_compression_hooks

    install_stateful_compression_hooks(Scheduler, KVCacheManager, selection)


# runner 延迟导入链：Finder 定位模块 → Loader 执行原 loader → 安装 runner hooks。
# 这样 API/scheduler 进程注册插件时不必提前加载 NPU worker 的完整依赖。
class _RunnerPatchLoader(Loader):
    def __init__(self, wrapped: Loader, selection: Any) -> None:
        self.wrapped = wrapped
        self.selection = selection

    def create_module(self, spec: ModuleSpec) -> ModuleType | None:
        create = getattr(self.wrapped, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module: ModuleType) -> None:
        from .compat import install_removed_routed_experts_types

        install_removed_routed_experts_types()
        self.wrapped.exec_module(module)
        _install_runner_hooks(module.NPUModelRunner, self.selection)


class _RunnerPatchFinder(MetaPathFinder):
    """仅包装 Ascend runner 加载器，避免在 API 进程中导入 runner。"""

    def __init__(self, selection: Any) -> None:
        self.selection = selection

    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        del target
        if fullname != _RUNNER_MODULE:
            return None
        spec = PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        spec.loader = _RunnerPatchLoader(spec.loader, self.selection)
        return spec


def _install_lazy_runner_hook(selection: Any) -> None:
    global _runner_patch_finder
    loaded = sys.modules.get(_RUNNER_MODULE)
    if loaded is not None and hasattr(loaded, "NPUModelRunner"):
        _install_runner_hooks(loaded.NPUModelRunner, selection)
        return
    if _runner_patch_finder is None:
        _runner_patch_finder = _RunnerPatchFinder(selection)
        sys.meta_path.insert(0, _runner_patch_finder)


def _install_runner_hooks(runner_cls: type[Any], selection: Any) -> None:
    if runner_cls.__dict__.get(_PATCH_MARKER, False):
        return
    if (
        runner_cls.__module__ == _RUNNER_MODULE
        and runner_cls.__name__ == "NPUModelRunner"
    ):
        from .compat import (
            install_dcp_length_alias,
            install_removed_async_output_routing_field,
            install_removed_output_routing_field,
            install_step3p5_fp32_linear_alias,
        )

        install_step3p5_fp32_linear_alias()
        install_dcp_length_alias()
        install_removed_output_routing_field()
        install_removed_async_output_routing_field()
        _install_gdn_metadata_hook()
    # 当前 vLLM 已将旧的 xdrope_section 检查并入 uses_mrope，
    # 但配套 Ascend runner 仍有三条路径读取旧属性。
    # 仅在宿主不再定义该属性时补上中性的旧版默认值；
    # 在已验证的 vLLM 版本中，使用 xdrope 的模型
    # 已由 uses_mrope 表示。
    if not hasattr(runner_cls, "uses_xdrope_dim"):
        runner_cls.uses_xdrope_dim = 0
    # 当前 Ascend runner 即使关闭路由专家捕获，也会读取此标志。
    # 其初始化逻辑只在启用分支设置该标志，
    # 而本插件明确不支持这项可选输出。
    if not hasattr(runner_cls, "routed_experts_initialized"):
        runner_cls.routed_experts_initialized = False
    # 包装器只分发通用生命周期事件；观测与 metadata 策略由选中的 provider 决定。
    # 这里不再根据 Query 或逐层能力标志选择执行路径。
    original_initialize = runner_cls.initialize_kv_cache
    original_load_model = runner_cls.load_model
    original_update = runner_cls._update_states
    original_metadata = runner_cls._build_attention_metadata
    original_sample = runner_cls.sample_tokens
    from .compat import qwen_gdn_list_compat_enabled

    gdn_list_compat = qwen_gdn_list_compat_enabled()
    original_execute_model = runner_cls.execute_model

    # v2 加载顺序：解析 Backend → 选择 provider → 宿主 ABI 准备 → 算法产物准备，
    # 随后才加载服务权重。此处不创建 method，调度端也不需要读取校准统计。
    def load_model(runner: Any) -> Any:
        from .compat import prepare_model_load

        config = runner.vllm_config
        backend = create_backend(
            selection.method,
            selection.method_config,
            config,
            model_shape_from_config(config.model_config),
        )
        provider = backend.create_provider()
        setattr(runner, RUNNER_PROVIDER_ATTRIBUTE, provider)
        prepare_model_load(config)
        provider.before_model_load(runner)
        result = original_load_model(runner)
        provider.after_model_load(runner)
        return result

    def initialize_kv_cache(
        runner: Any, kv_cache_config: Any, *args: Any, **kwargs: Any
    ) -> Any:
        # provider 生命周期与一个 runner 绑定。
        # 先检查兼容性，再让宿主分配 KV，最后绑定真实 cache view；
        # 不能用分配前传入的计划替代 runner 规范化后的配置。
        provider = getattr(runner, RUNNER_PROVIDER_ATTRIBUTE)
        provider.before_kv_cache_initialize(runner)
        from .compat import (
            install_removed_model_routing_flag,
            qwen_mtp2_cache_binding_bridge,
        )

        install_removed_model_routing_flag(runner.vllm_config)
        # 只在实验 MTP2 的 KV 绑定期间临时使用重复层号桥。
        # 退出 with 后恢复原 binder，避免启动兼容动作成为整个进程的长期行为。
        bridge = (
            qwen_mtp2_cache_binding_bridge()
            if supports_mtp2(runner.vllm_config)
            else nullcontext()
        )
        with bridge:
            result = original_initialize(runner, kv_cache_config, *args, **kwargs)
        # Ascend 会先深拷贝并规范化传入的缓存方案，
        # 再保存真正对应已分配张量的缓存配置。
        provider.bind_model_runner(runner, runner.kv_cache_config)
        _install_runtime_slot_mapping_hooks(runner, provider.attention_group_index)
        return result

    # 原更新前确认上一轮目的块，原更新后清理消失请求并按新批次行同步偏移。
    # 这两个时点分别依赖旧请求镜像和宿主已经完成的行重排，不能合并。
    def update_states(runner: Any, scheduler_output: Any) -> Any:
        provider = getattr(runner, RUNNER_PROVIDER_ATTRIBUTE, None)
        if provider is not None:
            provider.before_update_states(scheduler_output)
        result = original_update(runner, scheduler_output)
        if provider is not None:
            provider.after_update_states(scheduler_output)
        return result

    def build_attention_metadata(runner: Any, *args: Any, **kwargs: Any) -> Any:
        provider = getattr(runner, RUNNER_PROVIDER_ATTRIBUTE, None)
        if provider is not None:
            provider.apply_physical_attention_lengths()
        # full attention 在此之前切到物理长度；
        # GDN builder 通过上下文取回保存的语义长度。
        # finally 只复位 ContextVar，并不把 runner 的长度张量恢复为语义值。
        token = _METADATA_PROVIDER.set(provider)
        try:
            result = original_metadata(runner, *args, **kwargs)
        finally:
            _METADATA_PROVIDER.reset(token)
        if provider is not None:
            num_tokens = args[0] if args else kwargs["num_tokens"]
            num_reqs = args[1] if len(args) > 1 else kwargs["num_reqs"]
            # 默认直接返回统一 metadata；下游 provider 可在此创建逐层或其他专用视图。
            # 包装层只传递结果和批次尺寸，不判断下游能力标志。
            return provider.after_attention_metadata(result, num_tokens, num_reqs)
        return result

    # 先保存对应调度输出，原采样方法可能消费或清空 execute_model_state。
    # 采样成功后才能执行本步压缩；失败则通知 provider 丢弃尚未发布的观测。
    def sample_tokens(runner: Any, grammar_output: Any) -> Any:
        state = runner.execute_model_state
        scheduler_output = state[0] if state is not None else None
        provider = getattr(runner, RUNNER_PROVIDER_ATTRIBUTE, None)
        try:
            # 本步 forward 已完成后才执行选取/拷贝；
            # 返回采样结果时 worker 已提交写入任务。
            # 这里不切换 scheduler 的块表，
            # 事务确认和旧块释放由 runtime/scheduler.py 的输出屏障控制。
            output = original_sample(runner, grammar_output)
            if provider is not None and scheduler_output is not None:
                provider.compress_scheduled_requests(scheduler_output)
            return output
        except Exception:
            if provider is not None and scheduler_output is not None:
                provider.on_step_error(scheduler_output)
            raise

    # GDN list 缓存属于宿主 ABI 兼容，每步单独清空。
    # 其余步骤统一交给 provider：前置观测、正常返回后的处理、异常时的清理。
    def execute_model(runner: Any, *args: Any, **kwargs: Any) -> Any:
        if gdn_list_compat:
            from .compat import clear_qwen_gdn_list_compat_cache

            clear_qwen_gdn_list_compat_cache()
        provider = getattr(runner, RUNNER_PROVIDER_ATTRIBUTE, None)
        scheduler_output = args[0] if args else kwargs["scheduler_output"]
        try:
            if provider is not None:
                provider.before_execute_model(scheduler_output)
            result = original_execute_model(runner, *args, **kwargs)
            if provider is not None:
                return provider.after_execute_model(scheduler_output, result)
            return result
        except Exception:
            if provider is not None:
                provider.on_step_error(scheduler_output)
            raise

    # 保留原方法供检查与复用，再用 marker 防止同一类被多层重复包装。
    # 这些属性保存的是宿主方法，不是下游方法能力或每请求状态。
    setattr(runner_cls, f"{_PATCH_MARKER}_original_initialize", original_initialize)
    setattr(runner_cls, f"{_PATCH_MARKER}_original_load_model", original_load_model)
    setattr(runner_cls, f"{_PATCH_MARKER}_original_update", original_update)
    setattr(runner_cls, f"{_PATCH_MARKER}_original_metadata", original_metadata)
    setattr(runner_cls, f"{_PATCH_MARKER}_original_sample", original_sample)
    setattr(
        runner_cls, f"{_PATCH_MARKER}_original_execute_model", original_execute_model
    )
    runner_cls.load_model = load_model
    runner_cls.initialize_kv_cache = initialize_kv_cache
    runner_cls._update_states = update_states
    runner_cls._build_attention_metadata = build_attention_metadata
    runner_cls.sample_tokens = sample_tokens
    runner_cls.execute_model = execute_model
    setattr(runner_cls, _PATCH_MARKER, True)


# full attention 的 KV 长度可以缩短，但 GDN 的循环状态时间轴不能回退。
# 构建 GDN metadata 时从当前调用上下文取回原长度，其他字段沿用宿主生成值。
def _install_gdn_metadata_hook() -> None:
    """构造混合元数据时，保持循环 GDN 元数据处于语义坐标系。"""
    from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionMetadataBuilder

    marker = "_ascend_kvcompress_semantic_lengths_v4"
    if AscendGDNAttentionMetadataBuilder.__dict__.get(marker, False):
        return
    original = AscendGDNAttentionMetadataBuilder.build

    def build(
        builder: Any,
        common_prefix_len: int,
        common_attn_metadata: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        provider = _METADATA_PROVIDER.get()
        if provider is not None:
            common_attn_metadata = provider.semantic_gdn_metadata(common_attn_metadata)
        return original(
            builder, common_prefix_len, common_attn_metadata, *args, **kwargs
        )

    AscendGDNAttentionMetadataBuilder.build = build
    setattr(AscendGDNAttentionMetadataBuilder, marker, True)


def _prepare_current_triton_runtime() -> None:
    """预加载当前 Ascend JIT 特化器需要的模块。

    Triton Ascend 3.6 的原生参数特化器从 Gluon 命名空间解析描述符类型。
    即使 wheel 已包含对应包，首次 RoPE 启动前也必须先加载 NVIDIA
    描述符命名空间。该兼容措施仅作用于启用插件的进程；
    若缺少预期模块，则报告可用于排查安装问题的错误。
    """
    # Triton Ascend 3.2 不提供 Gluon API，也不使用其描述符特化器。
    # Frontier 宿主仍搭载该运行时，
    # 因此保留普通 JIT 路径，
    # 不导入仅属于 3.6 及更新版本的命名空间。
    # Triton 3.2 与较新版本的模块布局不同；此分支只处理 JIT 导入条件。
    # 导入 NVIDIA 描述符命名空间不代表在 NVIDIA 设备上执行推理。
    triton_module = sys.modules.get("triton")
    triton_version = str(getattr(triton_module, "__version__", ""))
    if triton_version.startswith("3.2."):
        return

    # vLLM-Ascend 的旧版兼容路径会创建 __path__ 为空的父模块。
    # Triton Ascend 的发行包名为 triton-ascend，
    # 因此宿主对 triton 发行包的检测即使在 3.6 上也可能选中旧路径。
    # 仅清除这些明确可识别的空占位模块，
    # 然后从 wheel 中导入真实包。
    legacy_stub_present = any(
        getattr(sys.modules.get(name), "__path__", None) == []
        for name in _TRITON_GLUON_MODULES[:-1]
    )
    if legacy_stub_present:
        for name in reversed(_TRITON_GLUON_MODULES):
            sys.modules.pop(name, None)

    try:
        for name in _TRITON_GLUON_MODULES:
            import_module(name)
    except ImportError as error:
        raise RuntimeError(
            f"the validated Triton Ascend runtime is incomplete: cannot import {name}"
        ) from error


def _configure_message_queue_defaults() -> None:
    """将 vLLM 显式配置的消息队列分块大小应用于所有本地队列。

    当前 vLLM 仅对调度器广播队列应用 VLLM_MQ_MAX_CHUNK_BYTES_MB，
    每个 worker 的响应队列仍以 MessageQueue(1, 1) 使用默认 24 MiB 分块。
    十个分块会占用 240 MiB 的 /dev/shm，即使操作人员已为低并发容器
    显式选择更小的分块。这里用同一环境变量统一未传入的默认值。
    超过分块容量的消息已有 socket 回退路径，因此这里只改变共享内存
    快速路径的容量，不改变允许传输的消息大小。
    """
    # 仅当操作者显式设置 MQ chunk 大小时，才统一 worker 响应队列的默认值。
    # 修改的是共享内存快路径容量，大消息仍由宿主的 socket 回退机制传输。
    configured = os.getenv("VLLM_MQ_MAX_CHUNK_BYTES_MB")
    if configured is None:
        return
    try:
        chunk_mebibytes = int(configured)
    except ValueError as error:
        raise RuntimeError("VLLM_MQ_MAX_CHUNK_BYTES_MB must be an integer") from error
    if chunk_mebibytes <= 0:
        raise RuntimeError("VLLM_MQ_MAX_CHUNK_BYTES_MB must be positive")

    queue_cls = import_module(_MESSAGE_QUEUE_MODULE).MessageQueue
    if queue_cls.__dict__.get(_MESSAGE_QUEUE_PATCH_MARKER, False):
        return
    initializer = queue_cls.__init__
    defaults = initializer.__defaults__
    # 当前 MessageQueue 的默认参数依次为 local_reader_ranks、
    # max_chunk_bytes、max_chunks、connect_ip。若宿主接口改变则报错，
    # 避免静默修改错误的参数。
    if defaults is None or len(defaults) != 4 or defaults[2] != 10:
        raise RuntimeError("unsupported vLLM MessageQueue constructor defaults")
    updated = list(defaults)
    updated[1] = chunk_mebibytes * 1024 * 1024
    initializer.__defaults__ = tuple(updated)
    setattr(queue_cls, _MESSAGE_QUEUE_PATCH_MARKER, True)
    logger.info(
        "Applied VLLM_MQ_MAX_CHUNK_BYTES_MB=%d to worker response queues",
        chunk_mebibytes,
    )


def _install_runtime_slot_mapping_hooks(
    runner: Any, attention_group_index: int = 0
) -> None:
    """为当前 Ascend runner 实际分配的块表安装钩子。

    vLLM-Ascend 使用设备专用的 BlockTable 实现，仅修改通用 vLLM
    块表无法覆盖实际槽位映射入口。这里从已初始化的 runner 获取块表；
    缓存组及组下标由此前的 provider.bind_model_runner 验证；
    此处只检查槽位映射方法这一新增的宿主 ABI 要求。
    """
    block_table = runner.input_batch.block_table.block_tables[attention_group_index]
    if not callable(getattr(block_table, "compute_slot_mapping", None)):
        raise RuntimeError("Ascend block table does not expose compute_slot_mapping")
    _install_slot_mapping_hook(type(block_table))


# 仅绑定了 provider 的 full-attention 表做位置偏移；同类的其他表保持原路径。
# 替换传给原方法的 positions 引用，不原地改 runner.positions，RoPE 仍用语义位置。
def _install_slot_mapping_hook(block_table_cls: type[Any]) -> None:
    marker = f"{_PATCH_MARKER}_slot_mapping"
    if block_table_cls.__dict__.get(marker, False):
        return
    original = block_table_cls.compute_slot_mapping

    def compute_slot_mapping(
        block_table: Any,
        num_reqs: int,
        query_start_loc: Any,
        positions: Any,
    ) -> None:
        provider = getattr(block_table, RUNNER_PROVIDER_ATTRIBUTE, None)
        if provider is not None:
            positions = provider.physical_positions_for_slot_mapping(positions)
        return original(block_table, num_reqs, query_start_loc, positions)

    setattr(block_table_cls, f"{marker}_original", original)
    block_table_cls.compute_slot_mapping = compute_slot_mapping
    setattr(block_table_cls, marker, True)
