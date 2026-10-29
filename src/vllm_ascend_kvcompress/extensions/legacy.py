# SPDX-License-Identifier: Apache-2.0
"""已停止维护的迁移适配器，计划删除。

Query 捕获与逐层物理元数据应由下游仓库负责。本模块临时保留这些
实现，供压缩后端显式选择；TriAttention 不导入它，也不接受
v1 工厂。可继承 LegacyExtensionBackend 并实现 create_method，
但应将适配代码迁入下游仓库，避免新增对本模块的依赖。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
from vllm.logger import logger

from ..methods.backend import CompressionBackend
from ..methods.base import (
    CompressionRequest,
    CompressionResult,
    KVCompressionMethod,
    LayerCache,
)
from ..runtime.provider import AscendKVCompressionProvider, PendingCompression
from ..runtime.transaction import CompressionAnchor, CompressionPlan
from .legacy_hooks import (
    install_attention_query_hook,
    install_layer_aware_fia_graph_replay,
)

# 以下类型和状态只属于显式启用的迁移适配器，默认核心不会导入此模块。
# 后续计划删除且不再维护；下游应把所需实现迁到自己的仓库。
ATTENTION_QUERY_PROVIDER_ATTRIBUTE = "_ascend_kvcompress_query_provider"


@dataclass(frozen=True)
class QueryBatchSpan:
    request_id: str
    start: int
    end: int


@dataclass(frozen=True)
# 只描述本次观测对应的请求、授权计划和完整层集合，不携带 Q 张量。
# 真正的 Q 缓冲由算法持有，可跨 chunk 累积；发布和清理由 provider 协调。
class QueryObservation:
    request_id: str
    plan: CompressionPlan
    semantic_num_tokens: int
    window_tokens: int
    layer_indices: tuple[int, ...]


@dataclass(frozen=True)
# 扩展输入才携带 CPU 页表、计划和逐层长度，通用算法输入不承担这些字段。
# 逐层长度是本次 forward 后的物理值，不是上一轮压缩时的锚点值。
class LayerCompressionRequest(CompressionRequest):
    plan: CompressionPlan
    source_block_ids: tuple[tuple[int, ...], ...]
    destination_block_ids: tuple[tuple[int, ...], ...]
    per_layer_physical_num_tokens: tuple[tuple[str, int], ...] | None = None


@dataclass(frozen=True)
# physical_num_tokens 是组容量上限，逐层长度可以小于它。
# 即使本次各层同长，选择这个适配器的算法也应返回该扩展结果类型。
class LayerCompressionResult(CompressionResult):
    per_layer_physical_num_tokens: tuple[tuple[str, int], ...] | None = None


@dataclass(frozen=True)
class LayerPendingCompression(PendingCompression):
    per_layer_physical_num_tokens: tuple[tuple[str, int], ...] | None = None


@dataclass(frozen=True)
class LayerActiveCompression(CompressionAnchor):
    per_layer_physical_num_tokens: tuple[tuple[str, int], ...] | None = None


class QueryCompressionMethod(KVCompressionMethod):
    """可选观测协议，应迁入下游算法实现。"""

    def capture_query(self, layer, query, spans) -> None:
        raise NotImplementedError

    def complete_query_observation(self, observation) -> None:
        raise NotImplementedError

    def discard_query_observation(self, request_id) -> None:
        raise NotImplementedError


# 用这个 Backend 显式选择旧扩展；它不恢复 v1 工厂协议。
# 能力放在静态 Backend 中，图捕获前即可决定是否需要逐层稳定缓冲。
class LegacyExtensionBackend(CompressionBackend):
    """必须显式选择的扩展，这些属性不属于核心 API。"""

    query_window_tokens = 0
    query_layer_indices: tuple[int, ...] | None = None
    requires_per_layer_physical_state = False

    def create_provider(self):
        logger.warning(
            "Query/per-layer adapter is unmaintained and scheduled for deletion; "
            "move its implementation into your downstream repository"
        )
        return LegacyExtensionProvider(self)

    def create_scheduler(self, scheduler):
        if (
            not 0
            <= self.query_window_tokens
            <= self.runtime_spec.required_recompute_tokens
        ):
            raise ValueError("query window exceeds the recomputed suffix")
        return super().create_scheduler(scheduler)


class LegacyExtensionProvider(AscendKVCompressionProvider):
    """临时持有从核心移除的全部 Query 与逐层状态。

    选用此适配器的算法即使各层长度相同，也应返回 LayerCompressionResult。
    额外协议仅定义在本模块中，核心的统一长度结果不含逐层字段。
    """

    def __init__(self, backend: LegacyExtensionBackend) -> None:
        super().__init__(backend)
        self.query_window_tokens = backend.query_window_tokens
        self.requires_per_layer_physical_state = (
            backend.requires_per_layer_physical_state
        )
        self.query_layer_caches: tuple[LayerCache, ...] = ()
        self.speculative_cache_layer: LayerCache | None = None
        self._has_per_layer_rows = False
        self._per_layer_metadata_seen = False
        self._per_layer_step_requires_metadata = False
        self._per_layer_slot_buffers: dict[str, torch.Tensor] = {}
        self._query_step_output: Any | None = None
        self._query_seen_layers: set[int] = set()
        self._query_forward_complete = False
        self._query_tracked_ids: set[str] = set()

    # 逐层 FIA 开关和 Query 后端 hook 必须在模型图捕获前准备好。
    # 此处只安装机制，真实 cache 还未分配，不能在这里推断设备页表。
    def before_model_load(self, runner: Any) -> None:
        super().before_model_load(runner)
        if self.requires_per_layer_physical_state:
            install_layer_aware_fia_graph_replay(self.vllm_config)
        if self.query_window_tokens:
            install_attention_query_hook()

    # 先从绑定层中确定需要观测的目标层，再单独识别 MTP 辅助 cache。
    # MTP 层可能与目标层共享数字层号，不能仅凭 layer_index 当作同一层。
    def after_cache_bind(self) -> None:
        self.query_layer_caches = self._resolve_query_layer_caches()
        mtp = tuple(c for c in self.layer_caches if ".mtp.layers." in f".{c.name}.")
        if self.vllm_config.speculative_config is not None:
            if len(mtp) != 1:
                raise RuntimeError(
                    "MTP2 requires exactly one auxiliary attention cache"
                )
            self.speculative_cache_layer = mtp[0]
        if self.requires_per_layer_physical_state:
            self._validate_per_layer_host()
            self._per_layer_slot_buffers = {
                c.name: torch.empty(
                    self.runner.max_num_tokens,
                    dtype=torch.int64,
                    device=self.runner.device,
                )
                for c in self.layer_caches
            }
        for cache in self.query_layer_caches:
            layer = self.runner.compilation_config.static_forward_context[cache.name]
            setattr(layer, ATTENTION_QUERY_PROVIDER_ATTRIBUTE, self)

    def discard_request(self, request_id: str) -> None:
        super().discard_request(request_id)
        if self.query_window_tokens:
            self.method.discard_query_observation(request_id)
            self._query_tracked_ids.discard(request_id)

    def after_update_states(self, scheduler_output: Any) -> None:
        super().after_update_states(scheduler_output)
        # 组偏移不变也可能出现逐层长度变化，因此每步按当前批次重新检查此标记。
        self._has_per_layer_rows = any(
            self.active[r].per_layer_physical_num_tokens is not None
            for r in self.runner.input_batch.req_ids[: self.runner.input_batch.num_reqs]
            if r in self.active
        )

    def before_execute_model(self, scheduler_output: Any) -> None:
        self.begin_per_layer_step(scheduler_output)
        self.begin_query_step(scheduler_output)

    def after_execute_model(self, scheduler_output: Any, result: Any) -> Any:
        # 同时检查逐层 metadata 路径被执行、Query 层覆盖完整。
        # 只有两个检查均正常返回，本步的后续采样和压缩才能继续。
        self.finish_per_layer_step()
        self.finish_query_step()
        return result

    def on_step_error(self, scheduler_output: Any) -> None:
        self._per_layer_step_requires_metadata = False
        self.abort_query_step(scheduler_output)

    def after_attention_metadata(self, result, num_tokens, num_reqs):
        return self.per_layer_attention_metadata(result, num_tokens, num_reqs)

    def build_compression_request(
        self, request_id, semantic, physical, source_device, destination_device, plan
    ):
        # 从上一轮各层锚点加上共同的语义增量，得到本轮各层有效长度。
        # 这里构造的是扩展输入，不会把逐层表写回通用 scheduler 的统一预算。
        active = self.active.get(request_id)
        lengths = (
            tuple(
                (name, length + semantic - active.semantic_anchor)
                for name, length in active.per_layer_physical_num_tokens
            )
            if active is not None and active.per_layer_physical_num_tokens is not None
            else None
        )
        return LayerCompressionRequest(
            request_id,
            semantic,
            physical,
            source_device,
            destination_device,
            plan,
            (plan.source_block_ids,),
            (plan.destination_block_ids,),
            lengths,
        )

    def compress_request(self, request, plan):
        if self.query_window_tokens:
            if not self._query_forward_complete:
                raise RuntimeError("query observation requires a completed forward")
            self.method.complete_query_observation(
                QueryObservation(
                    request.request_id,
                    plan,
                    request.semantic_num_tokens,
                    self.query_window_tokens,
                    tuple(layer.layer_index for layer in self.query_layer_caches),
                )
            )
        try:
            return self.method.compress(request)
        finally:
            if self.query_window_tokens:
                self.method.discard_query_observation(request.request_id)
                self._query_tracked_ids.discard(request.request_id)

    # 先规范化逐层长度顺序，再与目的块一同暂存。
    # 只有 activate_pending 被调用时，新逐层锚点才与块表一起生效。
    def make_pending(
        self, plan: CompressionPlan, result: LayerCompressionResult
    ) -> LayerPendingCompression:
        lengths = self._validate_per_layer_lengths(
            result.per_layer_physical_num_tokens, result.physical_num_tokens
        )
        return LayerPendingCompression(
            plan.semantic_anchor,
            result.physical_num_tokens,
            plan.destination_block_ids,
            lengths,
        )

    def activate_pending(
        self, pending: LayerPendingCompression
    ) -> LayerActiveCompression:
        return LayerActiveCompression(
            pending.semantic_anchor,
            pending.physical_anchor,
            pending.per_layer_physical_num_tokens,
        )

    def _resolve_query_layer_caches(self) -> tuple[LayerCache, ...]:
        if not self.query_window_tokens:
            return ()
        # None 表示观察全部绑定层；显式层号则必须能唯一定位 cache。
        # 若辅助层和目标层的层号冲突，应选择不含冲突的目标观测层。
        requested = self.backend.query_layer_indices
        if requested is None:
            return self.layer_caches
        if (
            not isinstance(requested, tuple)
            or not requested
            or any(
                isinstance(index, bool) or not isinstance(index, int)
                for index in requested
            )
            or len(set(requested)) != len(requested)
        ):
            raise RuntimeError(
                "query_layer_indices must be a non-empty tuple of unique integers"
            )
        by_index: dict[int, LayerCache] = {}
        duplicate_indices: set[int] = set()
        for cache in self.layer_caches:
            if cache.layer_index in by_index:
                duplicate_indices.add(cache.layer_index)
            else:
                by_index[cache.layer_index] = cache
        if duplicate_indices & set(requested):
            raise RuntimeError(
                "query_layer_indices are ambiguous across bound cache layers"
            )
        if any(index not in by_index for index in requested):
            raise RuntimeError("query_layer_indices contain an unbound model layer")
        return tuple(by_index[index] for index in requested)

    def capture_attention_query(self, layer_name: str, query: torch.Tensor) -> None:
        """从注意力自定义算子的执行路径观察 query 行。

        Dynamo 跟踪分段图时会运行模块 forward 钩子，但编译图回放时
        不会再次调用它们。Ascend 拆分出的 unified_attention_with_output
        自定义算子则会在每次 prefill 回放时执行后端 forward，因此包装器
        在该处调用本方法。将请求区间保留在图外，也能避免把某次调度批次
        固化进编译程序。
        """
        if self._query_step_output is None:
            return
        cache = next(
            (cache for cache in self.query_layer_caches if cache.name == layer_name),
            None,
        )
        if cache is None:
            raise RuntimeError(
                f"attention layer {layer_name!r} is not a query-observation layer"
            )
        spans = self._query_spans()
        if not spans:
            return
        if query.ndim < 2 or max(span.end for span in spans) > query.shape[0]:
            raise RuntimeError(f"attention layer {cache.name!r} query rows changed")
        self.method.capture_query(cache, query, spans)
        self._query_tracked_ids.update(span.request_id for span in spans)
        self._query_seen_layers.add(cache.layer_index)

    # Q 张量按整批 token 拼接，query_start_loc 给出每个请求的起始行。
    # 只截取本步还属于 prompt 的部分，decode 行不能补进 prefill 观测窗口。
    def _query_spans(self) -> tuple[QueryBatchSpan, ...]:
        output = self._query_step_output
        if output is None or self.runner is None:
            return ()
        batch = self.runner.input_batch
        offsets = self.runner.query_start_loc.cpu
        spans: list[QueryBatchSpan] = []
        for request_id, scheduled in output.num_scheduled_tokens.items():
            request = self.runner.requests.get(request_id)
            index = batch.req_id_to_index.get(request_id)
            if request is None or index is None or scheduled <= 0:
                continue
            remaining = int(request.num_prompt_tokens) - int(
                request.num_computed_tokens
            )
            if remaining <= 0:
                continue
            start = int(offsets[index])
            end = start + min(int(scheduled), remaining)
            if end > int(offsets[index + 1]):
                raise RuntimeError("prefill query rows exceed the runner batch span")
            spans.append(QueryBatchSpan(request_id, start, end))
        return tuple(spans)

    def begin_query_step(self, scheduler_output: Any) -> None:
        if not self.query_window_tokens:
            return
        for request_id in self._query_tracked_ids - self.runner.requests.keys():
            self.method.discard_query_observation(request_id)
            self._query_tracked_ids.discard(request_id)
        # 只保存当前步骤的调度上下文；Q 的复制在 attention 后端实际执行时发生。
        # 步骤结束后清空上下文，防止后续非本步调用误用旧的请求行范围。
        self._query_step_output = scheduler_output
        self._query_seen_layers.clear()
        self._query_forward_complete = False

    def finish_query_step(self) -> None:
        if not self.query_window_tokens:
            return
        try:
            if self._query_spans() and self._query_seen_layers != {
                layer.layer_index for layer in self.query_layer_caches
            }:
                raise RuntimeError(
                    "query observation missed a full-attention layer; "
                    "the current graph replay path is unvalidated"
                )
            self._query_forward_complete = True
        finally:
            self._query_step_output = None

    def abort_query_step(self, scheduler_output: Any) -> None:
        if not self.query_window_tokens:
            return
        self._query_step_output = None
        self._query_forward_complete = False
        self._query_seen_layers.clear()
        for request_id in scheduler_output.num_scheduled_tokens:
            self.method.discard_query_observation(request_id)
            self._query_tracked_ids.discard(request_id)

    def _validate_per_layer_lengths(
        self, lengths: tuple[tuple[str, int], ...] | None, maximum: int
    ) -> tuple[tuple[str, int], ...] | None:
        if lengths is None:
            return None
        # 逐层结果必须覆盖所有搬运层，包括存在时的辅助 attention cache。
        # 先按层名核对，再按绑定顺序排列，避免返回顺序影响后续长度计算。
        names = tuple(layer.name for layer in self.layer_caches)
        if not isinstance(lengths, tuple) or len(lengths) != len(names):
            raise RuntimeError(
                "per-layer physical lengths must cover every full-attention layer"
            )
        values: dict[str, int] = {}
        for entry in lengths:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise RuntimeError("invalid per-layer physical length entry")
            name, length = entry
            if not isinstance(name, str) or name in values or name not in names:
                raise RuntimeError(
                    "per-layer physical lengths contain an unknown or duplicate layer"
                )
            if (
                isinstance(length, bool)
                or not isinstance(length, int)
                or not 0 < length <= maximum
            ):
                raise RuntimeError(
                    "per-layer physical lengths must be positive and <= group maximum"
                )
            values[name] = length
        ordered = tuple((name, values[name]) for name in names)
        if any(length != maximum for _, length in ordered):
            # 声明逐层状态的后端已在绑定时验证；未声明的后端在接纳结果时验证。
            if not self.requires_per_layer_physical_state:
                self._validate_per_layer_host()
            return ordered
        # 包括 TriAttention 在内的统一长度方法保留原有快速路径。
        return None

    def _validate_per_layer_host(self) -> None:
        # 仅由绑定后的初始化和算法结果接纳边界调用。
        mode = getattr(
            getattr(self.runner, "compilation_config", None), "cudagraph_mode", None
        )
        mode_name = getattr(mode, "name", None)
        if mode_name == "FULL_AND_PIECEWISE":
            if not self.requires_per_layer_physical_state:
                raise RuntimeError(
                    "graph replay requires the method to declare per-layer "
                    "physical state before capture"
                )
            try:
                from vllm_ascend.attention.attention_v1 import (
                    AscendAttentionBackendImpl,
                )
            except (ImportError, OSError, RuntimeError) as error:
                raise RuntimeError(
                    "per-layer graph replay backend is unavailable"
                ) from error
            if not getattr(
                AscendAttentionBackendImpl,
                "_ascend_kvcompress_layer_aware_graph_v1",
                False,
            ):
                raise RuntimeError(
                    "per-layer graph replay requires the layer-aware FIA update seam"
                )
        elif mode_name != "NONE":
            raise RuntimeError(
                "per-layer physical state supports eager or "
                "FULL_AND_PIECEWISE execution only"
            )
        if any(
            getattr(self.runner, name, False) for name in ("use_dcp", "pcp_enabled")
        ):
            raise RuntimeError(
                "per-layer physical state is unvalidated with "
                "context-parallel attention"
            )

    def per_layer_attention_metadata(
        self, result: Any, num_tokens: int, num_reqs: int
    ) -> Any:
        """为每个全注意力层提供各自的 KV 长度与写入槽位。"""
        if not (self.requires_per_layer_physical_state or self._has_per_layer_rows):
            return result
        # 宿主能力在 after_cache_bind 或接纳非统一长度结果时验证；
        # 每步只核对宿主返回的 metadata，不重复检查固定的运行配置。
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("per-layer attention metadata hook was bypassed")
        metadata, common = result
        if not isinstance(metadata, dict):
            raise RuntimeError(
                "per-layer attention requires a single eager metadata batch"
            )
        runner = self.runner
        batch = runner.input_batch
        request_ids = batch.req_ids[:num_reqs]
        if num_tokens < 0:
            raise RuntimeError("per-layer attention batch dimensions changed")
        # 图捕获会使用尚无真实请求的合成 batch；此时仍需让图引用逐层稳定地址。
        # 真正的部分缺行 batch 不能沿用这条捕获特例，否则会掩盖请求映射错误。
        if len(request_ids) != num_reqs:
            # FULL 图捕获会在 runner 接纳请求前构造合成解码批次。
            # 此时没有需要特殊处理的压缩请求行，
            # 但图中的缓存写入内核必须捕获与回放时相同的逐层槽位缓冲区。
            # 对于真实批次或仅部分填充的输入批次，
            # 仍须在状态不满足要求时报错。
            batch_num_reqs = int(getattr(batch, "num_reqs", len(batch.req_ids)))
            if self._has_active_rows or batch_num_reqs != 0 or batch.req_ids:
                raise RuntimeError("per-layer attention batch dimensions changed")
            for layer in self.layer_caches:
                base = metadata.get(layer.name)
                slots = self._per_layer_slot_buffers.get(layer.name)
                if (
                    base is None
                    or not isinstance(getattr(base, "slot_mapping", None), torch.Tensor)
                    or slots is None
                    or slots.numel() < num_tokens
                    or base.slot_mapping.numel() < num_tokens
                ):
                    raise RuntimeError("per-layer capture slot buffers are unavailable")
                slots[num_tokens:].fill_(-1)
                slots[:num_tokens].copy_(base.slot_mapping[:num_tokens])
                metadata[layer.name] = replace(base, slot_mapping=slots[:num_tokens])
                if layer is self.speculative_cache_layer:
                    if common is None:
                        raise RuntimeError("MTP2 capture metadata is unavailable")
                    common = replace(common, slot_mapping=slots[:num_tokens])
            self._per_layer_metadata_seen = True
            return metadata, common
        table = batch.block_table.block_tables[self.attention_group_index]
        native = getattr(
            type(table), "_ascend_kvcompress_patch_v4_slot_mapping_original", None
        )
        if not callable(native):
            raise RuntimeError(
                "per-layer attention requires the native slot mapping hook"
            )
        if self._has_active_rows and not self._physical_lengths_applied:
            raise RuntimeError("per-layer attention requires saved semantic lengths")
        global_offsets = tuple(
            int(value) for value in self._request_offsets_cpu[:num_reqs]
        )
        # 逐层重算会临时复用宿主的 group slot 表，所以先保留统一视图。
        # 每层最终 metadata 指向自己的持久缓冲，函数退出时恢复 group 表内容。
        original_slots = table.slot_mapping.gpu[:num_tokens].clone()
        speculative_view: (
            tuple[list[int], tuple[int, ...], torch.Tensor, torch.Tensor] | None
        ) = None
        try:
            for layer in self.layer_caches:
                offsets = tuple(
                    self.active[request_id].semantic_anchor
                    - dict(
                        self.active[request_id].per_layer_physical_num_tokens or ()
                    ).get(layer.name, self.active[request_id].physical_anchor)
                    if request_id in self.active
                    else 0
                    for request_id in request_ids
                )
                base = metadata.get(layer.name)
                if (
                    type(base).__name__ != "AscendMetadata"
                    or type(base).__module__ != "vllm_ascend.attention.attention_v1"
                    or base.seq_lens is None
                    or base.seq_lens_list is None
                    or base.slot_mapping is None
                ):
                    raise RuntimeError(
                        "per-layer attention backend metadata is unvalidated"
                    )
                # base 已减去组级 offset，因此层级修正是 global_offset -
                # layer_offset。
                # 例如全局保留 2048、该层只保留 1024，则 delta=-1024，
                # 使该层长度再缩短 1024。
                delta = [
                    global_offset - offset
                    for global_offset, offset in zip(
                        global_offsets, offsets, strict=True
                    )
                ]
                seq_lens = base.seq_lens.clone()
                if any(delta):
                    seq_lens[:num_reqs] += torch.tensor(
                        delta, dtype=seq_lens.dtype, device=seq_lens.device
                    )
                seq_lens_list = list(base.seq_lens_list)
                for index, correction in enumerate(delta):
                    seq_lens_list[index] += correction
                slots = self._per_layer_slot_buffers.get(layer.name)
                if slots is None and not self.requires_per_layer_physical_state:
                    slots = torch.empty(
                        max(num_tokens, int(getattr(runner, "max_num_tokens", 0))),
                        dtype=torch.int64,
                        device=original_slots.device,
                    )
                    self._per_layer_slot_buffers[layer.name] = slots
                if slots is None or slots.numel() < num_tokens:
                    raise RuntimeError(
                        "per-layer attention slot buffers were not prepared "
                        "before capture"
                    )
                # 回放使用的已捕获批次桶可能大于当前实际批次。
                # 填充行的缓存写入不得复用上一个请求的槽位。
                slots[num_tokens:].fill_(-1)
                if offsets == global_offsets:
                    slots[:num_tokens].copy_(original_slots)
                else:
                    positions = runner.positions[:num_tokens]
                    req_indices = runner.req_indices.gpu[:num_tokens]
                    offset_tensor = torch.tensor(
                        offsets, dtype=positions.dtype, device=positions.device
                    )
                    native(
                        table,
                        num_reqs,
                        runner.query_start_loc.gpu[: num_reqs + 1],
                        positions - offset_tensor[req_indices.long()],
                    )
                    slots[:num_tokens].copy_(table.slot_mapping.gpu[:num_tokens])
                metadata[layer.name] = replace(
                    base,
                    seq_lens=seq_lens,
                    seq_lens_cpu=seq_lens,
                    seq_lens_list=seq_lens_list,
                    slot_mapping=slots[:num_tokens],
                )
                if layer is self.speculative_cache_layer:
                    speculative_view = delta, offsets, seq_lens, slots[:num_tokens]
        # 复用原生 slot-mapping 方法会临时覆盖 group table，返回前必须恢复，
        # 让本函数外的 group 消费者继续看到统一视图；逐层 metadata 引用独立稳定缓冲。
        finally:
            table.slot_mapping.gpu[:num_tokens].copy_(original_slots)
        if self.speculative_cache_layer is not None:
            if speculative_view is None or common is None:
                raise RuntimeError("MTP2 per-layer attention metadata is incomplete")
            delta, offsets, _, slots = speculative_view
            common = self._speculative_common_metadata(
                common,
                delta,
                offsets,
                slots,
                num_reqs,
            )
        self._per_layer_metadata_seen = True
        return metadata, common

    @staticmethod
    def _speculative_common_metadata(
        common: Any,
        delta: list[int],
        offsets: tuple[int, ...],
        slots: torch.Tensor,
        num_reqs: int,
    ) -> Any:
        required = (
            "seq_lens",
            "_seq_lens_cpu",
            "seq_lens_cpu_upper_bound",
            "slot_mapping",
        )
        if any(not hasattr(common, name) for name in required):
            raise RuntimeError("MTP2 common attention metadata ABI changed")

        def adjusted(value: Any, corrections: list[int] | tuple[int, ...]) -> Any:
            if value is None:
                return None
            if not isinstance(value, torch.Tensor) or value.numel() < num_reqs:
                raise RuntimeError("MTP2 sequence-length metadata ABI changed")
            result = value.clone()
            if any(corrections):
                result[:num_reqs] += torch.tensor(
                    corrections,
                    dtype=result.dtype,
                    device=result.device,
                )
            return result

        # common metadata 还会被 MTP 草稿路径读取，必须同步到辅助 cache 的长度和槽位。
        # seq_lens 已做组级换算，使用 delta；computed tokens 则减完整层偏移。
        changes = {
            "seq_lens": adjusted(common.seq_lens, delta),
            "_seq_lens_cpu": adjusted(common._seq_lens_cpu, delta),
            "seq_lens_cpu_upper_bound": adjusted(
                common.seq_lens_cpu_upper_bound,
                delta,
            ),
            "slot_mapping": slots,
        }
        if hasattr(common, "seq_lens_cpu"):
            changes["seq_lens_cpu"] = adjusted(common.seq_lens_cpu, delta)
        if hasattr(common, "num_computed_tokens_cpu"):
            changes["num_computed_tokens_cpu"] = adjusted(
                common.num_computed_tokens_cpu,
                tuple(-offset for offset in offsets),
            )
        try:
            return replace(common, **changes)
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                "MTP2 common attention metadata cannot be specialized by layer"
            ) from error

    def begin_per_layer_step(self, scheduler_output: Any) -> None:
        enabled = self.requires_per_layer_physical_state or self._has_per_layer_rows
        scheduled = getattr(scheduler_output, "num_scheduled_tokens", {})
        # 没有实际调度 token 的空步骤不要求 metadata，避免异步空执行被误判为漏 hook。
        self._per_layer_step_requires_metadata = enabled and any(
            int(tokens) > 0 for tokens in scheduled.values()
        )
        if self._per_layer_step_requires_metadata:
            self._per_layer_metadata_seen = False

    def finish_per_layer_step(self) -> None:
        try:
            if (
                self._per_layer_step_requires_metadata
                and not self._per_layer_metadata_seen
            ):
                raise RuntimeError(
                    "per-layer attention metadata hook was bypassed by the host"
                )
        finally:
            self._per_layer_step_requires_metadata = False
