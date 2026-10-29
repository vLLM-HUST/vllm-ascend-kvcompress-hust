# SPDX-License-Identifier: Apache-2.0
"""统一长度的 worker 事务与可覆盖的运行时生命周期。

核心仅管理块表镜像以及语义、物理坐标；观测数据、逐层元数据
和策略状态由下游 provider 自行持有。本模块不导入或探测可选扩展。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
from vllm.logger import logger

from ..core.constants import KERNEL_CACHE_BLOCK_SIZE
from ..host.contracts import (
    allowed_scheduler_block_sizes,
    common_compatibility_reasons,
    find_full_attention_group,
    request_max_tokens,
    unpack_layer_cache,
    validate_cache_tensor,
)
from ..methods.backend import CompressionBackend
from ..methods.base import (
    CompressionRequest,
    CompressionResult,
    KVCompressionMethod,
    LayerCache,
)
from .coordinates import blocks_for_tokens, expand_scheduler_block_ids, shift_positions
from .transaction import PLAN_ATTRIBUTE, CompressionAnchor, CompressionPlan

RUNNER_PROVIDER_ATTRIBUTE = "_ascend_kvcompress_provider"


@dataclass(frozen=True)
# 这里只记录已写入目的块的长度锚点和页号，不代表块表已经切换。
# worker 不拥有分配器，pending 的取消也不会在此直接释放 scheduler 的块。
class PendingCompression(CompressionAnchor):
    """worker 已写入、正在等待调度器确认的目标块及其状态。"""

    block_ids: tuple[int, ...]


class AscendKVCompressionProvider:
    """单个 runner 的统一长度缓存状态，通过 backend 工厂替换子类。"""

    def __init__(self, backend: CompressionBackend) -> None:
        self.backend = backend
        self.vllm_config = backend.vllm_config
        self.model_shape = backend.model_shape
        self.runtime_spec = backend.runtime_spec
        self.tensor_parallel_size = int(
            self.vllm_config.parallel_config.tensor_parallel_size
        )
        self.attention_group_index = 0
        self.scheduler_block_size = int(self.vllm_config.cache_config.block_size)
        self.cache_block_size = KERNEL_CACHE_BLOCK_SIZE
        self.cache_blocks_per_scheduler_block = 1
        # 构造 provider 时仍处于静态阶段；method 保持空值，直到 KV 初始化前创建。
        # 每个 runner 复用一个算法实例，后续重新绑定 cache 不重复加载统计。
        self.method: KVCompressionMethod | None = None
        self.runner: Any | None = None
        self.layer_caches: tuple[LayerCache, ...] = ()
        # pending 是已写未确认的事务镜像；active 是最近一次已确认的长度映射。
        # 只需保存锚点差值，后续 token 可追加，不必保存每个历史 token 的位置映射。
        self.pending: dict[str, PendingCompression] = {}
        self.active: dict[str, CompressionAnchor] = {}
        # 这些持久缓冲按 input_batch 当前行排列，pending/active 则按请求 ID 索引。
        # 行增删和重排后必须同步偏移，不能把上一步行号当作请求身份。
        self._request_offsets_cpu: torch.Tensor | None = None
        self._request_offsets_device: torch.Tensor | None = None
        self._physical_positions: torch.Tensor | None = None
        self._semantic_seq_lens_device: torch.Tensor | None = None
        self._semantic_seq_lens_cpu: torch.Tensor | None = None
        self._offset_row_snapshot: tuple[int, ...] = ()
        self._has_active_rows = False
        self._physical_lengths_applied = False

    # 下游覆盖生命周期方法时，若仍使用通用运行机制，应调用 super 保留默认动作。
    # 模型加载前、KV 初始化前、cache 绑定后是不同阶段，不宜混用。
    def before_model_load(self, runner: Any) -> None:
        """加载服务权重前准备算法产物，此时尚未创建算法实例。"""
        self.runner = runner
        self.backend.prepare_model_load(runner)

    def after_model_load(self, runner: Any) -> None:
        """供下游覆盖以绑定模型，此时尚未分配 KV 缓存。"""

    # 到达这里时，加载前准备与服务模型加载均已结束。
    # 先创建算法并检查兼容性，随后 plugin 才调用宿主的 KV 分配函数。
    def before_kv_cache_initialize(self, runner: Any) -> None:
        """产物与模型准备完成后，仅构造一次 worker 算法实例。"""
        self.runner = runner
        if self.method is None:
            self.method = self.backend.create_method()
        self.validate_host(runner)

    # 通用层已绑定真实 KV view 和长度缓冲后，才调用这个扩展点。
    # 可选扩展在这里分配自己的观测或逐层 slot 状态，默认 provider 不分配它们。
    def after_cache_bind(self) -> None:
        """供下游覆盖，在通用绑定完成后分配其自行持有的状态。"""

    def before_execute_model(self, scheduler_output: Any) -> None:
        """供下游覆盖，在宿主前向执行前观察请求或批次。"""

    # 这里的完成表示原 execute_model 正常返回；压缩还要等采样后的授权执行点。
    # 返回值会直接交还宿主调用者，下游如不变换结果应原样返回。
    def after_execute_model(self, scheduler_output: Any, result: Any) -> Any:
        """供下游覆盖，观察已完成的前向执行或变换输出。"""
        return result

    def on_step_error(self, scheduler_output: Any) -> None:
        """释放下游临时状态，随后继续抛出宿主原始异常。"""

    def after_attention_metadata(
        self, result: Any, num_tokens: int, num_reqs: int
    ) -> Any:
        """供下游覆盖以调整宿主元数据，默认每个请求行使用统一长度。"""
        return result

    # 统一清理入口，供结束、抢占、恢复以及请求从 worker 消失时使用。
    # 下游若持有额外请求缓冲，应覆盖此方法清理自身状态并保留通用清理。
    def discard_request(self, request_id: str) -> None:
        """请求结束、被抢占、恢复或从批次消失时清理其状态。"""
        self.pending.pop(request_id, None)
        self.active.pop(request_id, None)

    def validate_host(self, runner: Any) -> None:
        reasons = list(common_compatibility_reasons(self.vllm_config, runner))
        reasons.extend(self.method.compatibility_reasons(runner))
        if reasons:
            raise RuntimeError(
                "Ascend KV compression is incompatible with this launch:\n- "
                + "\n- ".join(reasons)
            )

    def bind_model_runner(self, runner: Any, kv_cache_config: Any) -> None:
        """宿主分配缓存后，校验并绑定稠密 K/V 视图。"""
        from vllm.model_executor.models.utils import extract_layer_index

        # 当前只支持一个 full-attention group，允许伴随 Mamba/GDN group。
        # 压缩只改变 attention group 的块表与长度；循环状态组保持原语义时间轴。
        groups = kv_cache_config.kv_cache_groups
        self.attention_group_index = find_full_attention_group(groups)
        group = groups[self.attention_group_index]
        self.scheduler_block_size = int(group.kv_cache_spec.block_size)
        allowed_block_sizes = allowed_scheduler_block_sizes(self.vllm_config)
        if self.scheduler_block_size not in allowed_block_sizes:
            raise RuntimeError(
                "Ascend KV compression does not support logical KV block size "
                f"{self.scheduler_block_size}; allowed sizes are "
                f"{sorted(allowed_block_sizes)}"
            )
        if self.runtime_spec.max_physical_num_tokens % self.scheduler_block_size:
            raise RuntimeError(
                "compression target must be divisible by the logical KV block "
                f"size {self.scheduler_block_size}"
            )

        block_tables = runner.input_batch.block_table.block_tables
        if len(block_tables) != len(groups):
            raise RuntimeError("worker block-table groups do not match KV groups")
        attention_block_table = block_tables[self.attention_group_index]
        physical_block_size = int(
            getattr(
                attention_block_table,
                "physical_block_size",
                self.scheduler_block_size,
            )
        )
        if physical_block_size != self.scheduler_block_size:
            raise RuntimeError(
                "worker full-attention block table disagrees with its KV cache "
                "group block size"
            )
        self.cache_block_size = int(
            getattr(attention_block_table, "block_size", self.scheduler_block_size)
        )
        if (
            self.cache_block_size != KERNEL_CACHE_BLOCK_SIZE
            or self.scheduler_block_size % self.cache_block_size
        ):
            raise RuntimeError(
                "Ascend KV compression requires 128-token kernel cache blocks "
                "that evenly divide the logical KV block size"
            )
        # 一个 scheduler 逻辑页可能容纳多个 128-token cache 页（hybrid 模型）。
        # scheduler 的跨组 LCM 对齐、attention manager 页大小、内核页大小不能混用。
        self.cache_blocks_per_scheduler_block = (
            self.scheduler_block_size // self.cache_block_size
        )

        layer_caches: list[LayerCache] = []
        context = runner.compilation_config.static_forward_context
        for layer_name in group.layer_names:
            layer = context.get(layer_name)
            if layer is None:
                raise RuntimeError(f"attention layer {layer_name!r} is not bound")
            k_cache, v_cache = unpack_layer_cache(layer_name, layer)
            local_kv_heads = self.model_shape.num_kv_heads // self.tensor_parallel_size
            validate_cache_tensor(
                layer_name,
                "K",
                k_cache,
                self.model_shape,
                local_kv_heads,
                self.cache_block_size,
            )
            validate_cache_tensor(
                layer_name,
                "V",
                v_cache,
                self.model_shape,
                local_kv_heads,
                self.cache_block_size,
            )
            layer_caches.append(
                LayerCache(
                    name=layer_name,
                    layer_index=extract_layer_index(layer_name),
                    k_cache=k_cache,
                    v_cache=v_cache,
                )
            )

        self.runner = runner
        self.layer_caches = tuple(layer_caches)
        # 通用层只暴露已验证的 K/V view；算法层自行分配统计、评分与搬运 workspace。
        # 原 cache 可有页间 padding，验证允许较大的 stride(0)，
        # 不要求整个张量 contiguous。
        self.method.bind_model_runner(runner, self.layer_caches)
        # 按最大批次容量一次分配偏移缓冲，避免每步临时申请。
        # CPU 副本修正 optimistic lengths，设备副本修正 seq_lens 和写入位置。
        pin_memory = bool(getattr(runner, "pin_memory", False))
        self._request_offsets_cpu = torch.zeros(
            runner.max_num_reqs,
            dtype=torch.int64,
            device="cpu",
            pin_memory=pin_memory,
        )
        self._request_offsets_device = torch.zeros(
            runner.max_num_reqs, dtype=torch.int64, device=runner.device
        )
        self._physical_positions = torch.empty(
            runner.max_num_tokens, dtype=torch.int64, device=runner.device
        )
        # 这是每步修改 attention 长度前的快照，供 GDN 恢复语义时间轴。
        # 它不是请求全部历史长度的永久备份，下一步由宿主重新填充当前长度。
        self._semantic_seq_lens_device = torch.empty_like(runner.seq_lens)
        self._semantic_seq_lens_cpu = torch.empty_like(runner.optimistic_seq_lens_cpu)
        setattr(attention_block_table, RUNNER_PROVIDER_ATTRIBUTE, self)
        logger.info(
            "Ascend KV compression cache bound method=%s "
            "materialization_layers=%d calibrated_scoring_layers=%d "
            "threshold_tokens=%d target_tokens=%d min_output_tokens=%d "
            "logical_block_size=%d cache_block_size=%d",
            self.backend.name,
            len(self.layer_caches),
            len(getattr(self.method, "layer_caches", self.layer_caches)),
            self.runtime_spec.compression_threshold_tokens,
            self.runtime_spec.max_physical_num_tokens,
            self.runtime_spec.min_output_tokens_for_compression,
            self.scheduler_block_size,
            self.cache_block_size,
        )
        self.after_cache_bind()

    def before_update_states(self, scheduler_output: Any) -> None:
        """越过同步模型执行屏障后，提交上一步事务。"""
        if self.runner is None:
            return
        reset_ids = set(scheduler_output.finished_req_ids)
        reset_ids.update(getattr(scheduler_output, "preempted_req_ids", ()))
        reset_ids.update(scheduler_output.scheduled_cached_reqs.resumed_req_ids)
        for request_id in reset_ids:
            self.discard_request(request_id)

        # scheduler 接受输出并解除冻结后，请求再次被调度，worker 才跟进切表。
        # 其他请求可继续执行；不能因看到任意一个新 batch 就提交所有 pending。
        for request_id, pending in tuple(self.pending.items()):
            # 异步模式中其他请求可以继续执行，本请求在确认前被冻结。
            # 它再次出现在调度结果中，才意味着可以在 worker 端切换目的块。
            if request_id not in scheduler_output.num_scheduled_tokens:
                continue
            request = self.runner.requests.get(request_id)
            if request is None:
                self.discard_request(request_id)
                continue
            # 只替换全注意力组的页表；Mamba/GDN 组的块及其循环状态不跟着压缩。
            # 同步更新已在 input_batch 中的行，后续宿主更新会在新页表基础上追加块。
            mutable = list(pending.block_ids)
            group_block_ids = list(request.block_ids)
            group_block_ids[self.attention_group_index] = mutable
            request.block_ids = tuple(group_block_ids)
            request_index = self.runner.input_batch.req_id_to_index.get(request_id)
            if request_index is not None:
                self.runner.input_batch.block_table.add_row(
                    request.block_ids, request_index
                )
            self.active[request_id] = self.activate_pending(pending)
            logger.info(
                "KV compression worker commit acknowledged request_id=%s "
                "semantic_tokens=%d physical_tokens=%d destination_blocks=%d",
                request_id,
                pending.semantic_anchor,
                pending.physical_anchor,
                len(pending.block_ids),
            )
            self.pending.pop(request_id, None)

    # 宿主完成请求增删和行重排后，再构造按批次行排列的偏移数组。
    # 重置长度修正标记，保证同一步多次构建 metadata 时不会重复减偏移。
    def after_update_states(self, scheduler_output: Any) -> None:
        del scheduler_output
        if self.runner is not None:
            live = self.runner.requests.keys()
            for request_id in (self.pending.keys() | self.active.keys()) - live:
                self.discard_request(request_id)
        self._physical_lengths_applied = False
        self._sync_request_offset_rows()

    def compress_scheduled_requests(self, scheduler_output: Any) -> None:
        """注意力步骤完成后，执行调度器已授权的压缩事务。"""
        if self.runner is None or not self.layer_caches:
            raise RuntimeError("Ascend KV compression provider is not cache-bound")
        # 只有 scheduler 授权且已保留私有目的块的计划才可执行；
        # worker 不自行申请或释放块。
        # 初次压缩只在 prompt 恰好完成时触发；普通 decode 之后可按物理阈值再次触发。
        plans = getattr(scheduler_output, PLAN_ATTRIBUTE, {})
        for request_id, scheduled in scheduler_output.num_scheduled_tokens.items():
            plan = plans.get(request_id)
            if plan is None:
                continue
            # 同一计划可能因异步流水线再次可见；已写入但未确认时不能重复搬运。
            # 首次压缩等 prompt 完成，之后按 active 锚点推算当前物理窗口。
            if request_id in self.pending:
                continue
            request = self.runner.requests.get(request_id)
            if request is None:
                continue
            if (
                request_max_tokens(request)
                < self.runtime_spec.min_output_tokens_for_compression
            ):
                continue
            # worker request 此时保留本步前计数，
            # 而 scheduler.schedule 返回时已推进计数，因此这里显式加 scheduled，
            # 随后与计划锚点严格比对。
            semantic = int(request.num_computed_tokens) + int(scheduled)
            active = self.active.get(request_id)
            if active is None:
                physical = semantic
                if semantic != int(request.num_prompt_tokens):
                    continue
            else:
                physical = active.physical_tokens(semantic)
            if (
                semantic != plan.semantic_anchor
                or plan.physical_anchor != self.runtime_spec.max_physical_num_tokens
            ):
                raise RuntimeError(
                    f"request {request_id!r} semantic length disagrees with "
                    "the scheduler compression plan"
                )
            if physical < self.runtime_spec.compression_threshold_tokens:
                continue
            if self.attention_group_index >= len(request.block_ids):
                raise RuntimeError("request has no full-attention block table")
            source_ids = tuple(
                int(value) for value in request.block_ids[self.attention_group_index]
            )
            required_blocks = blocks_for_tokens(
                self.runtime_spec.max_physical_num_tokens,
                self.scheduler_block_size,
            )
            if len(source_ids) < required_blocks:
                raise RuntimeError(
                    f"request {request_id!r} has too few blocks for compression"
                )
            if source_ids != plan.source_block_ids:
                raise RuntimeError(
                    f"request {request_id!r} source block table disagrees "
                    "with the scheduler compression plan"
                )
            # 目的页数必须对应统一目标，且与源页无交集。
            # 即使源块被 APC 共享，也只读取它们，再把结果写入本请求独占的目的块。
            destination_ids = plan.destination_block_ids
            if len(destination_ids) != required_blocks or (
                set(destination_ids) & set(source_ids)
            ):
                raise RuntimeError(
                    f"request {request_id!r} requires private destination blocks"
                )
            # CPU tuple 保留 scheduler 的逻辑页编号用于核对授权；
            # 设备 tensor 展开成内核页编号供评分和搬运，两者表达不同层次，
            # 不能直接互换。
            source_device = expand_scheduler_block_ids(
                source_ids,
                self.cache_blocks_per_scheduler_block,
                self.runner.device,
            )
            destination_device = expand_scheduler_block_ids(
                destination_ids,
                self.cache_blocks_per_scheduler_block,
                self.runner.device,
            )
            transaction = self.build_compression_request(
                request_id, semantic, physical, source_device, destination_device, plan
            )
            result = self.compress_request(transaction, plan)
            if result.physical_num_tokens != self.runtime_spec.max_physical_num_tokens:
                raise RuntimeError("method returned a length different from its target")
            # 算法已提交目的块写入任务，但本地页表仍指向源块。
            # make_pending 允许下游附加待确认状态；激活要等下一次请求调度边界。
            self.pending[request_id] = self.make_pending(plan, result)

    def build_compression_request(
        self,
        request_id: str,
        semantic: int,
        physical: int,
        source_device: torch.Tensor,
        destination_device: torch.Tensor,
        plan: CompressionPlan,
    ) -> CompressionRequest:
        """供下游覆盖，从事务构造自定义算法输入。"""
        # 源计划只交给 provider 的扩展点；默认构造的算法输入不携带事务和分配器信息。
        return CompressionRequest(
            request_id=request_id,
            semantic_num_tokens=semantic,
            physical_num_tokens=physical,
            source_block_ids_device=source_device,
            destination_block_ids_device=destination_device,
        )

    def compress_request(
        self, request: CompressionRequest, plan: CompressionPlan
    ) -> CompressionResult:
        """算法执行扩展入口，默认算法不接收事务方案 plan。"""
        return self.method.compress(request)

    def make_pending(
        self, plan: CompressionPlan, result: CompressionResult
    ) -> PendingCompression:
        """供下游覆盖，将额外状态与已写入的目标块一起暂存。"""
        return PendingCompression(
            plan.semantic_anchor, result.physical_num_tokens, plan.destination_block_ids
        )

    # 下游需要额外状态时，应在这里与通用锚点同时激活。
    # 不要在 compress 返回时提前发布新状态，否则可能与 scheduler 的源块表不一致。
    def activate_pending(self, pending: PendingCompression) -> CompressionAnchor:
        """供下游覆盖，在同一确认边界提交额外状态。"""
        return CompressionAnchor(pending.semantic_anchor, pending.physical_anchor)

    def physical_positions_for_slot_mapping(
        self, positions: torch.Tensor
    ) -> torch.Tensor:
        """将写入映射到压缩后的物理槽位，同时保留 RoPE 语义位置。"""
        if not self._has_active_rows:
            return positions
        # 例如语义位置 8192、累计删除 6144，则新 KV 写到物理位置 2048。
        # 只计算传给 slot mapping 的位置，runner 的原始位置仍供 RoPE 使用。
        num_tokens = positions.numel()
        return shift_positions(
            positions,
            self.runner.req_indices.gpu[:num_tokens],
            self._request_offsets_device,
            self._physical_positions[:num_tokens],
        )

    def apply_physical_attention_lengths(self) -> None:
        """宿主构造采样决策所需结构后，再替换注意力元数据中的语义长度。"""
        if self._physical_lengths_applied or not self._has_active_rows:
            return
        num_reqs = self.runner.input_batch.num_reqs
        self._semantic_seq_lens_device[:num_reqs].copy_(self.runner.seq_lens[:num_reqs])
        self._semantic_seq_lens_cpu[:num_reqs].copy_(
            self.runner.optimistic_seq_lens_cpu[:num_reqs]
        )
        # 只改本步供 attention 消费的长度，不回写 Request 的语义计数。
        # 保存的原长度专供 GDN metadata 使用；下一步长度由宿主重新构建。
        self.runner.seq_lens[:num_reqs].sub_(self._request_offsets_device[:num_reqs])
        self.runner.optimistic_seq_lens_cpu[:num_reqs].sub_(
            self._request_offsets_cpu[:num_reqs]
        )
        self._physical_lengths_applied = True

    def semantic_gdn_metadata(self, metadata: Any) -> Any:
        """为 GDN 提供原始循环状态位置，不应用注意力压缩偏移。"""
        if not self._physical_lengths_applied:
            return metadata
        num_reqs = metadata.num_reqs
        return replace(
            metadata,
            seq_lens=self._semantic_seq_lens_device[:num_reqs],
            _seq_lens_cpu=self._semantic_seq_lens_cpu[:num_reqs],
            seq_lens_cpu_upper_bound=self._semantic_seq_lens_cpu[:num_reqs],
        )

    def _sync_request_offset_rows(self) -> None:
        num_reqs = self.runner.input_batch.num_reqs
        offsets = tuple(
            self.active[request_id].removed_tokens if request_id in self.active else 0
            for request_id in self.runner.input_batch.req_ids[:num_reqs]
        )
        # 比较的是偏移数值序列，足以判断设备数组是否需要更新。
        # 若两个请求偏移相同，即使互换行，也无需再次复制相同数值。
        if offsets == self._offset_row_snapshot:
            return
        # 批次缩短时也覆盖上一批使用过的尾部行，避免后续复用读到陈旧偏移。
        # 当前没有压缩请求时，数组全零，slot mapping 会直接复用宿主位置。
        rows = max(len(self._offset_row_snapshot), num_reqs)
        self._request_offsets_cpu[:rows].zero_()
        if offsets:
            self._request_offsets_cpu[:num_reqs].copy_(
                torch.tensor(offsets, dtype=torch.int64)
            )
        self._request_offsets_device[:rows].copy_(
            self._request_offsets_cpu[:rows], non_blocking=True
        )
        self._offset_row_snapshot = offsets
        self._has_active_rows = any(offsets)
