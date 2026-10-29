# SPDX-License-Identifier: Apache-2.0
"""对齐上游接口的插件适配器所使用的调度侧状态。"""

from __future__ import annotations

import inspect
import math
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from vllm.logger import logger

from ..core.config import PluginConfig
from ..host.contracts import (
    allowed_scheduler_block_sizes,
    find_full_attention_group,
    request_max_tokens,
    supports_mtp2,
)
from ..host.model_config import model_shape_from_config
from ..methods import create_backend
from ..methods.backend import CompressionBackend
from .coordinates import blocks_for_tokens
from .transaction import PLAN_ATTRIBUTE, CompressionAnchor, CompressionPlan

# scheduler 持有块对象与分配器；worker 只收到 transaction.py 的不可变计划。
# 下面挂到宿主对象上的属性均指向本插件维护的状态，宿主仓库本身无需增加字段。
_STATE_ATTRIBUTE = "_ascend_kvcompress_scheduler_state"
_OFFSETS_ATTRIBUTE = "_ascend_kvcompress_active_offsets"
_DEFERRED_RELEASES_ATTRIBUTE = "_ascend_kvcompress_deferred_releases"
_PENDING_ATTRIBUTE = "_ascend_kvcompress_pending_transactions"
_GROUP_LENGTH_STATE_ATTRIBUTE = "_ascend_kvcompress_group_length_state"
_PREFIX_ADMISSION_STATE_ATTRIBUTE = "_ascend_kvcompress_prefix_admission_state"
_PATCH_MARKER = "_ascend_kvcompress_manager_patch"
_PREFIX_LOOKUP_PATCH_MARKER = "_ascend_kvcompress_prefix_lookup_patch"
_PREFIX_CACHE_HIT_CAP: ContextVar[tuple[int, int] | None] = ContextVar(
    "ascend_kvcompress_prefix_cache_hit_cap", default=None
)
_SUPPORTED_SCHEDULER_TYPES = frozenset(
    {
        ("vllm.v1.core.sched.scheduler", "Scheduler"),
        ("vllm.v1.core.sched.async_scheduler", "AsyncScheduler"),
        (
            "vllm_ascend.patch.platform.patch_balance_schedule",
            "BalanceScheduler",
        ),
    }
)


@dataclass
# reserved_blocks 是尚未接入请求块表的私有目的块；ready 只在宿主接受对应输出后置位。
# 异步调度时 previous_eligible_step 保存冻结前的值，提交或取消时恢复。
class SchedulerPendingCompression:
    plan: CompressionPlan
    reserved_blocks: tuple[Any, ...]
    ready: bool = False
    previous_eligible_step: int | None = None


@dataclass
# 异步流水线里即使本事务已确认，其他在途 batch 仍可能引用旧块。
# 按后续输出屏障计数延后 free，remaining_outputs 不是时间或 decode token 数。
class SchedulerDeferredRelease:
    request_id: str
    blocks: tuple[Any, ...]
    remaining_outputs: int


class SchedulerCompressionState:
    """在模型输出确认边界同步 worker 的压缩事务。"""

    def __init__(self, scheduler: Any, backend: CompressionBackend) -> None:
        self.backend = backend
        self.scheduler = scheduler
        # 调度端只消费静态预算，不创建 method、不读取校准文件。
        # 两端都据此计算触发阈值和目的页数，确保 worker 能复核计划。
        spec = backend.runtime_spec
        self.threshold = spec.compression_threshold_tokens
        self.budget = spec.max_physical_num_tokens
        self.required_recompute_tokens = spec.required_recompute_tokens
        self.min_output_tokens = spec.min_output_tokens_for_compression
        # Scheduler.block_size 是所有混合缓存组对齐粒度的最小公倍数，
        # 可能大于全注意力管理器的逻辑页大小。
        self.scheduler_alignment_size = int(scheduler.block_size)
        self.scheduler_block_size = self.scheduler_alignment_size
        # pending/active 是 scheduler 的事务镜像，不与 worker 共享 Python 对象。
        # deferred_releases 单独保存已替换但仍受在途输出保护的源块，避免过早复用。
        self.pending: dict[str, SchedulerPendingCompression] = {}
        self.active: dict[str, CompressionAnchor] = {}
        self.deferred_releases: dict[str, list[SchedulerDeferredRelease]] = {}
        host_config = scheduler.vllm_config
        self.async_scheduling = bool(
            getattr(host_config.scheduler_config, "async_scheduling", False)
        )
        self.max_concurrent_batches = int(
            getattr(
                host_config,
                "max_concurrent_batches",
                2 if self.async_scheduling else 1,
            )
        )
        if self.max_concurrent_batches < 1:
            raise RuntimeError("max_concurrent_batches must be positive")
        self.attention_group_index = 0
        self.group_specific_lengths = False
        # 先检查宿主与缓存组布局，并确定真正的全注意力组下标和逻辑页大小。
        # 只有完成这一步，才知道应把长度换算 hook 装到 coordinator 还是单组 manager。
        self._validate_host()
        coordinator = scheduler.kv_cache_manager.coordinator
        # 两条长度适配路径：align 模式的 hybrid coordinator 可用时按 group 分发，
        # 否则只包装 full-attention single-type manager；两者都不缩短 GDN 语义长度。
        # 此分支服务于宿主缓存布局/ABI，而非 token 重要性算法。
        coordinator_hooks = self.group_specific_lengths and all(
            hasattr(type(coordinator), name)
            for name in (
                "get_num_blocks_to_allocate",
                "allocate_new_blocks",
                "remove_skipped_blocks",
                "cache_blocks",
            )
        )
        if coordinator_hooks:
            _install_coordinator_hooks(type(coordinator))
            setattr(coordinator, _GROUP_LENGTH_STATE_ATTRIBUTE, self)
        # manager 保存的是状态字典的引用；后续清理和分配 hook 会读到最新锚点。
        # 这些引用只在本进程内共享，worker 端通过计划与确认边界维护自己的镜像。
        setattr(scheduler.kv_cache_manager, _OFFSETS_ATTRIBUTE, self.active)
        setattr(
            scheduler.kv_cache_manager,
            _PREFIX_ADMISSION_STATE_ATTRIBUTE,
            self,
        )
        setattr(scheduler.kv_cache_manager, _PENDING_ATTRIBUTE, self.pending)
        setattr(
            scheduler.kv_cache_manager,
            _DEFERRED_RELEASES_ATTRIBUTE,
            self.deferred_releases,
        )
        self._install_attention_cache_hook()
        if not coordinator_hooks:
            self._install_attention_length_hooks()
        logger.info(
            "Ascend KV compression scheduler bound threshold_tokens=%d "
            "target_tokens=%d min_output_tokens=%d scheduler_alignment=%d "
            "attention_logical_block_size=%d required_recompute_tokens=%d",
            self.threshold,
            self.budget,
            self.min_output_tokens,
            self.scheduler_alignment_size,
            self.scheduler_block_size,
            self.required_recompute_tokens,
        )

    # APC（自动前缀缓存）命中会跳过 forward；这里限制命中长度，
    # 留出方法要求重算的后缀。
    # required_recompute_tokens 是调度契约；TriAttention 将它设成 recompute_window，
    # 命中上限只对满足压缩门槛的请求生效。
    def prefix_cache_hit_cap(self, request: Any) -> int | None:
        """限制 APC 命中长度，为符合条件的算法保留所需的后缀重算区间。"""
        prompt_tokens = int(request.num_prompt_tokens)
        if (
            prompt_tokens < self.threshold
            or request_max_tokens(request) < self.min_output_tokens
        ):
            return None
        return max(
            0,
            min(
                prompt_tokens - 1,
                prompt_tokens - self.required_recompute_tokens,
            ),
        )

    def _validate_host(self) -> None:
        scheduler = self.scheduler
        config = scheduler.vllm_config
        reasons: list[str] = []
        scheduler_type = type(scheduler)
        scheduler_identity = (scheduler_type.__module__, scheduler_type.__name__)
        if scheduler_identity not in _SUPPORTED_SCHEDULER_TYPES:
            reasons.append(
                "the upstream v1 Scheduler or the current Ascend "
                "BalanceScheduler wrapper is required"
            )
        if bool(getattr(scheduler, "_balance_enabled", False)):
            reasons.append("Ascend balance scheduling must be disabled")
        if config.speculative_config is not None and not supports_mtp2(config):
            reasons.append(
                "speculative decoding is unqualified; experimental Qwen3.5 "
                "MTP2 requires mamba_cache_mode='align' and explicit opt-in"
            )
        if config.kv_transfer_config is not None:
            reasons.append("KV transfer is unsupported")
        if bool(getattr(config.cache_config, "enable_prefix_caching", False)):
            cache_manager = scheduler.kv_cache_manager
            if not callable(getattr(cache_manager, "get_computed_blocks", None)):
                reasons.append("APC requires KVCacheManager.get_computed_blocks")
            if not callable(
                getattr(cache_manager.coordinator, "find_longest_cache_hit", None)
            ):
                reasons.append("APC requires KVCacheCoordinator.find_longest_cache_hit")
        parallel = config.parallel_config
        for name, attr in (
            ("pipeline parallel", "pipeline_parallel_size"),
            ("data parallel", "data_parallel_size"),
            ("prefill context parallel", "prefill_context_parallel_size"),
            ("decode context parallel", "decode_context_parallel_size"),
        ):
            value = int(getattr(parallel, attr, 1))
            if value != 1:
                reasons.append(f"{name} size must be one, got {value}")
        try:
            self.attention_group_index = find_full_attention_group(
                scheduler.kv_cache_config.kv_cache_groups
            )
        except RuntimeError as error:
            reasons.append(str(error))
        single_managers = scheduler.kv_cache_manager.coordinator.single_type_managers
        if self.attention_group_index >= len(single_managers):
            reasons.append("full-attention cache manager is unavailable")
        else:
            manager_block_sizes = tuple(
                int(getattr(manager, "block_size", 0)) for manager in single_managers
            )
            if any(block_size <= 0 for block_size in manager_block_sizes):
                reasons.append("KV cache managers must expose positive block sizes")
            else:
                # 混合模型统一调度粒度是各组逻辑页大小的最小公倍数。
                # 压缩预算只需对齐全注意力页；GDN 页可能更大，不能用 LCM
                # 直接计算目的页数。
                expected_alignment = math.lcm(*manager_block_sizes)
                if self.scheduler_alignment_size != expected_alignment:
                    reasons.append(
                        "scheduler alignment must equal the LCM of cache-group "
                        f"block sizes {manager_block_sizes}; got "
                        f"{self.scheduler_alignment_size}"
                    )
                self.scheduler_block_size = manager_block_sizes[
                    self.attention_group_index
                ]
                allowed_block_sizes = allowed_scheduler_block_sizes(config)
                if self.scheduler_block_size not in allowed_block_sizes:
                    reasons.append(
                        "full-attention logical block size must be one of "
                        f"{sorted(allowed_block_sizes)}, got "
                        f"{self.scheduler_block_size}"
                    )
                if self.budget % self.scheduler_block_size:
                    reasons.append(
                        "compression target must be divisible by full-attention "
                        f"logical block size {self.scheduler_block_size}"
                    )
        if len(scheduler.kv_cache_config.kv_cache_groups) > 1:
            mamba_cache_mode = getattr(config.cache_config, "mamba_cache_mode", "none")
            if mamba_cache_mode not in {"none", "align"}:
                reasons.append(
                    "hybrid compression requires mamba_cache_mode='none' or 'align'"
                )
            self.group_specific_lengths = mamba_cache_mode == "align"
        if reasons:
            raise RuntimeError(
                "Ascend KV compression is incompatible with this scheduler:\n- "
                + "\n- ".join(reasons)
            )

    # 同步模式可在下一次 schedule 开头提交已确认事务。
    # 异步模式仍有其他在途 batch，提交改由输出接受后的路径处理。
    def before_schedule(self) -> None:
        """仅提交模型输出已越过 worker 执行屏障的方案。"""
        scheduler = self.scheduler
        for request_id in tuple(scheduler.finished_req_ids):
            self._discard_pending(request_id)
            self.active.pop(request_id, None)
        if self.async_scheduling:
            return
        self._commit_pending(defer_release_outputs=0)

    def after_model_output(self) -> None:
        """在 worker 执行屏障之后提交已确认的异步事务。"""
        if not self.async_scheduling:
            return
        # 先推进已有延迟释放的计数，再加入本次提交产生的新延迟项。
        # 这样当前确认边界不会被重复算作刚加入项的一个后续输出边界。
        self._drain_deferred_releases()
        self._commit_pending(
            defer_release_outputs=max(0, self.max_concurrent_batches - 1)
        )

    # 事务顺序：after_schedule 保留目的块 → worker 写入 → after_output 标 ready
    # →此处切换 scheduler 块表并释放/延迟释放源块 → 请求再次调度时 worker 跟进切表。
    # 同步模式在下次 before_schedule 提交；异步模式在输出确认后提交并解除请求冻结。
    def _commit_pending(self, *, defer_release_outputs: int) -> None:
        scheduler = self.scheduler
        manager = scheduler.kv_cache_manager
        single_managers = manager.coordinator.single_type_managers
        cache_manager = single_managers[self.attention_group_index]
        for request_id, pending in tuple(self.pending.items()):
            # ready 来自宿主成功接受对应输出，单纯发出调度计划并不代表写入完成。
            # 请求已结束或原块表发生变化时，不能把目的块接到错误的请求状态上。
            if not pending.ready:
                continue
            if request_id not in scheduler.requests:
                self._discard_pending(request_id)
                continue
            blocks = cache_manager.req_to_blocks.get(request_id)
            if blocks is None:
                self._discard_pending(request_id)
                continue
            plan = pending.plan
            actual_source = tuple(block.block_id for block in blocks)
            if actual_source != plan.source_block_ids:
                self._discard_pending(request_id)
                raise RuntimeError(
                    f"request {request_id!r} block table changed before "
                    "compression commit"
                )
            source_blocks = tuple(blocks)
            # 保持请求块列表对象身份，只替换内容；只操作 full-attention manager。
            # 此时源块仍有自己的引用计数，释放经 block_pool 完成，
            # 不能原地覆盖共享 APC 源块。
            blocks[:] = pending.reserved_blocks
            released = tuple(reversed(source_blocks))
            # 切表后，其他已经排队的 batch 仍可能引用源块。
            # 延迟释放保护的是源块地址的可用期，不会推迟本请求改用目的块。
            if defer_release_outputs:
                self.deferred_releases.setdefault(request_id, []).append(
                    SchedulerDeferredRelease(
                        request_id=request_id,
                        blocks=released,
                        remaining_outputs=defer_release_outputs,
                    )
                )
            else:
                manager.block_pool.free_blocks(released)
            if hasattr(cache_manager, "num_cached_block"):
                # 私有压缩 KV 与请求原始语义 token 序列不再对应，
                # 因此不能使用该序列的前缀哈希。
                cache_manager.num_cached_block[request_id] = 0
            self.active[request_id] = CompressionAnchor(
                plan.semantic_anchor, plan.physical_anchor
            )
            request = scheduler.requests[request_id]
            if pending.previous_eligible_step is not None:
                request.next_decode_eligible_step = pending.previous_eligible_step
            logger.info(
                "KV compression scheduler commit request_id=%s "
                "semantic_tokens=%d physical_tokens=%d source_blocks=%d "
                "destination_blocks=%d released_blocks=%d",
                request_id,
                plan.semantic_anchor,
                plan.physical_anchor,
                len(plan.source_block_ids),
                len(plan.destination_block_ids),
                len(plan.source_block_ids),
            )
            self.pending.pop(request_id, None)

    # 包装器先调用宿主 update_from_output，正常返回后才到这里。
    # 计划比较确认的是同一轮事务；这里没有额外的压缩成功响应对象。
    def after_output(self, output: Any) -> None:
        """仅在 vLLM 接受 worker 输出后确认缓存写入。"""
        for request_id, plan in getattr(output, PLAN_ATTRIBUTE, {}).items():
            pending = self.pending.get(request_id)
            if pending is not None and pending.plan == plan:
                pending.ready = True
        self.after_model_output()

    def _drain_deferred_releases(self) -> None:
        pool = self.scheduler.kv_cache_manager.block_pool
        for request_id, releases in tuple(self.deferred_releases.items()):
            retained: list[SchedulerDeferredRelease] = []
            # 按接受输出的次数消耗保护期；减到零才交给 block_pool 处理引用计数。
            # 一个请求可有多轮待释放源块，所以字典值是列表而非单个块集合。
            for release in releases:
                release.remaining_outputs -= 1
                if release.remaining_outputs <= 0:
                    pool.free_blocks(release.blocks)
                else:
                    retained.append(release)
            if retained:
                self.deferred_releases[request_id] = retained
            else:
                self.deferred_releases.pop(request_id, None)

    # 取消只归还预留的目的块；当前请求源块仍交给宿主正常管理。
    # 已提交源块若还在 deferred_releases 中，
    # 由输出屏障或 manager.free 的清理路径回收。
    def _discard_pending(self, request_id: str) -> None:
        pending = self.pending.pop(request_id, None)
        if pending is None:
            return
        self.scheduler.kv_cache_manager.block_pool.free_blocks(
            reversed(pending.reserved_blocks)
        )
        request = self.scheduler.requests.get(request_id)
        if request is not None and pending.previous_eligible_step is not None:
            request.next_decode_eligible_step = pending.previous_eligible_step

    def _install_attention_cache_hook(self) -> None:
        """阻止将私有压缩 KV 登记到原始 token 哈希下。"""
        cache_manager = (
            self.scheduler.kv_cache_manager.coordinator.single_type_managers[
                self.attention_group_index
            ]
        )
        # 压缩后的私有 KV 不再对应完整语义前缀，不能挂到原 token hash 下。
        # 未压缩的请求继续走原缓存登记方法，保留正常 APC 复用。
        original = getattr(cache_manager, "cache_blocks", None)
        if original is None:
            return

        def cache_blocks(request: Any, *args: Any, **kwargs: Any) -> Any:
            if request.request_id in self.active:
                return None
            return original(request, *args, **kwargs)

        cache_manager.cache_blocks = cache_blocks

    def _install_attention_length_hooks(self) -> None:
        """仅换算全注意力长度，Mamba 状态仍使用语义长度。

        KVCacheManager 向所有缓存组传递相同的 token 数。压缩后，
        该数值仍适用于 Mamba/GDN，但全注意力需要使用压缩后的长度。
        因此保持 Request 的语义计数不变，只在管理器入口换算。
        """
        cache_manager = (
            self.scheduler.kv_cache_manager.coordinator.single_type_managers[
                self.attention_group_index
            ]
        )

        # 只对已有压缩锚点的请求减偏移，且只改全注意力 manager 收到的计数。
        # 宿主 Request 与 GDN/Mamba manager 仍使用原始语义计数。
        def physical(request_id: str, tokens: int) -> int:
            active = self.active.get(request_id)
            return active.physical_tokens(tokens) if active else tokens

        original_count = getattr(cache_manager, "get_num_blocks_to_allocate", None)
        if original_count is not None:

            def get_num_blocks_to_allocate(
                request_id: str,
                num_tokens: int,
                new_computed_blocks: Any,
                total_computed_tokens: int,
                num_local_computed_tokens: int,
                num_tokens_main_model: int,
                apply_admission_cap: bool = False,
                prefill_end: int | None = None,
            ) -> Any:
                options: dict[str, Any] = {"apply_admission_cap": apply_admission_cap}
                if prefill_end is not None:
                    options["prefill_end"] = physical(request_id, prefill_end)
                return original_count(
                    request_id,
                    physical(request_id, num_tokens),
                    new_computed_blocks,
                    physical(request_id, total_computed_tokens),
                    physical(request_id, num_local_computed_tokens),
                    physical(request_id, num_tokens_main_model),
                    **options,
                )

            cache_manager.get_num_blocks_to_allocate = get_num_blocks_to_allocate

        original_allocate = getattr(cache_manager, "allocate_new_blocks", None)
        if original_allocate is not None:

            def allocate_new_blocks(
                request_id: str, num_tokens: int, num_tokens_main_model: int
            ) -> Any:
                return original_allocate(
                    request_id,
                    physical(request_id, num_tokens),
                    physical(request_id, num_tokens_main_model),
                )

            cache_manager.allocate_new_blocks = allocate_new_blocks

        original_skip = getattr(cache_manager, "remove_skipped_blocks", None)
        if original_skip is not None:

            def remove_skipped_blocks(
                request_id: str,
                processed_computed_tokens: int,
                num_prompt_tokens: int | None = None,
            ) -> Any:
                return original_skip(
                    request_id,
                    physical(request_id, processed_computed_tokens),
                    num_prompt_tokens,
                )

            cache_manager.remove_skipped_blocks = remove_skipped_blocks

    # 计划在原 schedule 返回后附加，宿主本步 forward 仍读写源块。
    # 首次只选 final-prefill；压缩后按物理长度增长触发下一轮；
    # MTP2 禁止 decode 再压缩。
    # 空闲块不足时跳过本轮，不通过原地搬运绕过目的块所有权约束。
    def after_schedule(self, output: Any) -> None:
        """安排与 worker 实际执行内容一致的确定性事务。"""
        reset_ids = set(getattr(output, "preempted_req_ids", ()))
        reset_ids.update(output.finished_req_ids)
        for request_id in reset_ids:
            self._discard_pending(request_id)
            self.active.pop(request_id, None)

        coordinator = self.scheduler.kv_cache_manager.coordinator
        single_manager = coordinator.single_type_managers[self.attention_group_index]
        plans: dict[str, CompressionPlan] = {}
        # 有草稿 token 的步骤不能授权压缩，接受结果可能使其乐观尾部回滚。
        # 已支持的 MTP2 只允许最初 prompt 边界压缩，不在投机 decode 中反复触发。
        speculative_requests = getattr(output, "scheduled_spec_decode_tokens", {})
        for request_id in output.num_scheduled_tokens:
            if speculative_requests.get(request_id):
                continue
            if request_id in self.pending:
                continue
            request = self.scheduler.requests.get(request_id)
            if request is None:
                continue
            if request_max_tokens(request) < self.min_output_tokens:
                continue
            # 当前宿主会在 Scheduler.schedule 返回前，
            # 通过 _update_after_schedule 推进此值。worker 收到的是步前值，
            # 并自行加上本步调度的 token 数。
            semantic = int(request.num_computed_tokens)
            active = self.active.get(request_id)
            if active is None:
                physical = semantic
                if semantic != int(request.num_prompt_tokens):
                    continue
            else:
                physical = active.physical_tokens(semantic)
            if physical < self.threshold:
                continue
            if self.scheduler.vllm_config.speculative_config is not None and active:
                # 解码步中的乐观 MTP 长度可能在验收后回退；
                # 首次提示词压缩则不包含草稿 KV。
                continue
            blocks = single_manager.req_to_blocks.get(request_id, ())
            keep = blocks_for_tokens(self.budget, self.scheduler_block_size)
            if len(blocks) < keep:
                raise RuntimeError(
                    f"request {request_id!r} has too few scheduler blocks"
                )
            pool = self.scheduler.kv_cache_manager.block_pool
            # 压缩本身也需要一份目的空间，不能先释放正在读取的源块来凑容量。
            # 空闲不足就跳过，不以原地覆盖代替私有目的块事务。
            if pool.get_num_free_blocks() < keep:
                logger.warning(
                    "Skipping KV compression for request %s: %d private "
                    "destination blocks unavailable",
                    request_id,
                    keep,
                )
                continue
            destination = tuple(pool.get_new_blocks(keep))
            plan = CompressionPlan(
                semantic_anchor=semantic,
                physical_anchor=self.budget,
                source_block_ids=tuple(block.block_id for block in blocks),
                destination_block_ids=tuple(block.block_id for block in destination),
            )
            previous_step = None
            # 冻结单个请求，其他请求仍可调度；
            # 目的块写入获确认之前禁止该请求继续 decode。
            # 1 << 62 是不可达的调度步哨兵，提交和异常/结束清理路径必须恢复原值。
            if self.async_scheduling:
                previous_step = int(request.next_decode_eligible_step)
                request.next_decode_eligible_step = 1 << 62
            self.pending[request_id] = SchedulerPendingCompression(
                plan=plan,
                reserved_blocks=destination,
                previous_eligible_step=previous_step,
            )
            plans[request_id] = plan
        if plans:
            setattr(output, PLAN_ATTRIBUTE, plans)


def install_stateful_compression_hooks(
    scheduler_cls: type[Any],
    manager_cls: type[Any],
    plugin_config: PluginConfig,
) -> None:
    """安装可重复调用而不叠加包装的调度与分配钩子。"""
    _install_manager_hooks(manager_cls)
    marker = f"{_PATCH_MARKER}_scheduler"
    if scheduler_cls.__dict__.get(marker, False):
        return
    original_init = scheduler_cls.__init__
    original_schedule = scheduler_cls.schedule
    original_update_from_output = scheduler_cls.update_from_output

    # 先让宿主构造完整 scheduler，再用 Backend 工厂选取状态实现。
    # 下游可替换整个调度状态类，不必向通用调度器加入算法名分支。
    def initialize(scheduler: Any, *args: Any, **kwargs: Any) -> None:
        original_init(scheduler, *args, **kwargs)
        setattr(
            scheduler,
            _STATE_ATTRIBUTE,
            create_backend(
                plugin_config.method,
                plugin_config.method_config,
                scheduler.vllm_config,
                model_shape_from_config(scheduler.vllm_config.model_config),
            ).create_scheduler(scheduler),
        )

    def schedule(scheduler: Any, *args: Any, **kwargs: Any) -> Any:
        state = getattr(scheduler, _STATE_ATTRIBUTE)
        state.before_schedule()
        output = original_schedule(scheduler, *args, **kwargs)
        state.after_schedule(output)
        return output

    def update_from_output(
        scheduler: Any, scheduler_output: Any, *args: Any, **kwargs: Any
    ) -> Any:
        # 输出先由宿主接受，返回后才标记事务 ready；异常时不会执行插件确认。
        result = original_update_from_output(
            scheduler, scheduler_output, *args, **kwargs
        )
        getattr(scheduler, _STATE_ATTRIBUTE).after_output(scheduler_output)
        return result

    setattr(scheduler_cls, f"{marker}_original_init", original_init)
    setattr(scheduler_cls, f"{marker}_original_schedule", original_schedule)
    setattr(
        scheduler_cls,
        f"{marker}_original_update_from_output",
        original_update_from_output,
    )
    scheduler_cls.__init__ = initialize
    scheduler_cls.schedule = schedule
    scheduler_cls.update_from_output = update_from_output
    setattr(scheduler_cls, marker, True)


def _install_manager_hooks(manager_cls: type[Any]) -> None:
    if manager_cls.__dict__.get(_PATCH_MARKER, False):
        return
    # 除了宿主当前页表中的块，还要清理未接表的目的块和延迟释放的旧源块。
    # 这三类所有权分开处理，避免漏释放或把当前页表重复释放。
    original_free = manager_cls.free
    original_get_computed_blocks = getattr(manager_cls, "get_computed_blocks", None)

    # 宿主释放请求时一并清理 active、尚未提交的目的块和延迟释放的源块。
    # 这些块没有全部挂在宿主 req_to_blocks 中，所以不能只调用原 free。
    def free(manager: Any, request: Any) -> Any:
        getattr(manager, _OFFSETS_ATTRIBUTE, {}).pop(request.request_id, None)
        pending = getattr(manager, _PENDING_ATTRIBUTE, {}).pop(request.request_id, None)
        if pending is not None:
            manager.block_pool.free_blocks(reversed(pending.reserved_blocks))
            if pending.previous_eligible_step is not None:
                request.next_decode_eligible_step = pending.previous_eligible_step
        releases = getattr(manager, _DEFERRED_RELEASES_ATTRIBUTE, {}).pop(
            request.request_id, ()
        )
        for release in releases:
            manager.block_pool.free_blocks(release.blocks)
        return original_free(manager, request)

    def get_computed_blocks(
        manager: Any, request: Any, *args: Any, **kwargs: Any
    ) -> Any:
        state = getattr(manager, _PREFIX_ADMISSION_STATE_ATTRIBUTE, None)
        cap = None if state is None else state.prefix_cache_hit_cap(request)
        if cap is None:
            return original_get_computed_blocks(manager, request, *args, **kwargs)
        coordinator = manager.coordinator
        _install_prefix_lookup_hook(type(coordinator))
        # 临时上限绑定当前 coordinator 身份，退出 admission 调用后恢复。
        # 避免修改 coordinator 的永久配置或影响同进程其他实例/请求的 prefix lookup。
        token = _PREFIX_CACHE_HIT_CAP.set((id(coordinator), cap))
        try:
            return original_get_computed_blocks(manager, request, *args, **kwargs)
        finally:
            _PREFIX_CACHE_HIT_CAP.reset(token)

    setattr(manager_cls, f"{_PATCH_MARKER}_original_free", original_free)
    if original_get_computed_blocks is not None:
        setattr(
            manager_cls,
            f"{_PATCH_MARKER}_original_get_computed_blocks",
            original_get_computed_blocks,
        )
        manager_cls.get_computed_blocks = get_computed_blocks
    manager_cls.free = free
    setattr(manager_cls, _PATCH_MARKER, True)


def _install_prefix_lookup_hook(coordinator_cls: type[Any]) -> None:
    """仅限制当前请求接纳调用内部的前缀查找长度。"""
    if coordinator_cls.__dict__.get(_PREFIX_LOOKUP_PATCH_MARKER, False):
        return
    original = coordinator_cls.find_longest_cache_hit

    def find_longest_cache_hit(
        coordinator: Any,
        block_hashes: Any,
        max_cache_hit_length: int,
    ) -> Any:
        scoped_cap = _PREFIX_CACHE_HIT_CAP.get()
        if scoped_cap is not None and scoped_cap[0] == id(coordinator):
            max_cache_hit_length = min(max_cache_hit_length, scoped_cap[1])
        return original(coordinator, block_hashes, max_cache_hit_length)

    setattr(coordinator_cls, f"{_PREFIX_LOOKUP_PATCH_MARKER}_original", original)
    coordinator_cls.find_longest_cache_hit = find_longest_cache_hit
    setattr(coordinator_cls, _PREFIX_LOOKUP_PATCH_MARKER, True)


def _install_coordinator_hooks(coordinator_cls: type[Any]) -> None:
    """仅换算已压缩全注意力缓存组的 token 长度。"""
    marker = f"{_PATCH_MARKER}_coordinator"
    if coordinator_cls.__dict__.get(marker, False):
        return
    original_count = coordinator_cls.get_num_blocks_to_allocate
    original_allocate = coordinator_cls.allocate_new_blocks
    original_remove = coordinator_cls.remove_skipped_blocks

    # coordinator 与各 manager 的新旧签名可能不同（尤其 Mamba）。
    # 先按真实宿主签名绑定参数，再只把支持的可选参数转发给各组；压缩组单独换算长度。
    count_signature = inspect.signature(original_count)

    def get_num_blocks_to_allocate(coordinator: Any, *args: Any, **kwargs: Any) -> int:
        # 先按实际宿主签名恢复参数语义，兼容位置参数和关键字参数的调用方式。
        # 随后逐组分发，避免把新参数直接传给仍使用旧签名的 Mamba manager。
        bound = count_signature.bind(coordinator, *args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        request_id = values["request_id"]
        num_tokens = values["num_tokens"]
        new_computed_blocks = values["new_computed_blocks"]
        total_computed_tokens = values["total_computed_tokens"]
        num_local_computed_tokens = values.get(
            "num_local_computed_tokens", total_computed_tokens
        )
        num_tokens_main_model = values["num_tokens_main_model"]
        apply_admission_cap = values.get("apply_admission_cap", False)
        prefill_end = values.get("prefill_end")
        state = getattr(coordinator, _GROUP_LENGTH_STATE_ATTRIBUTE, None)
        active = None if state is None else state.active.get(request_id)
        if state is None:
            return original_count(coordinator, *args, **kwargs)
        if active is None and all(
            "prefill_end"
            in inspect.signature(manager.get_num_blocks_to_allocate).parameters
            for manager in coordinator.single_type_managers
        ):
            return original_count(coordinator, *args, **kwargs)
        # 当前 Ascend Mamba 管理器仍使用旧版函数签名。
        # 未压缩请求也走同一个缓存组循环，
        # 只向支持 prefill_end 的管理器传入该参数。
        if active is None and getattr(coordinator, "retention_interval", 0):
            prefill_end = 0
        result = 0
        # 每组保留自己的计数空间：全注意力减压缩偏移，循环状态组不减。
        # 所需新块数再相加，由统一池判断是否能完成整个请求的分配。
        for index, manager in enumerate(coordinator.single_type_managers):
            if active is not None and index == state.attention_group_index:
                group_tokens = active.physical_tokens(num_tokens)
                group_computed = active.physical_tokens(total_computed_tokens)
                group_local = active.physical_tokens(num_local_computed_tokens)
                group_main = active.physical_tokens(num_tokens_main_model)
            else:
                group_tokens = num_tokens
                group_computed = total_computed_tokens
                group_local = num_local_computed_tokens
                group_main = num_tokens_main_model
            manager_values = {
                "request_id": request_id,
                "num_tokens": group_tokens,
                "new_computed_blocks": new_computed_blocks[index],
                "total_computed_tokens": group_computed,
                "num_tokens_main_model": group_main,
                "apply_admission_cap": apply_admission_cap,
            }
            manager_signature = inspect.signature(manager.get_num_blocks_to_allocate)
            if "num_local_computed_tokens" in manager_signature.parameters:
                manager_values["num_local_computed_tokens"] = group_local
            if (
                prefill_end is not None
                and "prefill_end" in manager_signature.parameters
            ):
                manager_values["prefill_end"] = (
                    active.physical_tokens(prefill_end)
                    if active is not None and index == state.attention_group_index
                    else prefill_end
                )
            result += manager.get_num_blocks_to_allocate(**manager_values)
        return result

    def allocate_new_blocks(
        coordinator: Any,
        request_id: str,
        num_tokens: int,
        num_tokens_main_model: int,
        num_encoder_tokens: int = 0,
    ) -> tuple[list[Any], ...]:
        state = getattr(coordinator, _GROUP_LENGTH_STATE_ATTRIBUTE, None)
        active = None if state is None else state.active.get(request_id)
        if active is None:
            return original_allocate(
                coordinator,
                request_id,
                num_tokens,
                num_tokens_main_model,
                num_encoder_tokens,
            )
        allocated = []
        for index, manager in enumerate(coordinator.single_type_managers):
            if index == state.attention_group_index:
                group_tokens = active.physical_tokens(num_tokens)
                group_main = active.physical_tokens(num_tokens_main_model)
            else:
                group_tokens = num_tokens
                group_main = num_tokens_main_model
            allocated.append(
                manager.allocate_new_blocks(request_id, group_tokens, group_main)
            )
        return tuple(allocated)

    def remove_skipped_blocks(
        coordinator: Any,
        request_id: str,
        total_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        state = getattr(coordinator, _GROUP_LENGTH_STATE_ATTRIBUTE, None)
        active = None if state is None else state.active.get(request_id)
        if active is None:
            return original_remove(
                coordinator,
                request_id,
                total_computed_tokens,
                num_prompt_tokens,
            )
        for index, manager in enumerate(coordinator.single_type_managers):
            group_tokens = (
                active.physical_tokens(total_computed_tokens)
                if index == state.attention_group_index
                else total_computed_tokens
            )
            manager.remove_skipped_blocks(request_id, group_tokens, num_prompt_tokens)

    setattr(coordinator_cls, f"{marker}_original_count", original_count)
    setattr(coordinator_cls, f"{marker}_original_allocate", original_allocate)
    setattr(coordinator_cls, f"{marker}_original_remove", original_remove)
    coordinator_cls.get_num_blocks_to_allocate = get_num_blocks_to_allocate
    coordinator_cls.allocate_new_blocks = allocate_new_blocks
    coordinator_cls.remove_skipped_blocks = remove_skipped_blocks
    setattr(coordinator_cls, marker, True)
