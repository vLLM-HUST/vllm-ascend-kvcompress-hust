# SPDX-License-Identifier: Apache-2.0
"""面向 Ascend KV 缓存的 TriAttention token 选择方法。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ...core.constants import KERNEL_CACHE_BLOCK_SIZE
from ..base import (
    CompressionRequest,
    CompressionResult,
    KVCompressionMethod,
    LayerCache,
    ModelShape,
)
from .cache import _gather_paged_range, materialize_token_slots, token_slots
from .config import TriAttentionConfig
from .kernels import (
    aggregate_normalized_scores,
    prepare_mean_phase_coefficients,
    score_paged_keys_mean_precomputed,
)
from .scoring import (
    _score_post_rope_keys,
    build_geometric_offsets,
    normalize_head_scores,
)
from .selection import _select_keep_indices
from .stats import CalibrationStats, DeviceLayerCalibrationStats


def _queries_per_kv_head(model_shape: ModelShape) -> int:
    # 宿主模型配置转换时已验证头数为正且满足 GQA 整除关系。
    return model_shape.num_attention_heads // model_shape.num_kv_heads


@dataclass(frozen=True)
class TriAttentionLayerCache:
    """
    评分层绑定：原 cache view + 当前 TP rank 的离线 Q 统计 + 与请求无关的预计算系数。
    每轮只需加入 semantic round_start 的相位；无需保存历史 Q 或每请求评分矩阵。
    """

    name: str
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    stats: DeviceLayerCalibrationStats
    frequency_scale: torch.Tensor | None = None
    extra_coefficient: torch.Tensor | None = None
    offset_cos_mean: torch.Tensor | None = None
    offset_sin_mean: torch.Tensor | None = None


class TriAttentionMethod(KVCompressionMethod):
    def __init__(
        self,
        config: TriAttentionConfig,
        vllm_config: Any,
        model_shape: ModelShape,
    ) -> None:
        self.config = config
        self.vllm_config = vllm_config
        self.model_shape = model_shape
        # 从文件加载的 CPU 校准统计，绑定时再生成本地设备数据。
        self.calibration = CalibrationStats.load(self.config.stats_path)

        # 评分 cache 仅含已校准的 target full-attention 层；
        # materialization_caches 还可含实验 MTP2 辅助层，该层不评分，
        # 但使用 target 选出的 token 索引一并搬运。
        # 下面 workspace 按 runner 复用，不按请求保存；
        # 同一 method 的调用应遵循 runner 顺序。

        # 已绑定的目标全注意力层及其设备统计，按步长抽样评分。
        self.layer_caches: tuple[TriAttentionLayerCache, ...] = ()
        # 需要搬运 KV 的全部缓存层，可包含不评分的辅助 MTP 层。
        self.materialization_caches: tuple[LayerCache, ...] = ()
        # 内核缓存块的 token 数，绑定后从实际缓存形状确认。
        self.cache_block_size = KERNEL_CACHE_BLOCK_SIZE
        # 未来 query 相对本轮起点的几何采样偏移，一维设备张量。
        self.offsets: torch.Tensor | None = None
        # 所选 K 的搬运缓冲区：[预算, 本地 KV 头数, head_dim]。
        self.k_workspace: torch.Tensor | None = None
        # 所选 V 的搬运缓冲区，形状与 K 缓冲区相同。
        self.v_workspace: torch.Tensor | None = None
        # 单层原始分数缓冲区：[本地 KV 头数, GQA 组宽, 最大上下文]。
        self.score_workspace: torch.Tensor | None = None
        # 跨头、跨层聚合后的逐 token 分数缓冲区。
        self.aggregate_workspace: torch.Tensor | None = None
        # TP 逐层分数缓冲区，展平存储，使用时恢复为 [评分层, token]。
        self.parallel_score_workspace: torch.Tensor | None = None
        # 目标密集位置索引 0 到 kv_budget-1，用于计算写入槽位。
        self.dense_indices: torch.Tensor | None = None
        # 抽样评分层的旋转频率，形状为 [评分层数, 旋转频率数]。
        self.scoring_omega: torch.Tensor | None = None
        # 各评分层未来采样偏移的余弦均值，与旋转频率一一对应。
        self.scoring_offset_cos: torch.Tensor | None = None
        # 各评分层未来采样偏移的正弦均值，与旋转频率一一对应。
        self.scoring_offset_sin: torch.Tensor | None = None
        # 加入本轮语义起点后的 mean 评分余弦系数缓冲区。
        self.phase_cos_workspace: torch.Tensor | None = None
        # 加入本轮语义起点后的 mean 评分正弦系数缓冲区。
        self.phase_sin_workspace: torch.Tensor | None = None

    def compatibility_reasons(self, worker: Any) -> tuple[str, ...]:
        del worker
        return self.calibration.validate(self.model_shape)

    def bind_model_runner(
        self,
        runner: Any,
        layer_caches: tuple[LayerCache, ...],
    ) -> None:
        parallel = self.vllm_config.parallel_config
        self.tensor_parallel_size = int(parallel.tensor_parallel_size)

        # 当前 TP 组内编号；非 TP 场景为 0。
        tensor_parallel_rank = 0
        if self.tensor_parallel_size > 1:
            from vllm.distributed import get_tp_group

            tensor_parallel_rank = int(get_tp_group().rank_in_group)

        # 离线统计先按 GQA 重排为 [local_kv_heads,queries_per_kv,frequency]，
        # 再绑定真实层号；TP 各 rank 只评分本地头，但最终必须一致选择 token。
        # 按原始层号索引、已完成 GQA 重排和 TP 分片的设备统计。
        device_stats = self.calibration.to_device(
            self.model_shape,
            runner.device,
            tensor_parallel_rank=tensor_parallel_rank,
            tensor_parallel_size=self.tensor_parallel_size,
        )
        self.offsets = build_geometric_offsets(
            int(self.vllm_config.model_config.max_model_len), runner.device
        )

        # 辅助 MTP attention 不属于 target 的校准层集合，
        # 仅允许精确的 Qwen3.5 MTP2 布局。
        # 这项模型特判与算法评分无关，但当前仍在 method 内决定搬运层集合。
        # 辅助 MTP 注意力缓存，只搬运、不参与目标模型评分。
        mtp_layers = tuple(
            layer for layer in layer_caches if _is_mtp_cache_layer(layer.name)
        )
        # 目标模型的全注意力缓存，需要匹配校准层。
        target_layers = tuple(
            layer for layer in layer_caches if not _is_mtp_cache_layer(layer.name)
        )
        if mtp_layers and (
            len(mtp_layers) != 1
            or mtp_layers[0].name != "mtp.layers.0.self_attn.attn"
            or getattr(self.vllm_config.speculative_config, "method", None) != "mtp"
            or getattr(
                self.vllm_config.speculative_config, "num_speculative_tokens", None
            )
            != 2
        ):
            raise RuntimeError("uncalibrated MTP cache requires Qwen3.5 MTP2")
        # 在索引设备统计前核对完整层集合；统计层集合已由兼容性检查验证。
        expected_indices = set(self.model_shape.full_attention_layer_indices)
        allocated_indices = {layer.layer_index for layer in target_layers}
        if allocated_indices != expected_indices:
            raise RuntimeError(
                "allocated target full-attention layers do not match calibration: "
                f"allocated={sorted(allocated_indices)} "
                f"expected={sorted(expected_indices)}"
            )

        # 逐层校验并完成系数预计算后的评分缓存绑定列表。
        bound: list[TriAttentionLayerCache] = []
        # 已绑定的原始层号，用于检查重复和缺失。
        seen_layer_indices: set[int] = set()
        for layer in target_layers:
            if layer.layer_index in seen_layer_indices:
                raise RuntimeError(
                    "multiple cache layers resolve to model layer index "
                    f"{layer.layer_index}"
                )
            seen_layer_indices.add(layer.layer_index)
            stats = device_stats[layer.layer_index]
            # cache K 已经经过 RoPE 与 key-side scaling，
            # 评分只补 query-side 的 sqrt(scale²)。
            # q_abs_mean - |E[Q]| 是幅度补偿；offset 相位均值可在绑定时预计算。
            frequency_scale = torch.sqrt(stats.freq_scale_sq)
            q_mean_abs = torch.sqrt(
                stats.q_mean_real.square() + stats.q_mean_imag.square() + 1e-8
            )
            # 未来采样偏移与旋转频率的乘积：[采样位置数, 频率数]。
            offset_phases = self.offsets.unsqueeze(1) * stats.omega.unsqueeze(0)
            bound.append(
                TriAttentionLayerCache(
                    name=layer.name,
                    k_cache=layer.k_cache,
                    v_cache=layer.v_cache,
                    stats=stats,
                    frequency_scale=frequency_scale,
                    extra_coefficient=stats.q_abs_mean - q_mean_abs,
                    offset_cos_mean=torch.cos(offset_phases).mean(dim=0),
                    offset_sin_mean=torch.sin(offset_phases).mean(dim=0),
                )
            )
        self.layer_caches = tuple(bound)
        self.materialization_caches = layer_caches
        # 按配置步长抽取的评分层，搬运仍覆盖全部缓存层。
        scoring_layers = self.layer_caches[:: self.config.score_layer_stride]
        # 以下系数沿评分层堆叠，形状为 [评分层数,旋转频率数]，便于一次准备本轮相位。
        self.scoring_omega = torch.stack(
            [layer.stats.omega for layer in scoring_layers]
        ).contiguous()
        self.scoring_offset_cos = torch.stack(
            [layer.offset_cos_mean for layer in scoring_layers]
        ).contiguous()
        self.scoring_offset_sin = torch.stack(
            [layer.offset_sin_mean for layer in scoring_layers]
        ).contiguous()
        self.phase_cos_workspace = torch.empty_like(self.scoring_omega)
        self.phase_sin_workspace = torch.empty_like(self.scoring_omega)
        # 以首个评分缓存为参照，确定块大小、头数、维度和数据类型。
        cache = self.layer_caches[0].k_cache
        # provider 已逐层验证缓存布局、内核块大小及 TP 本地头数。
        self.cache_block_size = int(cache.shape[1])

        # 搬运按层串行复用同一对 [budget,本地 KV 头,head_dim] 临时张量。
        # 因此显存开销不按搬运层数倍增，辅助 MTP cache 也复用这组空间。

        # 单层 K/V 搬运缓冲区形状，各层顺序复用。
        workspace_shape = (
            self.config.kv_budget,
            cache.shape[2],
            cache.shape[3],
        )
        self.k_workspace = torch.empty(
            workspace_shape, dtype=cache.dtype, device=runner.device
        )
        self.v_workspace = torch.empty_like(self.k_workspace)

        # GQA 分组宽度是模型全局比例，不随张量并行分片改变。
        # 若用全局 query 头数除以本地缓存头数，
        # 则在 TP=2 时会错误地将工作区宽度翻倍。

        # 每个 KV 头对应的 Query 头数，即 GQA 组宽。
        queries_per_kv = _queries_per_kv_head(self.model_shape)
        # 评分原始矩阵 [本地 KV 头,GQA Query 头组,最大上下文] 使用 FP32。
        # aggregate 保存每 token 的最终分数；TP 额外保留 [抽样层,token]，
        # 合成一次通信。
        # K/V 搬运 workspace 只需 budget 容量，所有层顺序复用，避免逐层常驻副本。
        self.score_workspace = torch.empty(
            (
                cache.shape[2],
                queries_per_kv,
                int(self.vllm_config.model_config.max_model_len),
            ),
            dtype=torch.float32,
            device=runner.device,
        )
        self.aggregate_workspace = torch.empty(
            int(self.vllm_config.model_config.max_model_len),
            dtype=torch.float32,
            device=runner.device,
        )
        if self.tensor_parallel_size > 1:
            self.parallel_score_workspace = torch.empty(
                len(scoring_layers) * int(self.vllm_config.model_config.max_model_len),
                dtype=torch.float32,
                device=runner.device,
            )
        self.dense_indices = torch.arange(
            self.config.kv_budget, device=runner.device, dtype=torch.long
        )

    def compress(self, request: CompressionRequest) -> CompressionResult:
        # dense_indices 在绑定末尾创建；到达此状态时其余工作区已准备完毕。
        if self.dense_indices is None:
            raise RuntimeError("TriAttention method is not bound to KV cache")

        # 一次选取供所有层使用，因此本方法是跨层统一的 token 保留策略。
        # physical_num_tokens 决定要评分的 cache 范围；
        # semantic_num_tokens 决定未来 Q 的绝对相位。
        # 压缩后第 j 个位置存的是 keep_indices[j] 的原 K/V，不重新施加 RoPE。

        # 本轮保留的源逻辑 token 索引，升序排列，所有搬运层共享。
        keep_indices = self._select_keep_indices(
            request.source_block_ids_device,
            request.physical_num_tokens,
            request.semantic_num_tokens,
        )
        # 实际内核页的 token 容量，用于逻辑位置到槽位的换算。
        cache_block_size = getattr(self, "cache_block_size", KERNEL_CACHE_BLOCK_SIZE)
        # 选中 token 在源缓存中的展平物理槽位。
        source_slots = token_slots(
            request.source_block_ids_device, keep_indices, cache_block_size
        )
        # 密集目标位置在私有目标缓存中的展平物理槽位。
        destination_slots = token_slots(
            request.destination_block_ids_device,
            self.dense_indices,
            cache_block_size,
        )
        # 先从源块 gather 到复用 workspace，再写密集目的 slots。
        # 只写 cache 内容；切换块表、确认事务和回收源块不在算法层执行。
        for layer in self.materialization_caches:
            materialize_token_slots(
                layer.k_cache,
                layer.v_cache,
                source_slots,
                destination_slots,
                self.k_workspace,
                self.v_workspace,
            )
        return CompressionResult(physical_num_tokens=self.config.kv_budget)

    def _select_keep_indices(
        self,
        source_block_ids: torch.Tensor,
        physical_num_tokens: int,
        round_start: int,
    ) -> torch.Tensor:
        """对当前物理 KV 序列评分，返回所有缓存层共享的保留索引。

        按 score_layer_stride 抽取评分层，使用校准统计估计未来 query 对
        各 token 的重要性；每头归一化后跨头取最大值，再按配置跨层聚合。
        TP 场景先同步各层分数，最后应用前缀/近期保护及 global 或 v3 选择策略。

        参数：
            source_block_ids：设备端源页表，元素为内核缓存块的物理编号。
            physical_num_tokens：源缓存中有效的 token 数，也是本轮评分范围。
            round_start：未来 query 采样的语义起点，通常为本轮前向后的语义长度；
                即使缓存已压缩，也不能替换为物理长度。

        返回：
            长度为 kv_budget 的升序设备索引张量，表示源缓存逻辑序列中的
            token 位置，而非原始语义位置或展平物理槽位。

        本方法复用并覆盖评分工作区，不搬运 KV，也不修改请求块表。
        """
        # 本轮有效 token 范围内的最终分数视图，与持久缓冲区共享存储。
        aggregated_scores = self.aggregate_workspace[:physical_num_tokens]
        # 分页 key 评分和读取使用的内核块大小。
        cache_block_size = getattr(self, "cache_block_size", KERNEL_CACHE_BLOCK_SIZE)
        # 参与本轮分数同步的 TP rank 数。
        tensor_parallel_size = getattr(self, "tensor_parallel_size", 1)
        # 对每个模型层评分会在调度热路径中串行执行数十次小型归约。
        # 因此均匀采样已校准层进行评分，
        # 但仍为每个缓存层写入所选 KV token。
        # 本轮实际评分的层子集，保留绑定顺序。
        scoring_layers = self.layer_caches[:: self.config.score_layer_stride]
        # 实际参与评分的层数，也是层均值聚合的除数。
        layer_count = len(scoring_layers)
        # TP 先保存每层的本地头聚合结果，组成 [评分层数,当前物理长度]。
        # 这样可将逐层通信合成一次，再沿层维做最终聚合。
        # TP 逐层分数视图；非 TP 时为 None，直接聚合到最终向量。
        parallel_scores = None
        if tensor_parallel_size > 1:
            parallel_scores = self.parallel_score_workspace[
                : layer_count * physical_num_tokens
            ].view(layer_count, physical_num_tokens)
        # 批量准备 mean 相位系数所需的频率、偏移均值和输出缓冲区。
        phase_inputs = (
            getattr(self, "scoring_omega", None),
            getattr(self, "scoring_offset_cos", None),
            getattr(self, "scoring_offset_sin", None),
            getattr(self, "phase_cos_workspace", None),
            getattr(self, "phase_sin_workspace", None),
        )
        # mean 模式可用三角加法公式合并未来 offset 均值与 round_start 相位。
        # NPU 上批量预计算所有评分层的系数；CPU 返回 False 后走可对照的 PyTorch 公式。
        # 是否已成功生成本轮相位系数，决定能否使用分页评分内核。
        phases_prepared = all(value is not None for value in phase_inputs) and (
            prepare_mean_phase_coefficients(
                self.scoring_omega,
                self.scoring_offset_cos,
                self.scoring_offset_sin,
                round_start,
                self.phase_cos_workspace,
                self.phase_sin_workspace,
            )
        )
        # 每层先得到所有本地 Query 头的原始分数，再沿完整 token 维标准化。
        # 只抽样评分层；最终得到的一组 keep_indices 仍用于所有搬运层。
        for layer_index, layer in enumerate(scoring_layers):
            # 当前层复用的原始评分缓冲区。
            score_workspace = self.score_workspace
            # 当前层分页评分内核需要的缩放、补偿和偏移相位系数。
            kernel_inputs = (
                layer.frequency_scale,
                layer.extra_coefficient,
                layer.offset_cos_mean,
                layer.offset_sin_mean,
            )
            if (
                self.config.score_aggregation == "mean"
                and phases_prepared
                and all(value is not None for value in kernel_inputs)
                and (
                    getattr(layer.stats, "rotary_dim", None) is None
                    or getattr(layer.stats, "rotary_dim", None)
                    == layer.k_cache.shape[-1]
                    or getattr(layer.stats, "q_pass_mean", None) is not None
                )
            ):
                # 截取当前物理长度的原始分数视图：[本地 KV 头, GQA 组宽, token]。
                raw_scores = score_workspace[:, :, :physical_num_tokens]
                # 本层是否已由分页评分内核写出原始分数。
                used_kernel = score_paged_keys_mean_precomputed(
                    layer.k_cache,
                    source_block_ids,
                    layer.stats.q_mean_real,
                    layer.stats.q_mean_imag,
                    layer.frequency_scale,
                    layer.extra_coefficient,
                    self.phase_cos_workspace[layer_index],
                    self.phase_sin_workspace[layer_index],
                    physical_num_tokens,
                    raw_scores,
                    cache_block_size,
                    layer.stats.rope_style,
                    q_pass_mean=getattr(layer.stats, "q_pass_mean", None),
                    rotary_dim=getattr(layer.stats, "rotary_dim", None),
                )
            else:
                used_kernel = False
            # 两条路径均输出 [本地 KV 头,GQA Query 组,物理 token] 的原始分数。
            # 标准化沿完整 token 维进行，不能在 score_chunk_size 的小块内各自标准化。
            if used_kernel:
                # 各 Query 头沿完整 token 维的分数方差和均值。
                head_variance, head_mean = torch.var_mean(
                    raw_scores, dim=-1, correction=0
                )
                # 当前层的聚合写入位置：TP 独立行或非 TP 最终分数向量。
                layer_output = (
                    parallel_scores[layer_index]
                    if parallel_scores is not None
                    else aggregated_scores
                )
                if not aggregate_normalized_scores(
                    raw_scores,
                    head_mean,
                    head_variance,
                    layer_output,
                    physical_num_tokens,
                    layer_aggregation=self.config.layer_aggregation,
                    first_layer=parallel_scores is not None or layer_index == 0,
                ):
                    self._aggregate_torch(
                        raw_scores,
                        head_mean,
                        head_variance,
                        layer_output,
                        layer_index,
                        defer_tp_reduce=parallel_scores is not None,
                    )
                continue
            raw_scores = score_workspace[:, :, :physical_num_tokens]
            # 通用回退先按页 gather，再用 scoring.py 计算；
            # 支持 mean/max 与 partial RoPE。
            # chunk 只控制中间张量峰值，不改变最终每头统计覆盖的上下文范围。
            for start in range(0, physical_num_tokens, self.config.score_chunk_size):
                # 当前评分分块的实际 token 数，最后一块可能较短。
                count = min(self.config.score_chunk_size, physical_num_tokens - start)
                # 从分页缓存读取的当前分块 K，保留原有 RoPE 编码。
                keys = _gather_paged_range(
                    layer.k_cache,
                    source_block_ids,
                    start=start,
                    count=count,
                    block_size=cache_block_size,
                )
                # PyTorch 回退路径生成的当前分块原始分数。
                chunk_scores = _score_post_rope_keys(
                    keys,
                    layer.stats,
                    round_start=round_start,
                    offsets=self.offsets,
                    aggregation=self.config.score_aggregation,
                )
                raw_scores[:, :, start : start + count].copy_(chunk_scores)
            head_variance, head_mean = torch.var_mean(raw_scores, dim=-1, correction=0)
            layer_output = (
                parallel_scores[layer_index]
                if parallel_scores is not None
                else aggregated_scores
            )
            if not aggregate_normalized_scores(
                raw_scores,
                head_mean,
                head_variance,
                layer_output,
                physical_num_tokens,
                layer_aggregation=self.config.layer_aggregation,
                first_layer=parallel_scores is not None or layer_index == 0,
            ):
                self._aggregate_torch(
                    raw_scores,
                    head_mean,
                    head_variance,
                    layer_output,
                    layer_index,
                    defer_tp_reduce=parallel_scores is not None,
                )

        if parallel_scores is not None:
            import torch.distributed as dist
            from vllm.distributed import get_tp_group

            # 各层、各 token 的 MAX 归约彼此独立，
            # 因此一次连续存储上的集合通信等价于每个评分层各通信一次。
            dist.all_reduce(
                parallel_scores,
                op=dist.ReduceOp.MAX,
                group=get_tp_group().device_group,
            )
            # 跨层归约算子；mean 先求和，最后再除以层数。
            reduction = (
                torch.sum if self.config.layer_aggregation == "mean" else torch.amax
            )
            reduction(parallel_scores, dim=0, out=aggregated_scores)

        # TP 先对每层每 token 做跨 rank 的 MAX，再对层求和/最大值；
        # 各 rank 得到相同全局分数，从而选择相同的块内 token 位置。
        # 层 mean 最后统一除层数，避免把跨头 max 误写成跨 rank 平均。
        if self.config.layer_aggregation == "mean":
            aggregated_scores.div_(float(layer_count))
        # 将 NaN 和无穷值规范化后再做位置选取，避免无效分数污染 top-k。
        # 保护窗口的强制保留由 selection.py 单独处理，不靠这里提高保护区分数。
        aggregated_scores.nan_to_num_(
            nan=float("-inf"), posinf=1e30, neginf=float("-inf")
        )
        return _select_keep_indices(
            aggregated_scores,
            budget=self.config.kv_budget,
            protected_prefix=self.config.protected_prefix_window,
            protected_recent=self.config.protected_recent_window,
            segments=self.config.position_segments,
            policy=self.config.position_policy,
        )

    # 这是融合聚合 kernel 的 PyTorch 对照路径：标准化 → 头间 max → 层间 sum/max。
    # defer_tp_reduce 表示先写当前层独立向量，跨 rank 归约留给外层统一执行。
    def _aggregate_torch(
        self,
        raw_scores: torch.Tensor,
        head_mean: torch.Tensor,
        head_variance: torch.Tensor,
        aggregated_scores: torch.Tensor,
        layer_index: int,
        *,
        defer_tp_reduce: bool = False,
    ) -> None:
        """以 PyTorch 完成单层分数归一化和聚合，作为融合内核的回退实现。

        参数：
            raw_scores：原始分数，形状为 [本地 KV 头数, GQA 组宽, token 数]。
            head_mean：每个 Query 头沿完整 token 维计算的均值。
            head_variance：同一范围内的总体方差，与 head_mean 形状一致。
            aggregated_scores：原地写入的逐 token 输出向量，可以是跨层累积结果，
                也可以是 TP 工作区中当前层独立的一行。
            layer_index：当前层在抽样评分层中的序号，不是原始模型层号。
            defer_tp_reduce：是否将 TP 归约推迟到外层；为 True 时直接写入
                当前层结果，不执行此处的跨 rank 通信或跨层累积。

        先沿 token 维标准化，再跨本地 KV/Query 头取最大值。
        未推迟 TP 归约时，TP 场景还会跨 rank 取最大值。
        首层或推迟归约时覆盖输出；后续层根据配置执行累加或逐元素最大值。
        mean 模式这里只累加，除以评分层数由调用方统一完成。
        """
        # 沿 token 维标准化后的逐 Query 头分数。
        normalized = normalize_head_scores(raw_scores, head_mean, head_variance)
        # 当前层跨本地 KV/Query 头取最大值得到的逐 token 分数。
        layer_scores = normalized.amax(dim=(0, 1))
        if getattr(self, "tensor_parallel_size", 1) > 1 and not defer_tp_reduce:
            import torch.distributed as dist
            from vllm.distributed import get_tp_group

            dist.all_reduce(
                layer_scores,
                op=dist.ReduceOp.MAX,
                group=get_tp_group().device_group,
            )
        if layer_index == 0 or defer_tp_reduce:
            aggregated_scores.copy_(layer_scores)
        elif self.config.layer_aggregation == "mean":
            aggregated_scores.add_(layer_scores)
        else:
            torch.maximum(aggregated_scores, layer_scores, out=aggregated_scores)


def _is_mtp_cache_layer(layer_name: str) -> bool:
    return ".mtp.layers." in f".{layer_name}."
