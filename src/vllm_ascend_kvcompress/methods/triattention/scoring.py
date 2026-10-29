# SPDX-License-Identifier: Apache-2.0
"""TriAttention 方法自有的向量化 PyTorch-NPU 评分实现。"""

from __future__ import annotations

import torch

from .stats import DeviceLayerCalibrationStats


# 用对数数量的位置覆盖近处到远处的未来 Query，而不是逐个预测未来 token。
# 这些 offset 是评分采样位置，与实际 decode 步数或保留 token 数无关。
def build_geometric_offsets(maximum: int, device: torch.device) -> torch.Tensor:
    """在 device 上构造 1、2、4……直至 maximum 的几何采样偏移。"""
    if maximum < 1:
        raise ValueError("maximum scoring offset must be positive")
    offsets: list[float] = []
    value = 1
    while value <= maximum:
        offsets.append(float(value))
        value *= 2
    return torch.tensor(offsets, device=device, dtype=torch.float32)


def score_post_rope_keys(
    keys: torch.Tensor,
    stats: DeviceLayerCalibrationStats,
    *,
    round_start: int,
    offsets: torch.Tensor,
    aggregation: str,
) -> torch.Tensor:
    """对已应用 RoPE 的 key 评分，不将设备数据传回主机。

    参数：
        keys：缓存 key，形状为 [tokens, kv_heads, head_dim]。
        stats：按 KV 头分组的 query 校准统计。
        round_start：首个未来 query 的绝对语义位置。
        offsets：按几何间隔采样的未来 query 偏移。
        aggregation：对未来偏移使用 mean 或 max 聚合。

    返回：
        形状为 [kv_heads, query_heads_per_kv_head, tokens] 的分数。
    """
    if keys.ndim != 3:
        raise ValueError("keys must have shape [tokens, kv_heads, head_dim]")
    token_count, kv_heads, head_dim = keys.shape
    if head_dim % 2:
        raise ValueError("key head dimension must be even")
    rotary_dim = head_dim if stats.rotary_dim is None else stats.rotary_dim
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError("calibration rotary dimension is incompatible with keys")
    freq_count = rotary_dim // 2
    if stats.q_mean_real.shape[0] != kv_heads:
        raise ValueError("calibration KV-head count does not match cache keys")
    if stats.q_mean_real.shape[-1] != freq_count:
        raise ValueError("calibration frequency count does not match cache keys")
    if aggregation not in ("mean", "max"):
        raise ValueError(f"unsupported score aggregation: {aggregation}")
    if rotary_dim < head_dim:
        if stats.q_pass_mean is None:
            raise ValueError("partial-RoPE scoring requires q_pass_mean")
        if stats.q_pass_mean.shape[-1] != head_dim - rotary_dim:
            raise ValueError("q_pass_mean width does not match unrotated keys")
    return _score_post_rope_keys(
        keys, stats, round_start=round_start, offsets=offsets, aggregation=aggregation
    )


def _score_post_rope_keys(
    keys: torch.Tensor,
    stats: DeviceLayerCalibrationStats,
    *,
    round_start: int,
    offsets: torch.Tensor,
    aggregation: str,
) -> torch.Tensor:
    """计算已验证的缓存和统计；运行时在绑定后按块调用，不重复校验形状。

    独立调用者使用上方公开入口；method 的配置、统计与缓存分别由
    配置解析、兼容性检查和 provider 绑定边界验证。
    """
    token_count, kv_heads, head_dim = keys.shape
    rotary_dim = head_dim if stats.rotary_dim is None else stats.rotary_dim
    freq_count = rotary_dim // 2

    # 参考公式入口：把 K 分成复数实/虚部，和离线未施加 RoPE 的 Q 均值相乘。
    # K 中已经含历史位置相位，因此只需用 future semantic position 旋转 Q；
    # 重排 cache 的物理下标不改变 K 的相位，不能用压缩后位置重新编码。
    keys_fp32 = keys.to(dtype=torch.float32).permute(1, 0, 2)
    rotated_keys = keys_fp32[..., :rotary_dim]
    if stats.rope_style == "interleaved":
        key_real = rotated_keys[..., 0::2]
        key_imag = rotated_keys[..., 1::2]
    else:
        key_real = rotated_keys[..., :freq_count]
        key_imag = rotated_keys[..., freq_count:rotary_dim]

    q_real = stats.q_mean_real.unsqueeze(2)
    q_imag = stats.q_mean_imag.unsqueeze(2)
    key_real = key_real.unsqueeze(1)
    key_imag = key_imag.unsqueeze(1)
    # 计算 Q 均值与已旋转 K 的复数乘积，随后施加未来 Query 的相位。
    # 先保留频率轴，再对频率求和，得到每个 Query 头对每个 token 的评分。
    product_real = q_real * key_real + q_imag * key_imag
    product_imag = q_imag * key_real - q_real * key_imag
    # 缓存 key 已包含一次 RoPE 注意力缩放因子，
    # 因此这里只补上 query 侧对应的因子。
    # TriAttention 的未旋转 key 路径则乘以该因子的平方。
    frequency_scale = torch.sqrt(stats.freq_scale_sq).unsqueeze(2)

    phases = (offsets.unsqueeze(1) + float(round_start)) * stats.omega.unsqueeze(0)
    # 未来位置采样仅为 1,2,4,...；mean 可先平均 sin/cos 后再求点积。
    # max 必须逐 offset 计算候选后取最大，不能用 mean 的预计算替代。
    if aggregation == "mean":
        cosine = torch.cos(phases).mean(dim=0).view(1, 1, 1, freq_count)
        sine = torch.sin(phases).mean(dim=0).view(1, 1, 1, freq_count)
        scores = (frequency_scale * (product_real * cosine - product_imag * sine)).sum(
            dim=-1
        )
    else:
        scores = torch.full(
            (kv_heads, stats.q_mean_real.shape[1], token_count),
            float("-inf"),
            dtype=torch.float32,
            device=keys.device,
        )
        for phase in phases.unbind(0):
            cosine = torch.cos(phase).view(1, 1, 1, freq_count)
            sine = torch.sin(phase).view(1, 1, 1, freq_count)
            candidate = (
                frequency_scale * (product_real * cosine - product_imag * sine)
            ).sum(dim=-1)
            scores = torch.maximum(scores, candidate)

    # 幅度项补偿 |E[Q]| 与 E[|Q|] 的差异，不随 future position 改变。
    # 它与上面的相位项相加；partial RoPE 还需下面未旋转子空间的普通 Q·K。
    key_abs = torch.sqrt(key_real.square() + key_imag.square())
    q_mean_abs = torch.sqrt(
        stats.q_mean_real.square() + stats.q_mean_imag.square() + 1e-8
    )
    extra_coefficient = (stats.q_abs_mean - q_mean_abs).unsqueeze(2)
    extra = (key_abs * extra_coefficient * frequency_scale).sum(dim=-1)
    scores = scores + extra

    # Qwen3.5 等部分 RoPE 模型的大多数 key 维度不参与旋转。
    # 这些维度经校准后的直接 Q·K 贡献与位置无关，也必须计入；
    # 否则仅依靠相位的评分会忽略每个 key 的 75% 维度。
    if rotary_dim < head_dim:
        key_pass = keys_fp32[..., rotary_dim:].unsqueeze(1)
        content_score = (stats.q_pass_mean.unsqueeze(2) * key_pass).sum(dim=-1)
        scores = scores + content_score
    return scores


def normalize_head_scores(
    scores: torch.Tensor, mean: torch.Tensor, variance: torch.Tensor
) -> torch.Tensor:
    """沿 token 维度归一化每个已校准 query 头的分数。"""
    denominator = torch.sqrt(variance.clamp_min(1e-6)).unsqueeze(-1)
    return (scores - mean.unsqueeze(-1)) / denominator
