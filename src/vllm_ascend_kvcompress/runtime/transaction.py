# SPDX-License-Identifier: Apache-2.0
"""可安全通过 pickle 从调度器传递给 worker 的 KV 压缩事务方案。"""

from __future__ import annotations

from dataclasses import dataclass

# 此属性把 request_id → CompressionPlan 字典挂在 SchedulerOutput 上，随宿主输出传输。
# 它携带授权和块编号，不携带 tensor、算法评分或 scheduler 的 Block 对象。
# 输出被宿主接受后，scheduler 用同一计划确认事务；worker 下次调度该请求时跟进切表。
PLAN_ATTRIBUTE = "_ascend_kvcompress_plans_v4"


@dataclass(frozen=True)
class CompressionPlan:
    """在 worker 完成私有目标块写入前，保持源 KV 不变。"""

    # anchor 都是本轮 forward 完成后的 token 数：semantic 是原始历史长度，
    # physical 是压缩目标长度；差值用于之后追加 token 的物理位置换算。
    # 源/目的编号属于 full-attention manager 的逻辑页，
    # worker 会展开成 128-token 内核页。
    semantic_anchor: int
    physical_anchor: int
    source_block_ids: tuple[int, ...]
    destination_block_ids: tuple[int, ...]


@dataclass(frozen=True)
class CompressionAnchor:
    """调度器与 worker 共享的最近一次已确认长度映射。

    下次压缩前，新 token 会让语义和物理时间线等量推进，
    因此两者的长度差在两次事务提交之间保持不变。
    """

    semantic_anchor: int
    physical_anchor: int

    @property
    def removed_tokens(self) -> int:
        return self.semantic_anchor - self.physical_anchor

    # 锚点后的新增 token 在两个时间轴上等量增长，因此只需减去累计删除量。
    # 下界截为零也适用于分配器询问落在压缩锚点之前的计数。
    def physical_tokens(self, semantic_tokens: int) -> int:
        return max(0, int(semantic_tokens) - self.removed_tokens)
