# Ascend 适配与 TriAttention 的差异

[English](ascend-adaptation-vs-triattention-vllm.md) | 简体中文

## 来源边界

算法来源是 TriAttention 论文以及采用 Apache-2.0 的参考仓库提交
`a4bc3c8f709db60f016ef42c3feb290fd0c00c1b`。参考实现用未旋转 query 的
统计量对 post-RoPE key 评分，并保护最近窗口。本仓库保留该评分模型，但为
vLLM Ascend 分页 KV cache 重新实现运行时，未复制上游 CUDA runtime kernel。

## Ascend 侧变化

| 方面 | 参考实现 | 本插件 |
| --- | --- | --- |
| 打包 | 独立研究脚本/patch | 带 `vllm.general_plugins` 与 Extension Manager manifest 的 Python 包 |
| Cache 布局 | 参考连续 tensor | 当前 vLLM-Ascend attention 实现解包的分页 K/V cache |
| 评分 | 面向 PyTorch/CUDA 的路径 | 直接读取分页 cache 的 Triton-Ascend 评分 |
| 聚合 | 中间归一化 score tensor | 融合归一化、query-head 最大值和跨层累加 |
| 内存 | 每次操作临时分配 | 持久 score、aggregate、dense index 和 K/V copy workspace |
| 生命周期 | 研究型集成 | scheduler/worker 镜像，在同步调度边界提交 |
| 位置 | 参考序列操作 | 语义 RoPE 位置保持单调，只缩短物理 slot 和 attention 长度 |
| 启用 | 环境变量/研究集成 | Manager 托管 JSON，或开发用显式环境变量 |

## 当前宿主挂接点

公开发现契约是 `vllm.general_plugins`。显式启用后，0.4 适配当前内部符号：

- `Scheduler` 和 `KVCacheManager.allocate_slots`：管理物理 block；
- scheduler 侧 `Request.max_tokens` 与 worker 侧
  `CachedRequestState.sampling_params.max_tokens`：对称判断短输出绕过；
- KV 初始化后解析出的 Ascend input-batch 具体 block table 的
  `compute_slot_mapping`：完成语义位置到物理 slot 的转换；
- `NPUModelRunner.initialize_kv_cache`、`_update_states`、
  `_build_attention_metadata` 和 `sample_tokens`：绑定 cache、镜像事务、设置
  attention 长度并在单步结束后物化。

NPU runner 采用延迟挂接。请求输出字段缺失或类型不正确时拒绝运行，不做猜测。
测试和运行时会检查精确类、属性与方法契约，但它们不是上游冻结 API；每次宿主升级
都必须重新检查兼容性并完成验收。

## 事务不变量

压缩与一个模型步骤同步：

1. scheduler 和 worker 在确定的 token 阈值分别为同一请求准备事务；
2. sample 后，worker 选择 token，并把每一层复制到私有且 block 对齐的目标区；
3. 下一个 scheduler 屏障记录已移除 token 偏移，缩短物理 token 数并释放尾部块；
4. 后续 RoPE 使用原始语义 token 编号，cache slot mapping 和 attention metadata
   使用缩短后的物理长度。

在具备单独协议和验收套件前，异步调度、speculative decoding、prefix cache、
KV transfer、量化 KV、hybrid/MLA/local attention、分布式并行以及 BidKV 等非
标准 scheduler 均会被拒绝。

## 优化边界

融合聚合和持久 workspace 用于降低压缩热路径的分配与 kernel launch；
`score_layer_stride` 可以抽样评分层，但所有层仍会完成物化；请求输出门槛可在短生成
无法摊薄开销时跳过评分和 copy。这些属于实现优化，不是性能保证。只有在支持的宿主
快照上完成匹配 NPU 测量后，才能作为 0.4 证据。
