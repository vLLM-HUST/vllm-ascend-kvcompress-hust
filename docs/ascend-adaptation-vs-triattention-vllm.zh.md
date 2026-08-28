# 相较 TriAttention vLLM Runtime 的核心 Ascend 适配

[English](ascend-adaptation-vs-triattention-vllm.md) | 简体中文

本文说明本仓库当前实现相较
`triattention/triattention/vllm/` 参考 runtime 的核心改动。内容聚焦
runtime 与平台适配，不是逐行移植记录，也不表示两者功能完全对等。

## 概要

Ascend 实现保留 TriAttention 的核心算法：用校准后的未来 query 统计为
缓存中的 post-RoPE key 打分，保护最近 token，选择价值最高的 token，
并保持其因果顺序。

算法外围的接入层则明显不同。参考 runtime 是面向通用 vLLM 的
TriAttention 专用兼容层，会 patch scheduler、worker、分配、输入、engine 和
runner 路径。本仓库改用 vLLM-HUST 的事务式 KV 压缩协议，只在 Ascend
model runner 上安装少量 hook，并通过与方法无关的 provider 暴露
TriAttention。

## 架构对比

| 方面 | 参考 `triattention/vllm` runtime | 当前 Ascend 实现 |
| --- | --- | --- |
| 接入 | TriAttention 专用 scheduler、worker、runner、分配与 engine patch | vLLM-HUST 原生压缩 plan 加少量 Ascend hook |
| 配置 | `TRIATTENTION_*`、`TRIATTN_RUNTIME_*` 环境变量 | 带方法名且经校验的 `--kv-cache-compression-config` JSON |
| 执行 | 面向 CUDA 的 PyTorch/Triton 路径 | torch-npu、Ascend Triton 打分 kernel 及 PyTorch 回退路径 |
| 缓存布局 | 适配多种推断出的 runner 与合并 K/V 布局 | 精确的独立 K/V 分页布局 `[blocks, 128, kv_heads, head_dim]` |
| 选择布局 | 逐 head、逐 layer 与逐 layer/head | 所有 head/layer 共享一个请求级有序 token 集 |
| 压缩周期 | runtime 管理的有状态多轮压缩 | final-prefill 初始事务加 decode 期间重复事务 |
| Block 所有权 | 自定义事件和 block manager 直接协调 | scheduler 负责预留、校验、提交、释放和回执 |
| 位置处理 | effective-length 跟踪及多处输入/分配 patch | 保留语义 RoPE 位置，只偏移物理 slot 与长度 |
| 执行模式 | 以 eager 兼容面为主 | 已验证 eager 和 ACL graph；仍拒绝 async scheduling |
| 扩展单元 | TriAttention runtime | 公共 provider 与公开 `KVCompressionMethod` registry |

## 1. 事务式调度与 Block 所有权

参考 runtime 将 TriAttention signal 和压缩事件附加到通用 vLLM 对象上，
由兼容层将这些事件与 block manager 协调，并在需要时 patch 异步队列边界。

Ascend 实现将首轮压缩交给 vLLM-HUST 原生生命周期：

1. KV 正式分配前，provider 声明压缩阈值、recompute 需求、最大物理长度和
   目标 block 策略。
2. 到达 final prefill 时，scheduler 开启事务并提供源 block table 与私有目标
   block。
3. `NPUModelRunner` 写入压缩后的 K/V 并返回 `KVCacheCompressionPlan`；
   它不自行释放 scheduler 所有的 block。
4. scheduler 校验预期 table、原子提交、释放 block，并在后续输出中返回回执。
5. runner 只在收到回执后才替换本地 request 和 input-batch block table。

后续周期由 `stateful.py` 中的插件本地兼容层扩展相同校验与提交规则。
当物理缓存再次达到 `kv_budget + recompute_window` 时，它会武装新事务、
预留新的私有目标，并在替换前校验当前 block table。因此无需修改
vLLM-HUST 或 vLLM-Ascend-HUST 源码树也可支持重复压缩。

当前 schema 没有另行定义异步 plan 交付协议，因此 async scheduling 仍按
fail-closed 原则拒绝。

## 2. Ascend Paged KV 写入

Ascend attention 绑定形状为 `[num_blocks, 128, num_kv_heads, head_dim]`
的独立连续 K/V 张量。因此实现会：

- 通过请求 block ID 将逻辑 token index 映射到物理 slot；
- 每次事务只计算一次源/目标 slot mapping；
- 覆盖任何目标之前，先将 K/V gather 到持久的设备端 workspace；
- 所有 layer 复用相同 mapping 和 workspace；
- 使用设备侧 `index_copy_` 写入选中 token。

持久 workspace 使源/目标 block 重叠时仍然安全，也避免为每层、每次事务
重新分配稠密 K/V buffer。绑定前会校验 block size 128 和精确的独立 K/V
布局，不支持的布局直接 fail closed。

## 3. Ascend 打分热路径

最初的 Ascend 移植只使用向量化 PyTorch 运算。当前 `mean` 聚合路径新增
Ascend Triton kernel，直接读取 paged cache 中的 post-RoPE key，并融合：

- 逻辑 token 到 paged cache 的寻址；
- 校准后的复数 query 均值；
- RoPE 相位和预计算的未来 offset 三角均值；
- 频率缩放；
- 幅值回归修正项。

这样可以消除打分热路径中的稠密 key gather。频率缩放、修正系数和
offset 三角均值在缓存绑定时预计算，打分 workspace 也会在多轮事务间复用。

如果专用 kernel 不适用，例如设备或聚合方式不支持，方法会回退到分块向量化
PyTorch；`score_chunk_size` 限制该回退路径的工作集。

## 4. 所有层共享一个 Token 布局

参考 runtime 可按 layer 或 head 保留不同 token 集。当前 vLLM-HUST 请求的
block table 表示所有 attention layer 共享的一个物理顺序，因此本实现只生成
一个请求级 `keep_indices` 张量：

1. 分别归一化每个校准 query head 的分数。
2. 每个参与打分的 layer 内以 `max` 聚合 query head。
3. 按配置使用 `mean` 或 `max` 聚合 layer。
4. 强制保留 protected recent window。
5. 执行一次全局 Top-K，再对选中 index 排序。
6. 为每个 K/V layer 写入同一组有序 index。

`score_layer_stride=4` 默认均匀抽样校准层参与选择，但仍会压缩每个
cache layer。设为 `1` 可恢复全层打分，代价是更高的事务延迟。这是 Ascend
性能策略，不会改变 cache 布局或 block 所有权协议。

## 5. 语义与物理状态

压缩后，每个请求同时具有用于模型进度/RoPE 的语义长度，以及用于 KV
分配/slot mapping 的较短物理长度。provider 为每个已提交请求保存一对锚点：

```text
已删除 token = 语义锚点 - 物理锚点
物理位置 = 语义位置 - 已删除 token
```

模型可见位置仍为语义位置。provider 只在 attention sequence length、乐观物理
长度和现有 block-table slot-mapping 调用中应用 offset。每请求 offset 常驻
NPU，只在 request row 或压缩状态变化时更新；稳定 decode step 不再发生
CPU→NPU offset 复制，也不会重算第二份完整 slot mapping。

每次提交回执都会更新 offset，因此重复压缩时语义位置仍单调递增，物理
decode 写入则保持紧密。

## 6. ACL Graph 兼容

provider 不再要求 `--enforce-eager`。物理位置转换使用固定设备 buffer 和可进入
graph 的 NPU tensor 操作，正常 torch-npu 启动预检仍保持开启。在 schema v1
兼容范围内，eager 和 ACL graph 都受支持。

这不代表同时支持 async scheduling、多设备、其他 model runner 或任意 cache
布局；这些组合仍会明确报告兼容性失败。

## 7. 严格的校准与 RoPE 校验

参考 loader 提供多种模型与 RoPE 重建回退。Ascend loader 在 KV 分配前会
刻意执行更严格的校验：

- 在 CPU 上使用 `torch.load(..., weights_only=True)`；
- 支持扁平逐 head 和结构化逐 layer payload；
- 校验 layer 覆盖、head 数、head dim、model type、RoPE style 和 `rope_theta`；
- 针对 GQA 对 query-head 统计分组，不复制 KV head；
- scaled 或非默认 RoPE 要求精确的逐层 `inv_freq`；
- scaled RoPE 还要求精确的逐层 `freq_scale_sq`。

存在歧义的 metadata 会 fail closed，而不是静默选择可能改变 token 选择语义的
频率模型。

生成流程、schema、当前本地文件和溯源要求见
[TriAttention 校准产物](calibration-artifacts.zh.md)。

## 8. 与方法无关的 Provider 边界

runtime 职责被有意拆分为：

- `provider.py`：兼容性、事务、回执与请求语义/物理状态；
- `stateful.py`：为原生 manager/scheduler 协议补充重复事务兼容层；
- `methods/base.py`：定义面向框架的方法接口；
- `methods/registry.py`：解析内置与第三方 factory；
- `methods/triattention/`：只负责校准、打分、选择和 K/V 写入。

其他压缩算法可复用同一 Ascend 生命周期，无需复制 TriAttention 接入层。

## 保留的算法行为

适配保留了以下 TriAttention 核心特性：

- 校准后的复数未来 query 统计；
- 基于几何间隔未来 offset 的 post-RoPE key 打分；
- 幅值回归修正；
- 逐 head 分数归一化；
- protected recent token；
- Top-K 选择后按因果顺序排序；
- K/V 写入使用相同选择 index。

## 当前范围与证据

已验证 schema 包括单张 Ascend NPU、标准 v1
`NPUModelRunner`/`AscendAttentionBackend`、eager 或 ACL graph、一个普通
full-attention KV group、独立 BF16/FP16 K/V 以及 block size 128。

多设备并行、async scheduling、混合/MLA cache、滑动窗口或 chunked-attention
cache 布局、推测解码、KV transfer、稀疏/模型内置压缩以及量化 KV 仍不支持。
独立验证时还必须关闭 prefix caching 和独立 Knorm 压缩器。

性能、容量、质量和验证状态见[当前基准测试结果](resuts.zh.md)；可复现协议见
[基准测试与结果解读](benchmarking.zh.md)。
