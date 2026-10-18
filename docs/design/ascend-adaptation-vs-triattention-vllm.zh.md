# 面向 vLLM Ascend 的 TriAttention 适配

[English](ascend-adaptation-vs-triattention-vllm.md) | 简体中文

评分思想和 V3 位置策略来自
[TriAttention 论文](https://arxiv.org/abs/2604.04921)及其
[Apache-2.0 参考实现](https://github.com/WeianMao/triattention/tree/a4bc3c8f709db60f016ef42c3feb290fd0c00c1b)。
本插件自行实现 Ascend 分页缓存运行时、事务协议与校准格式，没有复制参考 CUDA
运行算子。[0.4 历史设计](../archive/design/ascend-adaptation-vs-triattention-vllm-0.4.zh.md)
保留供对照。

## 缓存与评分

模型运行器绑定 128-token 内核块中的稠密 BF16/FP16 K/V 张量。Qwen3.5 混合
缓存管理器可使用 1,152 或 2,048-token 注意力页；插件核对实际布局，将每个
逻辑页展开成 9 或 16 个内核块。只评分全注意力层；Gated-DeltaNet 状态仍归
宿主管理。部分 RoPE 的旋转维度使用三角评分，直通维度加入校准内容点积。
受支持的 mean-score 路径用融合分页键算子，避免先把完整 K 收集成连续张量；
其他形状可使用 Torch 回退路径。详见[方法设计](methods.zh.md)与
[校准指南](../development/calibration-artifacts.zh.md)。

V3 选择保护设定的前缀和最近后缀，再按精确比例向中间各段分配驱逐配额。
TP 各 rank 必须选择相同 token，因此在选取前同步逐层评分。评分层步长可降低
打分工作，但所有目标层仍物化被选中的 K/V。

## 事务边界

scheduler 为符合条件的请求预留私有目标块；worker 在模型步骤后评分并物化
所选 K/V。下一调度屏障提交相应物理长度 offset，已排队批次结束后才释放旧块。
语义 RoPE 位置继续单调增长，缩短的只有物理 attention 长度与 slot mapping。
逐层状态和图捕获写入 slot 会接受检查，避免 MTP 注意力缓存或 GDN 循环状态
错误套用全注意力 offset。

实现对 APC、异步调度、Qwen3.5 的 `mamba_cache_mode=none|align`、KV 头数
可整除时的 TP=2，以及显式开启的真实 MTP2 提供有边界支持。KV transfer、量化
KV、MLA/local/sliding attention 与 PP/DP/DCP/PCP 仍在启动检查中拒绝。
这些是由版本范围和测试保护的宿主内部契约，不是上游冻结的 KV 生命周期 API。

## 性能证据边界

算子加速不能推导服务加速。已发布 0.9 的 SWE 配对逐用户解码 P90 提升，
但每卡输出 token/s 下降。优化结论须选择覆盖相应负载和指标的
[测试规程](../benchmark/README.zh.md)。0.9 精确数值见
[结果范围](../results/README.zh.md)。
