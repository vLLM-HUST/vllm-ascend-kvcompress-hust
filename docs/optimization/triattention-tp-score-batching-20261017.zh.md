# TriAttention TP 评分合并与 SWE 配置筛选

[English](triattention-tp-score-batching-20261017.md) | 简体中文 |
[优化记录索引](README.zh.md)

日期：2026-10-17。范围：固定的 Qwen3.5-35B-A3B BF16、TP=2、Ascend 910B2
宿主上的 0.10.0 源码实现。这是工程记录，不能外推到所有模型、宿主或请求组合。

## 问题与实现

原 TP=2 评分路径
对每个采样的注意力层分别执行 HCCL MAX 归约。Qwen3.5 采用 stride 2 时，
一次压缩选择要为五个采样层执行五次 collective。

新实现预分配连续的评分工作区，将各采样层归一化后的评分写入独立行，
对整个工作区执行一次 HCCL MAX，再沿层轴聚合。MAX 对每个层/token 项独立
生效，因此合并通信与逐层跨 rank 归约得到相同的结果。Ascend 归一化算子
使用与 Torch 回退路径一致的最小方差。TriAttention 的校准评分、V3 保护窗口、
分段配额及全部十个全注意力层的 KV 搬运均保留。实现见
[method.py](../../src/vllm_ascend_kvcompress/methods/triattention/method.py)
和[kernels.py](../../src/vllm_ascend_kvcompress/methods/triattention/kernels.py)。

此项代码改动减少每次压缩事件的同步工作。服务性能还取决于**何时**触发压缩，
因此候选筛选保持同一份优化源码，只调整物理 KV 预算和请求输出门槛。
门槛检查的是请求配置的 `max_tokens`；2048-token 门槛会跳过输出上限较小的请求。

## 历史测试与证据

[完整冻结记录](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/b3d2ed2692b5575b5ae77b93fc82e32c9119be8e/archive/mod-targeted-20261023/compress-mixed/snapshot/docs/optimization/triattention-tp-score-batching-20261017.zh.md)保留固定工况、测量表、负收益和限制。
[聚合原始数据](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/results/evidence/kvcompress-qwen35-20261017-v010-swe-pair.json)。
旧原文中的 B0/B1 是其原始运行标签，不重定义项目当前固定的基线。
