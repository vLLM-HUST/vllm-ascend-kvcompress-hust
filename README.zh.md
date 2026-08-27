# vLLM Ascend KV 缓存压缩

[English](README.md) | 简体中文

面向 HUST 维护的 vLLM 昇腾技术栈的可扩展 KV 缓存压缩插件。它将压缩方法接入
vLLM-HUST 的事务式 KV 缓存生命周期，无需修改 vLLM-HUST 或
vLLM-Ascend-HUST 源码树。

当前版本内置以正确性为优先的 TriAttention 方法，并提供用于接入其他压缩算法
的公开方法接口。

## 主要特性

- 与具体算法无关的昇腾 provider，通过配置选择压缩方法。
- 内置 `triattention` token 选择方法。
- 提供公开 Python registry 和第三方 entry-point group。
- 在正式分配 KV 缓存前执行 fail-closed 兼容性检查。
- 通过 vLLM-HUST 原生生命周期实现 scheduler 原子提交、block 回收以及语义/
  物理位置分离。
- 提供英文和简体中文用户文档。

## 兼容性

当前版本面向以下源码版本线：

| 组件 | 兼容版本 |
| --- | --- |
| vLLM-HUST | `>=0.23.1,<0.24` |
| vLLM-Ascend-HUST | `>=0.19.1,<0.20` |
| Python | `>=3.10,<3.15` |

经过测试的参考快照为 vLLM-HUST
`1a06c55468966de8ef471ecb7612c199e15a153a` 和 vLLM-Ascend-HUST
`ac2b94f1536090e2cd0d6c2f8bc8087e336193d5`。

Schema v1 当前支持单张昇腾 NPU、eager 执行、一个普通全注意力 KV group、
block size 128，以及独立连续的 BF16/FP16 K/V 张量。多设备、图模式、混合/
MLA 缓存、滑动窗口注意力、推测解码、KV transfer、稀疏布局和量化 KV 缓存
都会按 fail-closed 原则拒绝。

## 安装

安装到已经包含匹配 editable vLLM-HUST 和 vLLM-Ascend-HUST 包的环境：

```bash
uv pip install -e /path/to/vllm-ascend-kvcompress-hust
```

如果显式设置了 `VLLM_PLUGINS`，请加入规范插件名称：

```bash
export VLLM_PLUGINS=ascend_kvcompress
```

当前 vLLM-HUST 在禁用 prefix caching 时会默认启用独立的 Knorm 压缩器。
必须将其关闭，确保只有一个组件负责修改 KV block table：

```bash
export VLLM_KNORM_ENABLED=0
```

## 快速开始：TriAttention

不同算法共用稳定的 provider 名，通过扁平 JSON 标量 `method` 选项选择具体
实现：

```bash
vllm serve /path/to/model \
  --enforce-eager \
  --no-async-scheduling \
  --no-enable-prefix-caching \
  --block-size 128 \
  --max-model-len 12288 \
  --gpu-memory-utilization 0.8 \
  --kv-cache-compression-config '{
    "schema_version": 1,
    "provider": "ascend_kvcompress",
    "provider_config": {
      "method": "triattention",
      "stats_path": "/path/to/triattention_stats.pt",
      "kv_budget": 2048,
      "recompute_window": 128,
      "protected_recent_window": 128,
      "score_aggregation": "mean",
      "layer_aggregation": "mean",
      "score_chunk_size": 512
    }
  }'
```

提示达到 `kv_budget + recompute_window` 之后开始压缩。budget、recompute
window 和 score chunk 必须是 block size 128 的正整数倍。旧 provider 名
`triattention_ascend` 仍可用于现有配置，但不能选择其他方法。

`score_chunk_size=512` 是偏保守的默认值。增大分块可减少 kernel 启动开销，
但会消耗更多临时设备内存。下方 Qwen2.5-Coder-14B 压力测试在确认单张 64 GiB
昇腾 910B2 可容纳后使用了 8192；该值应针对具体模型和设备调优。

## 内置方法

| 方法 | 策略 | 必需产物 | 状态 |
| --- | --- | --- | --- |
| `triattention` | 基于 query 感知的 post-RoPE token 选择和 KV 压缩 | 完整逐层 TriAttention 统计 | 已完成单 NPU eager 验证 |

TriAttention 支持扁平的逐 query-head 或逐 KV-head 统计，也支持结构化
`layer_stats` 张量。加载器使用 `torch.load(..., weights_only=True)`，绝不
回退到不安全的 pickle 加载。缩放 RoPE 需要精确的 `inv_freq` 和逐层
`freq_scale_sq`。

## 添加压缩方法

新方法需要实现 `KVCompressionMethod` 协议，并在进程内注册 factory，或通过
`vllm_ascend_kvcompress.methods` Python entry-point group 注册。公共 provider
继续负责 vLLM hook、缓存布局检查、调度事务、提交确认和物理解码位置。

完整 API、配置协议、扩展示例和必需测试见
[压缩方法架构](docs/methods.zh.md)。

## A/B 参考结果

重构后的 A/B 套件使用单张昇腾 910B2、Qwen2.5-Coder-14B-Instruct、BF16 KV、
block size 128、eager 模式以及启动时预分配的 20.93 GiB KV 池。服务侧唯一的
A/B 差异是 TriAttention 配置：2048-token budget、128-token recompute/recent
window 和 `score_chunk_size=8192`。

四组匹配测试持久化的输入/输出长度完全一致且没有失败：baseline 和
TriAttention 都是 30/30 请求成功。延迟单位为毫秒，变化值均为 TriAttention
相对 baseline 的变化。

| 场景 | 输入/输出 x 请求数 | 总 tok/s，关到开 | P99 TTFT，关到开 | 平均 TPOT，关到开 |
| --- | ---: | ---: | ---: | ---: |
| prefix-repetition-online | 2560/256 x 4 | 131.76 到 129.72 (-1.55%) | 590.92 到 747.60 (+26.52%) | 81.79 到 82.69 (+1.11%) |
| random-online | 2560/32 x 2 | 796.29 到 769.50 (-3.36%) | 365.03 到 455.17 (+24.69%) | 80.72 到 81.55 (+1.03%) |
| knorm-kv-compression-longctx | 8192/256 x 8 | 1246.91 到 1219.72 (-2.18%) | 2628.74 到 2808.23 (+6.83%) | 93.84 到 95.72 (+2.00%) |
| kv-pressure-online | 8192/64 x 16 | 5060.13 到 5113.44 (+1.05%) | 20440.16 到 20057.73 (-1.87%) | 182.85 到 194.95 (+6.61%) |

每个 TriAttention 请求都产生一次压缩提交并收到回执。2560-token 提示的物理
block 从 20 个降至 16 个；8192-token 提示从 64 个降至 16 个，每请求归还
48 个 block。长上下文场景观测到的活跃 KV 池峰值由 29.6% 降至 12.9%，压力
场景则由 94.7% 降至 26.2%。

allocator 会在服务启动时预留 KV 池，因此回收 block 不会降低进程级 HBM
占用。活跃 KV 池占用和归还的 block 才是有意义的容量信号。这些是单次工程
验证结果，不是跨平台性能保证；有损压缩质量仍需单独评估。

[基准测试与结果解读](docs/benchmarking.zh.md)提供 baseline/TriAttention 的
准确服务启动命令、四个 benchmark 客户端命令、复现协议和结果报告要求。

## 验证

```bash
python -m pytest -q
ruff check .
ruff format --check .
python -m compileall -q src tests
```

运行时变更还必须在其他条件完全一致的长上下文负载上执行压缩关闭/开启对比。
记录提示和输出长度、并发度、TTFT、TPOT/ITL、吞吐量、成功请求数、释放的 KV
block 和 NPU 内存。始终选择空闲且已分配的 NPU，并在测试结束后确认服务进程
退出。

## 文档维护规范

- `README.md`、`README.zh.md`、`CONTRIBUTING.md`、
  `CONTRIBUTING.zh.md` 和 `docs/` 下的公开文件只描述稳定、可发布的行为。
- 本地实验、中间 benchmark 日志、机器路径和模型专用验证报告放入
  `docs/dev/`。
- 生成的校准文件和本地产物放入 `artifacts/`。
- `docs/dev/` 和 `artifacts/` 被有意加入 git ignore。
- 英文公开文档为规范版本；同一修改必须同步更新对应 `.zh.md` 文档。

## HUST 相关项目

- [vLLM-HUST](https://github.com/vLLM-HUST/vllm-hust)
- [vLLM-Ascend-HUST](https://github.com/vLLM-HUST/vllm-ascend-hust)
- [vLLM-HUST Benchmark](https://github.com/vLLM-HUST/vllm-hust-benchmark)

## 许可证

Apache License 2.0。详见 [LICENSE](LICENSE) 和 [NOTICE](NOTICE)。
