# vLLM Ascend KV Compression

[English](README.md) | 简体中文

面向与上游对齐的 vLLM-HUST、vLLM-Ascend-HUST 的独立 TriAttention KV-cache
压缩插件。已发布的 0.7 版加入论文 V3 位置策略与实验性 Qwen3.5 支持；未发布的
0.8 工作树适配同步后的宿主，且不修改宿主源码仓库。

> 状态：0.8 开发候选版，尚未发布或完成验收。匹配的 CANN 9.1/PyTorch 2.13
> 宿主已在 Qwen3.5 BF16/TP=2、配置 262,144-token 上下文时启用 APC、align、
> 异步调度、显式 MTP2 与 FULL_AND_PIECEWISE 图。官方 SWE C4／60 秒协议短测
> 通过，服务端确有前缀命中和双 rank 压缩回执。官方 SWE C4 两侧 900 秒窗口
> 均协议有效；AgentX C4 两侧 900 秒也通过协议检查。但压缩使 SWE 与 AgentX
> 总输出吞吐分别下降 7.88% 和 8.46%。Qwen3.5 部分 RoPE 融合评分已通过
> NPU 数值冒烟与微基准；新的 SWE 900 秒配对把 1,024-token 重评分间隔下的
> 吞吐退化缩小至 2.75%，另测的 4,096-token 候选仍退化 3.00%。批量 V3
> 选词与五层评分的单独质量冒烟为 4/4，但 SWE 同源码配对吞吐仍退化
> 2.95%；将压缩输出门槛提高到 512 的另一组 SWE 配对仍退化 4.22%。
> 同一门槛的 AgentX C4／900 秒新配对，总输出吞吐提升 11.85%、解码 P90
> 提升 5.31%，但 TTFT P95 变差 20.60%。这是本地 900 秒正收益点，
> 不是一小时正式成绩或版本验收；仍待合入后复测与用户确认。
> MTP2 仍是默认拒绝启动的实验性功能；AgentX 两侧按其
> 强制接受率规则均关闭 MTP。详见
> [当前前沿测试协议](docs/frontier-benchmarking.zh.md)；
> [0.7 验收记录](docs/validation.zh.md)仅作为历史资料。

## 归属与维护

- 学校：华中科技大学（HUST）
- 课题组：CGCL
- 指导教师：万瑶教授
- 主要负责人：刘思辰（[@Seas0](https://github.com/Seas0)）
- 当前维护者：张家万（[@Jiawan23](https://github.com/Jiawan23)）、
  韦若皓（[@kotoriqaq0](https://github.com/kotoriqaq0)）、刘思辰
  （[@Seas0](https://github.com/Seas0)）

团队同意持续维护 vLLM-HUST、vLLM-Ascend-HUST 兼容性，并通过 vLLM-HUST
Extension Manager 发布。本项目是 CGCL 维护的独立插件，不是宿主内置代码。

## 来源与可再分发范围

评分方法改编自 [TriAttention](https://github.com/WeianMao/triattention) 提交
[`a4bc3c8f`](https://github.com/WeianMao/triattention/tree/a4bc3c8f709db60f016ef42c3feb290fd0c00c1b)
及论文 [*TriAttention: Efficient Long Reasoning with Trigonometric KV
Compression*](https://arxiv.org/abs/2604.04921)。Ascend 分页 KV 运行时为重新实现，
未复制参考仓库的 CUDA 算子。

仓库代码和文档采用 Apache-2.0。模型权重、数据集、原始日志和校准统计不进入
wheel/sdist。仓库中仅供开发使用的统计产物缺少完整模型/数据来源记录，不能仅凭
本仓库许可证进行再分发。详见 [NOTICE](NOTICE)、[归属与许可](docs/ownership-and-licensing.zh.md)
和[校准产物说明](docs/calibration-artifacts.zh.md)。

## 兼容范围

| 组件 | 支持版本 | 已验证快照 |
| --- | --- | --- |
| vLLM-HUST / `vllm` | `>=0.29.1.post1.dev0,<0.30`（0.8 候选） | `fc06902b7d`（`0.29.1.post1.dev843+gfc06902b7.empty`） |
| vLLM-Ascend-HUST / `vllm-ascend` | `>=0.25.1rc2.dev0,<0.26` | `5422a07c4`（`0.25.1rc2.dev616+hust.20260903.4.g5422a07c4`） |
| Extension Manager | `>=0.2.0.dev0,<0.3` | 已安装 `0.2.0.dev0` |
| Python | `>=3.10,<3.15` | 3.11.16 |

标准验收拓扑仍为单 Ascend NPU、v1 调度器与 `NPUModelRunner`、单一全注意力
KV 组、block size 128、稠密 BF16/FP16 K/V。Qwen3.5 text/MoE 混合模型额外支持
一个全注意力组加 Gated-DeltaNet 状态组，支持 `mamba_cache_mode=none` 或
`align`；只有 TP
大小可整除 KV 头数时才允许张量并行（Qwen3.5-35B-A3B 为 TP=2）。已验证的
Ascend 宿主使用 32,768-token 跨组 scheduler 对齐、2,048-token 全注意力页和
128-token attention 内核缓存块；插件会校验这三层粒度，并按 16:1 展开注意力
映射。其他组合在启动阶段拒绝。manifest 使用稳定的
`vllm.general_plugins` 发现入口；当前宿主没有冻结的原生 KV 生命周期 API，因此
通过窄版本范围与契约测试保护内部适配点。

## 独立安装、启用、禁用和卸载

先安装成套锁定的宿主栈，再安装发布包。插件 wheel 特意不把 `vllm` 和
`vllm-ascend` 声明为 Python 包依赖：这些硬件相关包必须作为一套经过验证的运行时
统一部署；宿主兼容范围由 Extension Manager 清单和启动检查约束。这样安装插件时
也不会让 pip 重新解析已经部署好的整套宿主环境。

当前 vLLM 0.29 / vLLM-Ascend 0.25 开发版本线不能混入 Triton-Ascend 3.2.2。
旧 Triton wheel 锁定 NumPy 1.26.4，而当前 vLLM 要求
`opencv-python-headless>=4.13`，其可用 wheel 要求 NumPy 2；把 OpenCV 降到
4.9 又会违反 vLLM 的依赖约束。应使用宿主栈配套的 Triton-Ascend 3.6。

以下命令安装已发布的 0.7 包，**不是**当前 0.8 开发候选版，也不是 0.29 宿主的
已验证安装命令：

```bash
python -m pip install --no-deps vllm-ascend-kvcompress-hust==0.7.0
```

全新环境需先从项目组认可的软件源单独安装 `vllm-hust-ext`，再执行上述命令。
源码/插件测试也应先准备锁定宿主栈，再执行
`python -m pip install --no-deps .`。

复制 [examples/triattention.json](examples/triattention.json)，将 `stats_path`
改为产物保存位置。若文件不存在，启用后的插件会在 vLLM 加载服务权重前生成
模型匹配的统计，原子保存并释放临时模型，然后继续启动。生产环境应设置
`calibration_input_path`，指向许可清晰、具有代表性的 UTF-8 语料；内置文本仅用于
bootstrap。然后执行：

```bash
vllm-hust-ext extension validate org.vllm-hust.ascend-kvcompress
vllm-hust-ext extension configure org.vllm-hust.ascend-kvcompress \
  --file /absolute/path/triattention.json
vllm-hust-ext extension enable org.vllm-hust.ascend-kvcompress
vllm-hust-ext extension status org.vllm-hust.ascend-kvcompress

export VLLM_PLUGINS=ascend,ascend_kvcompress
vllm-hust-ext run -- vllm serve /path/to/model \
  --block-size 128 --no-enable-prefix-caching --no-async-scheduling
```

若未设置 `VLLM_PLUGINS`，vLLM 会发现全部已安装插件；但本插件在 Manager 或
直接启用标志激活前保持惰性。切换状态前先停止所有宿主进程：

```bash
vllm-hust-ext extension disable org.vllm-hust.ascend-kvcompress
vllm-hust-ext extension forget org.vllm-hust.ascend-kvcompress
python -m pip uninstall vllm-ascend-kvcompress-hust
```

上述操作仅改变插件包和 Manager 状态，不修改宿主代码。源码环境安装见
[环境指南](docs/environment-installation.zh.md)，隔离 wheel 流程见
[打包与发布](docs/packaging-and-release.zh.md)。

不经过 Manager 的开发启用方式：

```bash
export VLLM_PLUGINS=ascend,ascend_kvcompress
export VLLM_ASCEND_KVCOMPRESS_ENABLED=1
export VLLM_ASCEND_KVCOMPRESS_CONFIG=/absolute/path/triattention.json
vllm serve /path/to/model --block-size 128 --no-enable-prefix-caching
```

取消两个 `VLLM_ASCEND_KVCOMPRESS_*` 变量并重启即可禁用。

Qwen3.5-35B-A3B 使用专用的
[Qwen3.5 示例配置](examples/qwen3.5-35b-a3b-triattention.json)与 TP=2。服务加载
权重前，临时校准模型会自动分布到可见 NPU。详见
[Qwen3.5 适配说明](docs/qwen3.5-35b-a3b-adaptation.zh.md)。较早宿主快照的
GDN 自定义算子仍使用旧式 `int[]` 元数据 ABI，需 `--enforce-eager` 与显式
`VLLM_ASCEND_KVCOMPRESS_QWEN_GDN_LIST_COMPAT=1` 兼容桥；同步后的宿主已暴露
Tensor ABI，且无需 `--enforce-eager` 即完成 FULL_AND_PIECEWISE 图捕获。
兼容桥仅在精确旧 schema 上生效，不修改任一宿主仓库。

## 运行时与长上下文优化

显式启用后，适配器校验并挂接当前调度器、KV-cache manager、Ascend 具体 block
table 和 `NPUModelRunner`。压缩仅在同步模型步之后提交：语义 RoPE 位置保持不变，
物理 attention/slot 索引使用逐请求 offset，旧 block 在下一调度屏障释放。

0.6 将所有抽样层的 RoPE 相位系数一次性生成，并在分页 K 直接评分中复用。Ascend
910B2 三进程中位数中，每个抽样层的摊销评分耗时从原相位内算路径的 0.460 ms 降至
0.270 ms，降低 41.3%。已验证 stride 为 8，即 48 层中抽 6 层进行选择；全部 K/V
层仍会物化。持久 workspace、融合聚合、设备端 offset、动态 JIT 长度与短输出绕过
继续保留。曾实现并验证一个 Triton K/V 融合 copy 候选，但其耗时为 14.504 ms，
而保留路径仅 0.334 ms，因此已撤回该负优化。

V3 策略会硬保护可配置的前缀和最近窗口，再把驱逐额度按比例分配到等长的中间
上下文分段。对 Qwen3.5 而言，若只评分 64 个旋转维度，就会忽略 256 维 key 中
75% 的内容，因此适配器还会在其余 192 个非旋转维度上加入校准后的直接 Q·K
内容项。只评分和压缩 10 个全注意力层；Gated-DeltaNet 循环状态继续由宿主原生
生命周期管理。张量并行各 rank 在选择前同步逐层分数，从而得到完全相同的 token
集合。

完整 Qwen2.5 A2 工程矩阵覆盖 8K/16K、0.05/0.1/0.2/0.4 RPS，每组各三轮冷
生命周期。八个 cell 全部通过，768 个实测组别请求均正确、完整。16K cell 的最小
物理 KV 缩减为 50%，总吞吐中位数相对 B0 为 +0.71% 至 +16.36%；8K 未跨压缩
阈值，性能近似中性。已完成的 A3 矩阵总吞吐中位数提升 13.17%、最小物理 KV 缩减
73.33%，但 B0 每轮只完成 23 个请求而非至少 24 个，且六窗口吞吐 CV 在 B0/B1
分别为 12.86%/8.94%，均超过 5%，因此 A3 结论为失败。这些仍是同宿主
commissioning 结果，不等同于 V4.6 官方 B0。

另一次公开长上下文测试使用 Qwen2.5-14B-Instruct 和冻结采样契约，覆盖
LongBench-v2（116 条）、LongBench `passage_retrieval_en`（200 条）和 LongBench
`qasper`（93 条）的全部可接纳、未截断样本。8K 物理预算下，三项质量分别变化
−0.86 pp、0.00 pp、−0.24 pp，均通过 1 pp 门槛；请求吞吐分别变化 +7.78%、
+2.41%、+1.35%。B1 记录 127/127 次 scheduler/worker 提交确认，实际压缩样本
物理 block 合计减少 60.67%。补充的 Qwen3.5-35B-A3B TP=2 测试也通过全部质量与
事务门槛，但性能没有优于 B0。完整结果和单轮限制见
[公开 benchmark 记录](docs/public-long-context-benchmarks.zh.md)。

补充的 Qwen3.5 移植 A2/A3 矩阵作为负结果原样保留：792 个实测组别请求全部正确，
TP=2 事务精确匹配，但 A2 八个 cell 只有两个满足 1% 吞吐回退预算；A3 每轮只完成
4 个而不是 24 个请求，六窗口吞吐 CV 为 100%。这组 BF16/eager 证据不改变
Qwen2.5-14B-Instruct 作为标准发布模型的定位。

## 冲突矩阵

| 功能 | 0.8 工作树状态 | 行为 |
| --- | --- | --- |
| Prefix cache | SWE C4 配对协议有效 | Qwen3.5/TP=2/align/图模式确有服务端命中，但吞吐退化 |
| Speculative decoding | 实验性 | 显式 MTP2 完成 SWE 两侧 900 秒窗口且有真实接受率；默认启动保护仍生效 |
| KV transfer / 分离式 P/D | 冲突 | 启动拒绝 |
| 量化 KV | 冲突 | 启动拒绝，仅支持稠密 BF16/FP16 |
| Qwen3.5 全注意力 + Gated-DeltaNet 混合架构 | 实验支持 | `mamba_cache_mode=none` 或 `align`；只压缩全注意力 KV |
| 其他 hybrid / MLA / sliding / local attention | 冲突 | 启动拒绝 |
| 异步调度 | SWE C4 配对协议有效 | Qwen3.5/TP=2/APC/align/图模式完成两侧 900 秒；性能未通过 |
| TP > 1 | 条件支持 | 必须整除 KV 头数；各 rank 同步评分 |
| PP/DP/DCP/PCP > 1 | 冲突 | 启动拒绝 |
| BidKV 或其他调度器 | 冲突 | 要求标准 v1 调度器；拒绝实际启用的平衡调度 |
| 已移除的 Prefix Router、KV Tiering、KNorm、PyramidKV Ascend、SliceGPT | 未集成 | 不导入、不假设原宿主代码存在 |
| 其他通用插件 | 未验证 | 使用显式 allowlist 并独立验证组合 |

## 配置与验收

默认示例采用 8192-token 预算、1024-token 重算窗口、V3 的 128/512-token
前缀/最近保护、8 个位置分段、8192-token 评分分块、每八层评分一次，并在请求
输出少于 64 token 时绕过压缩。Qwen3.5 示例会评分全部 10 个全注意力层。KV
相关 token 数必须为 block size 128 的正整数倍。若 Qwen3.5 的全注意力页被提升为
2,048 token，`kv_budget` 还必须能被 2,048 整除；不要求被 scheduler 的
32,768-token 跨组对齐整除。

- [文档导航（当前与历史）](docs/README.zh.md)
- [2026-09-20 历史工程验证记录](docs/validation.zh.md)
- [历史 V4.6 派生测试要求](docs/kv-compress-test-requirements.zh.md)
- [历史 A2/A3 规程](docs/benchmarking.zh.md)
- [历史公开长上下文 Benchmark](docs/public-long-context-benchmarks.zh.md)
- [HTML 测试榜单（Frontier 实测点与历史证据）](docs/benchmark-leaderboard.html)
- [Qwen3.5-35B-A3B 适配](docs/qwen3.5-35b-a3b-adaptation.zh.md)
- [当前 0.8 前沿测试协议与工程证据](docs/frontier-benchmarking.zh.md)
- [打包与发布](docs/packaging-and-release.zh.md)
- [方法扩展 API](docs/methods.zh.md)
- [校准产物](docs/calibration-artifacts.zh.md)

```bash
VLLM_PLUGINS='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -q
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
```

NPU 候选版本还必须在声明的精确宿主栈上通过数值 smoke 以及长上下文质量、性能与
HBM 矩阵。
