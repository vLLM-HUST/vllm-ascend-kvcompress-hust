# vLLM Ascend KV Compression

[English](README.md) | 简体中文

面向与上游对齐的 vLLM-HUST、vLLM-Ascend-HUST 的独立 TriAttention KV-cache
压缩插件。0.7 版加入论文 V3 位置策略与实验性 Qwen3.5 支持；0.8 版适配同步后
的宿主，且不修改宿主源码仓库。0.9 版加入供独立压缩方法使用的 method API v1，
包括逐层物理状态与 Query 观测契约。0.10 版合并 TP 评分同步；0.11 版加入
私有目标 K/V 直拷及非整齐 V3 分段批量选择。Qwen3.5 示例采用
16,128-token 预算和 2,048-token 请求输出门槛。

> 0.11.0 源码工程验证：一组匹配的 Qwen3.5 BF16/TP=2 SWE C4／900 秒
> 配对，在 16,128-token 预算下，每卡输出吞吐较 B0 高 **0.17%**，逐请求
> 解码 P90 高 **0.13%**，TTFT P95 变差 **14.27%**。配置上下文为
> 262,144 token，实际最长 prompt 为 52,505 token。两项微小正差值低于
> 跨次运行波动，不能证明可重复性、SWE 答案质量、生产适用性或官网榜单成绩。
> MTP2 仍为默认拒绝启动的实验功能。精确的宿主、负载和负结果见
> [结果范围](docs/results/README.zh.md)、[多轮证据](docs/results/evidence/kvcompress-qwen35-20261023-v011-swe-campaign.json)
> 和[优化记录](docs/optimization/triattention-private-copy-and-v3-selection-20261023.zh.md)。

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
本仓库许可证进行再分发。详见 [NOTICE](NOTICE)、[归属与许可](docs/development/ownership-and-licensing.zh.md)
和[校准产物说明](docs/development/calibration-artifacts.zh.md)。

## 兼容范围

| 组件 | 支持版本 | 已验证快照 |
| --- | --- | --- |
| vLLM-HUST / `vllm` | `>=0.29.1.post1.dev0,<0.30` | `ebfcfba650`（`0.29.1.post1.dev456+gebfcfba65.empty`） |
| vLLM-Ascend-HUST / `vllm-ascend` | `>=0.25.1rc2.dev0,<0.26` | `7c8ec865a1`（`0.25.1rc2.dev774+hust.20260903.4.g7c8ec865a`） |
| Extension Manager | `>=0.3,<0.4` | Manifest 0.3 契约 |
| Python | `>=3.10,<3.15` | 3.11.16 |

标准验收拓扑仍为单 Ascend NPU、v1 调度器与 `NPUModelRunner`、单一全注意力
KV 组、block size 128、稠密 BF16/FP16 K/V。Qwen3.5 text/MoE 混合模型额外支持
一个全注意力组加 Gated-DeltaNet 状态组，支持 `mamba_cache_mode=none` 或
`align`；只有 TP
大小可整除 KV 头数时才允许张量并行（Qwen3.5-35B-A3B 为 TP=2）。已验证的
Ascend 宿主使用由布局决定的跨组 scheduler 对齐、1,152 或 2,048-token
全注意力页，以及 128-token attention 内核缓存块；插件会校验实际报告的粒度，
并把每个逻辑页展开为 9 或 16 个内核块。其他组合在启动阶段拒绝。manifest 使用稳定的
`vllm.general_plugins` 发现入口；当前宿主没有冻结的原生 KV 生命周期 API，因此
通过窄版本范围与契约测试保护内部适配点。

## 独立安装、启用、禁用和卸载

分步骤部署请阅读[用户指南](docs/usage/README.zh.md)：安装、配置、校准、
服务验证与故障排查。

先安装成套锁定的宿主栈，再安装发布包。插件 wheel 特意不把 `vllm` 和
`vllm-ascend` 声明为 Python 包依赖：这些硬件相关包必须作为一套经过验证的运行时
统一部署；宿主兼容范围由 Extension Manager 清单和启动检查约束。这样安装插件时
也不会让 pip 重新解析已经部署好的整套宿主环境。

当前 vLLM 0.29 / vLLM-Ascend 0.25 开发版本线不能混入 Triton-Ascend 3.2.2。
旧 Triton wheel 锁定 NumPy 1.26.4，而当前 vLLM 要求
`opencv-python-headless>=4.13`，其可用 wheel 要求 NumPy 2；把 OpenCV 降到
4.9 又会违反 vLLM 的依赖约束。应使用宿主栈配套的 Triton-Ascend 3.6。

先部署匹配的宿主栈，将 `KVCOMPRESS_PLUGIN_VERSION` 设为与宿主兼容的
已发布插件版本，再安装且不重新解析硬件相关依赖：

```bash
python -m pip install --no-deps "vllm-ascend-kvcompress-hust==${KVCOMPRESS_PLUGIN_VERSION}"
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
[环境指南](docs/development/environment-installation.zh.md)，隔离 wheel 流程见
[打包与发布](docs/development/packaging-and-release.zh.md)。

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
[Qwen3.5 适配说明](docs/design/qwen3.5-35b-a3b-adaptation.zh.md)。较早宿主快照的
GDN 自定义算子仍使用旧式 `int[]` 元数据 ABI，需 `--enforce-eager` 与显式
`VLLM_ASCEND_KVCOMPRESS_QWEN_GDN_LIST_COMPAT=1` 兼容桥；同步后的宿主已暴露
Tensor ABI，且无需 `--enforce-eager` 即完成 FULL_AND_PIECEWISE 图捕获。
兼容桥仅在精确旧 schema 上生效，不修改任一宿主仓库。

## 运行时与证据

TriAttention 保留校准评分、V3 的前缀/最近窗口保护和中间分段比例配额。
Qwen3.5 只评分十个全注意力层；Gated-DeltaNet 状态仍由宿主管理。
scheduler 与 worker 在步骤边界提交压缩事务，保持语义 RoPE 位置，同时缩短
物理 KV 长度。详见[方法设计](docs/design/methods.zh.md)和
[Qwen3.5 适配](docs/design/qwen3.5-35b-a3b-adaptation.zh.md)。

2026-10-17 的 SWE C4／900 秒配对中，每卡输出吞吐提升 1.33%，逐用户解码
P90 提升 1.62%，TTFT P95 变差 8.65%。0.9 的 SWE 配对中解码 P90
提升 1.05%，但每卡吞吐下降 2.71%；AgentX 是另一 cohort。详见
[结果范围](docs/results/README.zh.md)、
[完整 0.9 过程](docs/archive/results/frontier-benchmarking-0.9.zh.md)和
[测试规程索引](docs/benchmark/README.zh.md)。历史 Qwen2.5
A2/A3 与公开长上下文结果保留在[归档](docs/archive/results/public-long-context-benchmarks.zh.md)。

## 冲突矩阵

| 功能 | 实现状态 | 行为 |
| --- | --- | --- |
| Prefix cache | 本地 C4 配对协议有效 | 私有目标块保留共享源块哈希；0.11 宽预算单组微小双指标正差值伴随 TTFT 退化 |
| Speculative decoding | 实验性 | 显式 MTP2 完成 SWE 两侧 900 秒窗口且有真实接受率；默认启动保护仍生效 |
| KV transfer / 分离式 P/D | 冲突 | 启动拒绝 |
| 量化 KV | 冲突 | 启动拒绝，仅支持稠密 BF16/FP16 |
| Qwen3.5 全注意力 + Gated-DeltaNet 混合架构 | 实验支持 | `mamba_cache_mode=none` 或 `align`；仅全注意力 KV 使用压缩后的物理长度，Gated-DeltaNet 保持语义长度 |
| 其他 hybrid / MLA / sliding / local attention | 冲突 | 启动拒绝 |
| 异步调度 | 合并后官方 C4 配对协议有效 | 匹配模型输出后才提交；已排队批次结束前保留源块 |
| TP > 1 | 条件支持 | 必须整除 KV 头数；各 rank 同步评分 |
| PP/DP/DCP/PCP > 1 | 冲突 | 启动拒绝 |
| BidKV 或其他调度器 | 冲突 | 要求标准 v1 调度器；拒绝实际启用的平衡调度 |
| 已移除的 Prefix Router、KV Tiering、KNorm、PyramidKV Ascend、SliceGPT | 未集成 | 不导入、不假设原宿主代码存在 |
| 其他通用插件 | 未验证 | 使用显式 allowlist 并独立验证组合 |

## 配置与验收

通用示例采用 8192-token 预算、1024-token 重算窗口、V3 的 128/512-token
前缀/最近保护、8 个位置分段、8192-token 评分分块、每八层评分一次，并在请求
输出少于 64 token 时绕过压缩。Qwen3.5 示例以 stride 2 采样十个
全注意力层，采用 16,128-token 预算、1,024-token 最近保护和 2,048-token
请求输出门槛。KV
相关 token 数必须为 block size 128 的正整数倍。若 Qwen3.5 的全注意力页被提升为
1,152 或 2,048 token，`kv_budget` 还必须能被当前注意力页大小整除；不要求
能被 scheduler 的跨组对齐粒度整除。

- [文档导航（当前与历史）](docs/README.zh.md)
- [2026-09-20 历史工程验证记录](docs/archive/results/validation.zh.md)
- [历史 V4.6 派生测试要求](docs/archive/benchmark/kv-compress-test-requirements.zh.md)
- [历史 A2/A3 规程](docs/archive/benchmark/benchmarking.zh.md)
- [历史公开长上下文 Benchmark](docs/archive/results/public-long-context-benchmarks.zh.md)
- [HTML 测试榜单（Frontier 实测点与历史证据）](docs/results/benchmark-leaderboard.html)
- [Qwen3.5-35B-A3B 适配](docs/design/qwen3.5-35b-a3b-adaptation.zh.md)
- [测试规程](docs/benchmark/README.zh.md)
- [打包与发布](docs/development/packaging-and-release.zh.md)
- [优化记录](docs/optimization/README.zh.md)
- [方法扩展 API](docs/design/methods.zh.md)
- [校准产物](docs/development/calibration-artifacts.zh.md)

```bash
VLLM_PLUGINS='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -q
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
```

NPU 候选版本还必须在声明的精确宿主栈上通过数值 smoke 以及长上下文质量、性能与
HBM 矩阵。
