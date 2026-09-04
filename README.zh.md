# vLLM Ascend KV Cache Compression

[English](README.md) | 简体中文

这是面向已与上游对齐的 vLLM-HUST、vLLM-Ascend-HUST 的独立
TriAttention KV-cache 压缩插件。0.3 版本不再依赖已删除的 fork 专有压缩
生命周期，也不会修改两个宿主仓库。

> 当前状态：实验性。声明源码快照上的打包与 Extension Manager 生命周期、
> Ascend 910B2 kernel smoke、Manager 包装的服务启动和重复压缩均已通过。
> 默认 2048-token budget 未通过已记录的长上下文质量门槛；精确依赖栈以及完整
> 性能/HBM 矩阵仍是发布门槛，详见[验收记录](docs/validation.zh.md)。

## 归属与维护

- 学校：华中科技大学（HUST）
- 课题组：CGCL
- 指导教师：万瑶教授
- 主要负责人：刘思辰（[@Seas0](https://github.com/Seas0)）
- 当前维护者：张家万（[@Jiawan23](https://github.com/Jiawan23)）、
  韦若皓（[@kotoriqaq0](https://github.com/kotoriqaq0)）、刘思辰
  （[@Seas0](https://github.com/Seas0)）

团队同意持续维护 vLLM-HUST、vLLM-Ascend-HUST 的版本兼容性，并同意接入
vLLM-HUST Extension Manager。本仓库是由 CGCL 维护的独立插件，不是直接
内置于上游对齐宿主仓库中的代码。

## 算法来源与许可证

评分方法改编自 [TriAttention](https://github.com/WeianMao/triattention) 的
[`a4bc3c8f709db60f016ef42c3feb290fd0c00c1b`](https://github.com/WeianMao/triattention/tree/a4bc3c8f709db60f016ef42c3feb290fd0c00c1b)
快照，对应论文 [*TriAttention: Efficient Long Reasoning with Trigonometric
KV Compression*](https://arxiv.org/abs/2604.04921)。本实现针对 Ascend 分页
K/V 存储重新实现，并未复制上游 CUDA runtime kernel。详见 [NOTICE](NOTICE)
和[适配说明](docs/ascend-adaptation-vs-triattention-vllm.zh.md)。

仓库代码和已提交文档采用 Apache-2.0。已提交的校准统计是模型派生的聚合
产物，生成来源信息并不完整；不能仅凭本仓库许可证推定原始模型或数据集可
再分发。原始 benchmark 输入、模型权重、服务日志和未公开数据集不属于可
分发安装包。范围说明见[归属与许可](docs/ownership-and-licensing.zh.md)。

## 兼容范围

| 组件 | 支持版本线 | 已核对快照 |
| --- | --- | --- |
| vLLM-HUST / `vllm` | `>=0.17.2rc1.dev0,<0.18` | `5b343ed52`（`0.17.2rc1.dev5941+g5b343ed52.empty`） |
| vLLM-Ascend-HUST / `vllm-ascend` | `>=0.25.1rc1,<0.26` | `4e57439`（`0.25.1rc1+hust.20260903.4`） |
| Extension Manager | `>=0.2.0.dev0,<0.3` | `9fb467e` |
| Python | `>=3.10,<3.15` | 3.11.16 |

当前仅支持单张 Ascend NPU、标准 v1 Scheduler（包括当前 Ascend 内置但未启用
balance 功能的 `BalanceScheduler` 包装类）和 `NPUModelRunner`、单个普通
full-attention KV group、block size 128，以及 BF16/FP16 稠密 K/V。不支持的
组合会在启动阶段失败并给出原因。

## 安装与管理

先安装当前宿主栈，再安装插件和 Extension Manager。源码安装方式：

```bash
python -m pip install ./extension-manager
python -m pip install ./vllm-ascend-kvcompress-hust
```

发布到 PyPI 后，对应的插件安装命令为：

```bash
python -m pip install 'vllm-ascend-kvcompress-hust[manager]==0.3.0'
```

复制 [examples/triattention.json](examples/triattention.json)，把 `stats_path`
替换为与目标模型版本严格匹配的统计文件绝对路径，然后配置并启用：

```bash
vllm-hust-ext extension validate org.vllm-hust.ascend-kvcompress
vllm-hust-ext extension configure org.vllm-hust.ascend-kvcompress \
  --file /absolute/path/triattention.json
vllm-hust-ext extension enable org.vllm-hust.ascend-kvcompress
vllm-hust-ext extension status org.vllm-hust.ascend-kvcompress
```

通过管理器启动服务。如果环境已经用 `VLLM_PLUGINS` 作为白名单，必须同时
包含 Ascend 平台和本插件：

```bash
export VLLM_PLUGINS=ascend,ascend_kvcompress
vllm-hust-ext run -- vllm serve /path/to/model \
  --block-size 128 \
  --no-enable-prefix-caching \
  --no-async-scheduling
```

若未设置 `VLLM_PLUGINS`，vLLM 会发现所有已安装的 general plugin；但在
Extension Manager 或直接启用变量没有激活本插件时，注册函数仍保持无操作。

安全禁用流程是先停止宿主进程，再执行下列命令并重启：

```bash
vllm-hust-ext extension disable org.vllm-hust.ascend-kvcompress
```

卸载时先停止所有宿主进程，再执行：

```bash
vllm-hust-ext extension disable org.vllm-hust.ascend-kvcompress
vllm-hust-ext extension forget org.vllm-hust.ascend-kvcompress
python -m pip uninstall vllm-ascend-kvcompress-hust
```

这些操作只改变安装包和 Extension Manager 状态，不修改 vLLM-HUST 或
vLLM-Ascend-HUST。

### 不通过 Extension Manager 的直接启用

仅建议开发调试时使用；配置变量既可接收 JSON 文件路径，也可接收内联 JSON：

```bash
export VLLM_PLUGINS=ascend,ascend_kvcompress
export VLLM_ASCEND_KVCOMPRESS_ENABLED=1
export VLLM_ASCEND_KVCOMPRESS_CONFIG=/absolute/path/triattention.json
vllm serve /path/to/model --block-size 128 --no-enable-prefix-caching
```

取消设置两个 `VLLM_ASCEND_KVCOMPRESS_*` 变量并重启，即可关闭直接启用。

## 当前接入方式

安装包使用公开的 `vllm.general_plugins` entry point 和静态 Extension Manager
manifest。显式启用后，适配层会检查并挂接当前宿主中的以下符号：

- `vllm.v1.core.sched.scheduler.Scheduler`
- `vllm.v1.core.kv_cache_manager.KVCacheManager.allocate_slots`
- `vllm.v1.worker.block_table.BlockTable.compute_slot_mapping`
- `vllm_ascend.worker.model_runner_v1.NPUModelRunner` 的
  `initialize_kv_cache`、`_update_states`、`_build_attention_metadata`、
  `sample_tokens`

NPU runner 挂钩会在 worker 模块真正加载时才注入，API 和管理器进程不会因此
提前导入重型 NPU runner。这些宿主符号仍是未冻结内部接口，所以插件刻意使用
窄版本范围；每次宿主版本线更新都必须先完成验收再扩大范围。

压缩在同步模型步骤之后执行。调度器只在下一次调度屏障释放旧尾部 block；语义
RoPE position 保持不变，物理 slot 和 attention length 使用请求级偏移。由于
压缩后的 block 不再代表可哈希的语义前缀，prefix cache 必须关闭。

## 长上下文优化

0.3 版本保持 block 对齐的固定物理预算，并加入：

- 直接读取 Ascend 分页 K cache 评分，不先物化所有 key；
- 复用全长分数、K/V copy、聚合和 dense index 工作区；
- NPU 上融合归一化、query-head 最大值和跨层聚合；
- 把压缩轮次和请求长度保持为 JIT 动态参数，避免每种长度重新编译评分/聚合；
- 专用 kernel 和通用路径复用同一份 score workspace；
- 只抽样部分 layer 评分，但对每层 K/V 执行物化；
- 设备驻留的语义/物理偏移，以及单次 slot mapping。

这些改动减少临时分配、编译和 kernel launch 压力。在 Ascend 910B2 kernel
微基准中，分页 copy、直接评分和融合聚合路径相对通用 reference 分别达到
1.70x、2.47x 和 1.04x。将轮次/长度改为动态参数后，已观察到的每种长度
5--6 秒重复编译被消除：一次冷编译后，2,176 与 6,311 token 事务交替执行均为
22--26 ms。6,311-token prompt 加 300-token 输出的服务测试中，优化后插件平均
25.71 秒，优化前为 35.97 秒，但仍比配对 baseline 慢 2.0%；四路并发总吞吐为
897.6 tok/s，baseline 为 923.1 tok/s（-2.8%），日志 KV-cache 占用约 20%，
baseline 约 60%。这些是有限验收数据，不是普遍吞吐提升声明，详见
[验收记录](docs/validation.zh.md)。旧生命周期结果仍只保留在
[results](docs/resuts.zh.md)，不能作为 0.3 验收结果。

## 冲突矩阵

| 功能 | 0.3 状态 | 行为 |
| --- | --- | --- |
| Prefix cache | 冲突 | 启动拒绝，必须关闭 |
| Speculative decoding | 冲突 | 启动拒绝 |
| KV transfer / P-D 分离 | 冲突 | 启动拒绝 |
| 量化 KV | 冲突 | 启动拒绝，仅支持 BF16/FP16 稠密 K/V |
| Hybrid、MLA、sliding/local attention | 冲突 | 启动拒绝 |
| Async scheduling | 冲突 | 启动拒绝 |
| TP、PP、DP、DCP、PCP 大于 1 | 冲突 | 启动拒绝 |
| BidKV 或其它 scheduler class | 冲突 | 必须使用上游 v1 `Scheduler` 或当前未启用 balance 功能的 Ascend `BalanceScheduler` 包装类；启用 balance 时拒绝启动 |
| 原 Prefix Router、KV Tiering、KNorm、PyramidKV Ascend、SliceGPT 代码 | 未接入 | 当前宿主已不存在，本插件不导入也不作任何假设 |
| 其它 `vllm.general_plugins` | 未验证 | 使用显式白名单并逐项验证 |

## 配置与校准

示例配置选择 2048-token 物理预算、128-token 重算窗口，并每四层选一层评分。
`kv_budget`、`recompute_window` 和 `score_chunk_size` 必须是 128 的正整数倍。
部署前必须使用模型匹配统计，并阅读[校准产物指南](docs/calibration-artifacts.zh.md)。

## 验收与开发

- [当前验收记录](docs/validation.zh.md)
- [Benchmark 规范](docs/benchmarking.zh.md)
- [打包与发布指南](docs/packaging-and-release.zh.md)
- [方法扩展接口](docs/methods.zh.md)
- [校准产物](docs/calibration-artifacts.zh.md)

CPU 和打包检查：

```bash
VLLM_PLUGINS='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -q
python -m ruff check src tests
```

NPU release candidate 还必须在声明的精确宿主快照上通过 kernel 数值 smoke、
长上下文质量、baseline/压缩配对吞吐与时延，以及 block/HBM 验收。
