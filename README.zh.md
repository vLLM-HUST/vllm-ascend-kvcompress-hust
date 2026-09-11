# vLLM Ascend KV Compression

[English](README.md) | 简体中文

面向与上游对齐的 vLLM-HUST、vLLM-Ascend-HUST 的独立 TriAttention KV-cache
压缩插件。0.4 版本已适配当前宿主与 Extension Manager，且不修改两个宿主仓库。

> 状态：实验性候选版本。插件包生命周期、Ascend 算子 smoke、服务正确性和三轮
> 16K 冷启动工程对照均已通过；V4.6 要求的官方基线、模型匹配校准、质量与稳定性
> 完整矩阵仍是发布门槛。详见[验收记录](docs/validation.zh.md)。

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
| vLLM-HUST / `vllm` | `>=0.28.1.post1.dev0,<0.29` | `6cdc0304a8`（`0.28.1.post1.dev260`） |
| vLLM-Ascend-HUST / `vllm-ascend` | `>=0.25.1rc2.dev0,<0.26` | `5901bedbb7`（`0.25.1rc2.dev232+hust.20260903.4.g5901bedbb`） |
| Extension Manager | `>=0.2.0.dev0,<0.3` | `cf1ea71e3e` |
| Python | `>=3.10,<3.15` | 3.11.16 |

当前仅支持单 Ascend NPU、v1 调度器与 `NPUModelRunner`、单一全注意力 KV 组、
block size 128、稠密 BF16/FP16 K/V。其他组合在启动阶段拒绝。manifest 使用稳定的
`vllm.general_plugins` 发现入口；当前宿主没有冻结的原生 KV 生命周期 API，因此
通过窄版本范围与契约测试保护内部适配点。

## 独立安装、启用、禁用和卸载

先安装宿主栈，再安装发布包：

```bash
python -m pip install 'vllm-ascend-kvcompress-hust[manager]==0.4.0'
```

复制 [examples/triattention.json](examples/triattention.json)，将 `stats_path`
改为与模型及 revision 严格匹配的统计文件，然后执行：

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

## 运行时与长上下文优化

显式启用后，适配器校验并挂接当前调度器、KV-cache manager、Ascend 具体 block
table 和 `NPUModelRunner`。压缩仅在同步模型步之后提交：语义 RoPE 位置保持不变，
物理 attention/slot 索引使用逐请求 offset，旧 block 在下一调度屏障释放。

0.4 使用分页 K 直接评分、持久 workspace、NPU 归一化/head/layer 聚合融合、动态
JIT 长度、分层抽样、设备端 offset，并将物理预算调优为 4096 token。910B2 算子
最终测试中，分页 copy、直接评分和聚合相对通用参考分别为 1.82x、2.81x、1.34x；
offset 更新为 0.83x，不宣称该项优化有效。

Qwen2.5-14B 三轮冷启动工程对照均使用 4 个固定 16,384+1,024 请求、0.4 RPS、
并发 4；24/24 请求均达到预期输出长度。压缩后每请求保留 32/128 block（物理 KV
减少 75%）。总吞吐中位数为 1431.3 对 1129.5 tok/s（+26.7%）；平均 TPOT
39.16 对 52.43 ms（-25.3%）；平均端到端时延 44.37 对 57.81 s（-23.2%）。
这是同宿主兼容性对照的工程证据，不等同于 V4.6 要求的官方 B0 结论。

## 冲突矩阵

| 功能 | 0.4 状态 | 行为 |
| --- | --- | --- |
| Prefix cache | 冲突 | 启动拒绝，必须禁用 |
| Speculative decoding | 冲突 | 启动拒绝 |
| KV transfer / 分离式 P/D | 冲突 | 启动拒绝 |
| 量化 KV | 冲突 | 启动拒绝，仅支持稠密 BF16/FP16 |
| Hybrid/MLA/sliding/local attention | 冲突 | 启动拒绝 |
| 异步调度 | 冲突 | 启动拒绝 |
| TP/PP/DP/DCP/PCP > 1 | 冲突 | 启动拒绝 |
| BidKV 或其他调度器 | 冲突 | 要求标准 v1 调度器；拒绝实际启用的平衡调度 |
| 已移除的 Prefix Router、KV Tiering、KNorm、PyramidKV Ascend、SliceGPT | 未集成 | 不导入、不假设原宿主代码存在 |
| 其他通用插件 | 未验证 | 使用显式 allowlist 并独立验证组合 |

## 配置与验收

示例配置采用 4096-token 预算、1024-token 重算窗口、512-token 最近保护窗口、
8192-token 评分分块和每四层评分一次。相关 token 数必须为 block size 128 的正整数倍。

- [当前验收记录](docs/validation.zh.md)
- [从 V4.6 提取的测试要求](docs/kv-compress-test-requirements.zh.md)
- [验收规程](docs/benchmarking.zh.md)
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
