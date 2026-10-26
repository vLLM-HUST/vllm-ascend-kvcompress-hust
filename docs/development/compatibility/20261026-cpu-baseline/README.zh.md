# 候选基线：2026-10-26 CPU 准备阶段

[English](README.md) | 简体中文

独立候选环境已完成构建和安装。插件最终测试为 **223 项通过、2 项失败、1 项未执行**。
两项失败停在没有分配 NPU 时的设备型号查询阶段，仍须保留为失败并上卡复测。
本记录不能证明服务兼容，也不扩大插件已验证的兼容版本范围。

## 固定版本组合

| 组件 | 源码 / 运行版本 |
| --- | --- |
| vLLM-HUST core | `ebfcfba6507501c3a359a75eb4911ecc442bc2ff` |
| vLLM-Ascend-HUST | `7c8ec865a12f7b6a14d53b61e7f61b5e811dd483`，附下述依赖补丁 |
| Triton-Ascend-HUST | `b90ab001238f8dde175aebfea23de36bd2cf06a6` |
| KVCompress | `3ddff1c2afee1eccc3d1b65def61167752298f5f`，包版本 `0.10.0` |
| Extension Manager | `36cd420b28c0569d94d5205b9e9e4cc1c8bc0e31`，包版本 `0.3.0` |
| 平台 | AArch64、Python `3.12.13`、CANN / ATB `9.1.0` |
| PyTorch | `torch==2.13.0+cpu`、`torch-npu==2.13.0rc1` |
| Transformers / Hub | `5.17.0` / `1.33.0` |
| 尚未执行的目标配置 | Qwen3.5-35B-A3B、BF16、两张 910B2、TP=2 |

[manifest.json](manifest.json) 记录安装版本、构建退出码、wheel 身份、未完成检查和
SHA256。这是经过适配的候选组合，并非历史性能环境的逐包复现；Python 3.12.13
也不同于插件 README 中已验证的 Python 3.11.16 快照。

## 依赖适配

固定 core 要求 `transformers>=5.16.1,<5.19.0`，原 Ascend 的运行和构建声明均固定为
`5.14.1`，两者无交集。[依赖补丁](patches/ascend-transformers-5.17.patch) 只在独立
Ascend 副本中把两处声明改为 `5.17.0`，没有修改插件算法。Ascend wheel 的
`.d20261026` 后缀反映构建工作树存在修改。

Triton 构建过程会自动应用上游自带补丁，修改 17 个受版本控制的文件。补丁哈希已记入
manifest，这不是手工算子适配。core 使用 `VLLM_TARGET_DEVICE=empty`，Ascend 显式
指定 `SOC_VERSION=ascend910b2`，Triton 构建包含 Ascend 后端。已安装 wheel 的身份见
[wheels-installed.json](evidence/wheels-installed.json)。

## 已完成检查及边界

| 检查 | 证据及解释 |
| --- | --- |
| Python 依赖 | [200 个已安装包检查通过](evidence/dependency-check.log)，没有声明层面的依赖冲突 |
| Transformers 接口 | [38 处导入检查](evidence/transformers-api-check.json)及 Qwen3.5 Dense/MoE 配置构造通过；未执行设备前向 |
| 模型输入 | [离线配置、分词器及 14 个权重分片存在性检查](evidence/offline-model-check.json)通过；未加载权重或校验全部权重字节 |
| core 输入路径 | [样例 token ID 和聊天模板输出](evidence/core-model-inputs-check.json)与 Transformers 一致 |
| 宿主源码接口 | [AST/文本检查](evidence/source-contract-check.json)：预期缓存辅助函数和 worker 方法存在；所选 Triton 源码不生成 `enable-optimized-metaop` 参数 |
| 原生包 | [Triton Ascend 绑定导入、编译器版本及三个库的链接检查](evidence/native-package-check.json)通过；不等于 Ascend 原生初始化或算子执行通过 |
| Manager 生命周期 | 使用独立测试配置完成[验证](evidence/manager-validate.log)、[配置](evidence/manager-configure.log)、[启用](evidence/manager-enable.log)、[启用状态查询](evidence/manager-enabled-status.log)、[禁用](evidence/manager-disable.log)和[禁用状态查询](evidence/manager-disabled-status.log) |
| 插件测试 | [最终原始日志](evidence/plugin-cpu-final-tests.log)、[JUnit XML](evidence/plugin-cpu-final-tests.xml)：223 通过、2 失败、1 未执行 |

链接检查只覆盖 `libvllm_ascend_kernels.so`、`libtriton.so` 和 `libcust_opapi.so`。
Manager 启用不会启动模型，也不能证明运行兼容。[包版本清单](evidence/installed-package-freeze.log)
是环境记录，不是可跨机器直接执行的安装命令，本地构建的 wheel 须另行提供。
原始文件中的绝对路径用于标识当时运行位置。

## 尚未完成的测试

`test_grouped_topk_router_scaling_compatibility` 和
`test_runner_generates_calibration_before_loading_model` 的导入路径依次经过
Ascend MoE、core FLA、Triton `get_current_target/get_arch`，在查询 SoC 时出现
`cann_get_soc_name failed` 和 `get_arch returned NULL without setting an exception`。
失败发生在目标断言之前；上卡后仍可能暴露其他不兼容，不能将这两项计为通过。

`test_ascend_v1_worker_seams_match_current_host` 因需要加载原生 worker 运行时而
主动排除。最终测试使用仓库原有测试，未修改断言：

```bash
export VLLM_HUST_BENCHMARK_ORIGINALS=/absolute/path/to/verified-originals
VLLM_PLUGINS='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  python -m pytest -q \
  -k 'not test_ascend_v1_worker_seams_match_current_host' \
  --junitxml=/absolute/path/to/plugin-cpu-final-tests.xml
```

执行前须配好上述宿主环境并取得[已校验的原始测试输入](../../../usage/benchmark-originals.zh.md)。
准备期间曾出现 216 通过、6 失败：5 项缺少原始测试输入，1 项发生在 Ascend 安装前。
最终测试已替代该准备阶段结果；早期日志和 XML 的哈希仍记录在 manifest 中。
没有将失败结果改写为通过。

## 上卡验收

分配 NPU 后，按[环境指南](../../environment-installation.zh.md)检查原生初始化和
BF16 运算，复测上述三项，验证 TP2/HCCL、插件 kernel smoke 及 Qwen3.5 GDN
output smoke。随后在同一环境依次完成插件关闭和启用的 Qwen3.5 BF16/TP2 实际请求；
启用侧必须触发真实压缩，不能只证明插件成功加载。通过后再进行 B0/B1 或 TTFT 测量。

本阶段没有运行 NPU kernel、模型服务、B0/B1 或 TTFT 测量。公开证据按原始字节选取，
见[校验和](SHA256SUMS)。含完整构建日志与早期尝试的本地归档通过 manifest 中的
SHA256 标识，本次不重复上传。wheel、权重、虚拟环境、构建缓存、SSH 配置和已归档
的历史 benchmark 输入均未包含在本提交中。
