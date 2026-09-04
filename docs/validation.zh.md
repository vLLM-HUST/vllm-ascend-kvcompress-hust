# 0.3 版本验收记录

[English](validation.md) | 简体中文

日期：2026-09-04

## 声明快照

| 组件 | Revision/版本 |
| --- | --- |
| vLLM-HUST | `5b343ed52`，`0.17.2rc1.dev5941+g5b343ed52.empty` |
| vLLM-Ascend-HUST | `4e57439`，`0.25.1rc1+hust.20260903.4` |
| Extension Manager | `9fb467e`，`0.2.0.dev0` |
| vLLM Ascend KV Compression | 基于 `e4a4864` 的工作树，包版本 `0.3.0` |
| Triton-Ascend 源码 | `ee4b0ecef`，要求的包版本线 `3.6.0` |
| Python | 3.11.16 |

宿主、管理器、文档和 Triton 仓库仅用于读取或重新安装，没有修改其中任何源码；
全部源码和文档改动都限制在本插件仓库。

## 包与静态验收

| 范围 | 结果 | 证据 |
| --- | --- | --- |
| CPU/unit suite | PASS | 52 项，覆盖配置、校准、选择、provider/scheduler 事务及抢占重置、精确 scheduler 准入、runner 延迟挂接、归一化 cache plan 绑定、版本一致性、manifest 和 registry |
| Lint/format | PASS | `src`/`tests` Ruff check 与 format check、`git diff --check` 均通过 |
| 宿主接口核对 | PASS | 已按声明快照核对当前 `Scheduler`、未开启 balance 的 Ascend `BalanceScheduler`、`KVCacheManager`、`BlockTable` 和 `NPUModelRunner` 挂接点 |
| Manifest schema/发现 | PASS | `0.2-experimental`；active carrier、ID/版本、activation、provider、权限和入口无需导入 runtime 即可发现 |
| 已安装宿主管理器检查 | PASS | `vllm-ascend 0.25.1rc1+hust.20260903.4` 满足声明范围，启动配置可渲染 |
| Manager 生命周期 | PASS | 隔离环境中 configure、enable、status/env、disable、forget 和 uninstall 成功 |
| 禁用行为 | PASS | 未被直接或 Manager 显式启用时，注册无副作用 |
| Worker 延迟导入 | PASS | 控制进程可挂接 scheduler/cache seam，不提前导入 NPU runner |

已测试候选 wheel `vllm_ascend_kvcompress_hust-0.3.0-py3-none-any.whl`，配套
sdist 已通过元数据和内容检查；上传前须从受保护 tag 构建再次记录两者哈希。
不含宿主的隔离环境会将兼容性标记为 unverified，并拒绝对受信任进程内扩展执行
`run --dry-run`；这是预期的失败关闭行为。

## 环境恢复结论

开发环境最初残留旧版 vLLM-Ascend distribution，并同时存在官方 Triton 与
editable Triton-Ascend，导致平台重复发现和过期 `triton._C.libtriton` 二进制。
现已整理包状态并重装当前 vLLM-Ascend，未修改对应源码 checkout；补装缺少的
`numba` 和 `torchvision` 后，当前服务可以启动。

精确依赖验收仍未完成。当前 `vllm-ascend` 元数据要求 Torch 2.13.0、
Torch-NPU 2.13.0rc1、Triton-Ascend 3.6.0 和 Transformers 5.14.1；可用开发
环境为 Torch/Torch-NPU 2.10、已发布的 Triton-Ascend 3.2.2 与 Transformers
5.15.1。未修改的 Triton-Ascend 3.6.0 源码构建需下载约 1.2 GB Ascend LLVM，
本次窗口内未完成。因此下述 NPU 数据是回退环境的兼容性证据，不等于精确依赖栈
的正式验收。

## NPU kernel 验收

设备为 Ascend 910B2、物理设备 3。slot 偏移、重叠安全 K/V 物化、直接分页评分
对比 PyTorch reference、融合归一化/query-head 最大值/跨层聚合对比 PyTorch
reference 的数值 smoke 均通过。

| 热路径 | 通用/reference | 优化后 | 加速比 |
| --- | ---: | ---: | ---: |
| K/V 分页 copy | 0.564 ms | 0.332 ms | 1.70x |
| 直接分页评分 | 1.239 ms | 0.501 ms | 2.47x |
| 融合聚合 | 0.155 ms | 0.149 ms | 1.04x |
| position/slot offset wrapper | 0.043 ms | 0.053 ms | 0.82x |

较慢的 offset wrapper 不会取代通用路径。以上是孤立 kernel 计时，不是请求级
吞吐声明。

同时对完整 48 层压缩事务检查了 JIT 特化。修改前，每遇到新请求长度会产生一次
约 5--6 秒编译；把轮次/长度设为动态参数后，新进程只需一次冷编译即可覆盖
2,176 与 6,311 token 形状，之后交替事务均为 22--26 ms，不再按长度编译。

## 当前宿主服务验收

通过 Manager 包装成功启动 Qwen2.5-Coder-14B-Instruct 服务，使用 FP16、block
size 128、最大长度 12,288、关闭 prefix cache、eager 模式及示例中的 2,048-token
物理预算。6,311-token prompt 加 300 个强制输出 token 完成并触发重复压缩。
单请求生成阶段服务日志的 KV-cache 占用约 5%，配对 baseline 约 15%。

### 有限配对性能单元

下表保持模型、prompt 文本、生成配置、设备、宿主参数和四路并发一致；微小的
prompt token 计数差异原样披露，不进行归一化。

| 模式 | Prompt + 输出 token | 耗时 | 输出 tok/s | 总 tok/s | 生成稳态 KV 占用 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 插件 | 25,244 + 1,200 | 29.461 s | 40.731 | 897.578 | 约 20% |
| Baseline | 25,232 + 1,200 | 28.634 s | 41.908 | 923.086 | 约 60% |

该单元中插件总吞吐比 baseline 低 2.8%，不能宣称端到端加速。单路 6,311-token
prompt 加 300-token 输出中，两次优化后插件平均 25.709 秒；两次 baseline 平均
25.216 秒（插件慢 2.0%）。优化前曾观测到一次插件耗时 35.972 秒，说明 JIT
修改消除了较大插件开销，但这并不是统计完整的前后对照矩阵。

插件与 baseline 的静态 KV 分配基本相同（7.98 GiB 与 7.97 GiB），因为宿主在
启动时预留 cache pool。请求期 cache 使用率下降是 block pressure 证据，不是
峰值 HBM 声明。本轮未采集 TTFT/TPOT 分位数、压缩耗时分位数、峰值/稳态 HBM
以及完整多单元重复矩阵。

### 质量门槛

使用含 11,569 个输入 token 的 needle retrieval prompt：baseline 返回完整目标
`BLUE-HERON-4729`，插件采用示例的 2,048-token budget 时只返回 `BLUE-`；
61-token 短控制 prompt 下插件可返回完整目标。因此默认示例配置的长上下文质量
门槛为 **FAIL**。发布或生产使用前必须验证模型匹配的校准数据和/或更大 budget。

| 必须补齐的行 | 状态 |
| --- | --- |
| score/aggregate/copy 数值 smoke | PASS——910B2 回退环境 |
| Manager 包装启动及重复压缩 smoke | PASS——回退环境 |
| 长上下文质量门槛 | **FAIL**——默认 2,048-token budget |
| 匹配吞吐/时延 | PARTIAL——一个单请求和一个 c=4 单元；未超过 baseline |
| Cache block pressure | PASS——服务日志证据，c=4 约 20% 对 60% |
| TTFT/TPOT、峰值/稳态 HBM、完整矩阵 | NOT RUN |
| 当前精确依赖栈 | BLOCKED——无 Triton-Ascend 3.6 wheel，源码依赖下载未完成 |

## 发布判断

仓库可作为实验性源码插件进入 Extension Manager 集成评审，但按 BidKV 发布指南，
目前**不具备发布 PyPI alpha 或形成生产 NPU/质量/性能声明的条件**：默认长上下文
质量门槛失败，精确依赖栈未验收，完整性能/HBM 矩阵也未完成。不得把 0.2 历史
结果表述为 0.3 验收数据。
