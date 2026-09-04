# 0.3 版本验收规程

[English](benchmarking.md) | 简体中文

本规程适用于当前独立插件。旧 fork 专用的
`--kv-cache-compression-config` 参数已不存在，不能继续使用。baseline 与压缩组
必须保持宿主提交、模型、校准产物、prompt、请求顺序、预热、设备和环境一致。

## 1. 固定溯源

每次运行前记录：

- vLLM-HUST、vLLM-Ascend-HUST、Extension Manager、插件、Triton-Ascend、
  PyTorch、torch-npu、CANN、驱动和固件版本；
- 模型路径/ID 与不可变 revision、tokenizer revision、dtype、RoPE 配置和模型
  文件 hash；
- 校准产物路径、SHA-256、生成器 revision、输入来源和 metadata；
- NPU 型号/编号、可用 HBM、功耗/频率模式和其他进程；
- 插件 JSON、完整启动参数、benchmark 命令、prompt 集 hash 和原始输出目录。

无法重建溯源的行不能公开成性能结论。

## 2. 验证包生命周期

在干净虚拟环境安装 wheel 和管理器，依次执行 validate、configure、enable、
status、disable、forget 和 pip uninstall。还要确认：禁用时导入无副作用；
enable/disable 在进程重启后生效；卸载后 Manager 不再发现该扩展。命令见主
[README](../README.zh.md#安装与管理)。

## 3. NPU kernel 数值验收

用 PyTorch reference 对直接分页评分、融合聚合、选择及可能重叠的 K/V 物化做
对照。覆盖空/短尾部、非二次幂长度、全部支持 dtype、重复压缩以及多种层/head
形状，并记录最大绝对和相对误差，而不仅是 pass/fail。

```bash
python tests/run_npu_kernel_smoke.py
python tests/run_npu_kernel_benchmark.py
```

## 4. 服务正确性与质量

测试低于、等于和高于首次/重复压缩阈值的序列，检查：无崩溃、非法 slot、block
泄漏或跨请求污染；多次事务后语义位置仍单调；下一调度屏障释放 block，请求结束
或取消后全部归还；确定性配置输出稳定；预先约定的长上下文质量套件不越界。

记录任务名、样本数、seed、评分代码 revision、baseline/压缩分数及允许差值。
单个 smoke prompt 不能替代质量验收。

## 5. 匹配性能矩阵

至少选择三个必定触发压缩的输入长度/并发单元，例如模型支持时使用 8K/c=1、
32K/c=4、64K/c=8。每格至少一次预热、三次测量，并交替 baseline/压缩顺序。

报告中位数和范围：输入/输出/总 token 吞吐；TTFT 和 TPOT p50/p90/p99；端到端
时延 p50/p90/p99；峰值/稳态 HBM 和 cache block；压缩次数与压缩耗时分位数；
OOM、拒绝和失败数。

baseline 通过 Manager 禁用插件并重启宿主；压缩组使用同一宿主命令：

```bash
export VLLM_PLUGINS=ascend,ascend_kvcompress
vllm-hust-ext run -- vllm serve /path/to/model \
  --block-size 128 --no-enable-prefix-caching --no-async-scheduling
```

不能与 prefix cache、speculative decoding、KV transfer、量化 KV、BidKV 或已
删除的宿主优化混合测量；这些是未支持组合，不是独立调优变量。

## 6. 发布门槛

生命周期和 CPU suite 通过后，可以声明 Manager/包兼容；只有在声明快照上通过
kernel 与服务正确性后，才能声明 NPU 支持；只有发布匹配性能矩阵和完整溯源后，
才能声明吞吐/HBM 提升。历史 0.2 数据不能替代本门槛。
