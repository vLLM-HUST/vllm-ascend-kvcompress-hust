# Ascend 宿主环境安装

[English](environment-installation.md) | 简体中文

本指南适用于受支持的 Ascend 机器。个人路径和重启保留规则应写在被 Git
忽略的 `AGENTS.dev.md` 中（如果存在）。旧宿主的构建诊断保存在
[2026-09-27 历史安装记录](../archive/development/environment-installation-20260927.zh.md)。

## 宿主栈契约

插件 wheel 故意不声明核心 `Requires-Dist`。安装插件前，须配套部署兼容的
vLLM-HUST、vLLM-Ascend、Triton-Ascend、PyTorch、torch-npu、CANN、ATB
和设备驱动；Extension Manager 清单记录受支持宿主范围。原生扩展必须与
选定的 Python ABI 相符，并使用每项[结果](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/results/README.zh.md)记录的
精确受测宿主栈。发行包版本号本身不能证明原生模块或算子可执行。

使用隔离环境，按机器条件选择模型、缓存、工具链和证据位置。若工作区系统层
不持久，应把环境、模型、编译缓存和原始结果放在持久存储。可选启动器接受
`KVCOMPRESS_HOST_ENV`、`KVCOMPRESS_CANN_ENV`、`KVCOMPRESS_ATB_ENV` 和
`KVCOMPRESS_TRITON_SOURCE`，用于指定本地路径；启动器会先清除继承的
Python/CANN 路径，再加载指定工具链。

```bash
scripts/kvcompress_ascend_env.sh python - <<'PY'
import torch, torch_npu, triton
print(torch.__version__, torch_npu.__version__, triton.__version__)
assert torch.npu.device_count() > 0
device = torch.device("npu:0")
assert torch.mm(torch.ones((32, 32), device=device),
                torch.ones((32, 32), device=device)).sum().item() == 32768
PY
```

宿主栈完成后，不重新解析加速依赖地安装插件源码。运行 manifest 生命周期
测试前，另行安装兼容的 Extension Manager：

```bash
scripts/kvcompress_ascend_env.sh python -m pip install \
  --no-deps --no-build-isolation -e .
scripts/kvcompress_ascend_env.sh python -m pytest -q
scripts/kvcompress_ascend_env.sh python tests/run_npu_kernel_smoke.py
```

先确认宿主原生扩展可导入，再完成一项实际模型请求，方可声明服务兼容。
性能测量前停止周期性 NPU 保活程序，并按适用的
[测试规程](../benchmark/README.zh.md)安排服务生命周期。
