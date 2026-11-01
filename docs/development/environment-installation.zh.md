# 持久化 HUST 宿主环境

[English](environment-installation.md) | 简体中文

插件后续版本共用同一套固定 HUST 源码基座。默认插件环境为仓库相邻的
`.envs/kvcompress-host`，持久 CANN/ATB 位于 `.envs/kvcompress-toolchain/Ascend`。
环境目录名不随插件版本变化。

## 固定依赖

机器可读配置为 [`requirements/ascend-host.json`](../../requirements/ascend-host.json)。

| 组件 | 固定身份 |
| --- | --- |
| Python | 3.11 |
| vLLM-HUST | `0.29.1.post1.dev456+gebfcfba65.empty` / `ebfcfba6507501c3a359a75eb4911ecc442bc2ff` |
| vLLM-Ascend-HUST | `0.25.1rc2.dev774+hust.20260903.4.g7c8ec865a` / `7c8ec865a12f7b6a14d53b61e7f61b5e811dd483` |
| Triton-Ascend-HUST | `3.6.0+gitf5e142a9` / `f5e142a9963b01844625a4fdcf16bf8c24b3a5de` |
| Triton / NumPy / SciPy | `3.6.0` / `2.2.6` / `1.13.1` |
| PyTorch / torch-npu / torchvision | `2.13.0+cpu` / `2.13.0rc1` / `0.28.0+cpu` |
| Transformers / Numba / llvmlite | `5.18.0` / `0.63.1` / `0.46.0` |
| Extension Manager | `0.3.0` |
| CANN / ATB | `9.1.0` / `9.1.0` |
| A2 FLA NPU | `26.9.1+deva4a7958` （锁文件记录 SHA256） |

HUST Triton 的元数据允许 `numpy>=1.26.4,<2.3`，实际使用 NumPy 2.2.6。
公共 Triton-Ascend wheel 不能直接替代该修补源码构建，原因在于其要求其要求 `numpy==1.26.4`,与 vLLM、 vLLM-Ascend 冲突
启动器优先选择匹配的 Triton 源码路径，防止普通 Triton 包遮住 Ascend 后端。

## 一键安装插件环境

前提是 Linux aarch64、**已经构建好的持久 Python 3.11 HUST 基座**，其中包含固定版本的
vLLM/Ascend/Triton 源码及 NumPy/Transformers/SciPy，并提供配套 CANN/ATB。
脚本负责插件独立环境，不从零构建 HUST 编译器或安装厂商 SDK。
修改目标环境前先检查包元数据及三套源码 HEAD，不重置源码工作树，记录已有改动的哈希和 Git 状态。

当前 workspace 自动识别持久 HUST 基础环境，直接执行：

```bash
scripts/kvcompress_install_env.sh
```

其他机器可指定持久基座、CANN/ATB 和 FLA wheel：

```bash
export KVCOMPRESS_BASE_PYTHON=/path/to/python/hust-base/bin/python
scripts/kvcompress_install_env.sh \
  --cann-env /path/to/Ascend/cann-9.1.0/set_env.sh \
  --atb-env /path/to/Ascend/nnal/atb/set_env.sh \
  --fla-wheel /path/to/wheels/flash_linear_attention_npu_a2-26.9.1+deva4a7958-py3-none-manylinux_2_34_aarch64.whl
```

脚本创建带 `--system-site-packages` 的虚拟环境继承 HUST 基座，补齐固定 PyTorch/Numba/Manager
插件环境依赖；需要安装 A2 FLA 时校验 wheel 哈希，然后 editable 安装当前插件，不重新解析宿主依赖。
基座、源码、SDK、wheel 缓存和环境均需放在持久存储。本锁仅对应 A2/aarch64；其他架构或 FLA 变体需单独审核。

若已有匹配的插件环境，可用 `--seed-env /path/to/environment` 安装到一个新目标目录：
复制独立环境中的包并重建启动脚本，无需重新下载；要求同一 Python 基座和全部固定版本，保留原环境。
通过 `--prefix /path/to/envs/kvcompress-host` 或 `KVCOMPRESS_HOST_ENV` 指定目标。
目标已存在时直接重跑（不带 `--seed-env`），复用已匹配的包。

`--check` 仅检查版本、源码身份和厂商脚本路径，不安装；可与 `--seed-env` 配合预检。
脚本不会启用压缩 MOD 或启动模型服务。

## 验证与启动

```bash
scripts/kvcompress_install_env.sh --check
scripts/kvcompress_ascend_env.sh python - <<'PYTHON'
from importlib.metadata import version
import triton
print(version("vllm"), version("vllm-ascend"), version("triton-ascend"))
print(triton.__file__)
PYTHON
# 分配 NPU 后：
scripts/kvcompress_ascend_env.sh python tests/run_npu_kernel_smoke.py
```

目标内的 `kvcompress-host.json` 保存包位置、源码提交、已跟踪改动哈希、工作树状态和启动路径。
`dependency-check.txt` 保存完整 `pip check` 输出。
继承基座存在已知可选包缺口（如 torchaudio、KV transfer 依赖、FLA 测试工具）和历史 Transformers
元数据不一致；脚本保留并报告它们，打印审计退出码。
安装成功仅表示固定插件环境配置完成，不表示宿主全部 Python 依赖或任意 NPU 组合均通过验收。
请根据启用功能检查该报告。

启动器清理继承的 Python/CANN 路径，恢复固定 Triton 源码路径，加载 CANN/ATB 与 Ascend vendor
脚本，按实际 PyTorch 查询 CXX11 ABI。支持 `KVCOMPRESS_CANN_ENV`、`KVCOMPRESS_ATB_ENV`、
`KVCOMPRESS_TRITON_SOURCE`、`KVCOMPRESS_HOST_VENDOR_ENV` 覆盖配置。
驱动由分配到的机器提供；未分配 NPU 时的元数据和源码导入检查不能证明 kernel/模型兼容。
benchmark 前还需完成插件完整测试、NPU smoke 和模型检查。

使用自定义 `--prefix` 时，启动器需设置相同路径的 `KVCOMPRESS_HOST_ENV`。
