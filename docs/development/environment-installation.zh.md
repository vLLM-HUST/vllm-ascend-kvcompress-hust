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
公共 Triton-Ascend wheel 可能采用不同的依赖约束，应使用锁文件指定的 HUST 源码构建。
启动器优先选择匹配的 Triton 源码路径，防止普通 Triton 包遮住 Ascend 后端。

## 安装插件环境

前提是 Linux aarch64、glibc >= 2.34，以及**已经构建好的持久 Python 3.11 HUST 基座**。
基座包含锁定版本的 editable vLLM/Ascend/Triton 包及 NumPy/Transformers/SciPy，
机器还需提供已部署的 CANN/ATB 9.1.0。
脚本只创建插件覆盖环境，不下载 Python、不拉取或编译 HUST 仓库，也不安装厂商 SDK。
修改目标环境前先检查基座包元数据及三套源码 HEAD；保留源码已有改动，记录其哈希和 Git 状态。

首次安装显式选择持久基座：

```bash
export KVCOMPRESS_BASE_PYTHON=/path/to/persistent/hust-base/bin/python
bash scripts/set_env.sh install b1 \
  --cann-env /path/to/persistent/Ascend/cann-9.1.0/set_env.sh \
  --atb-env /path/to/persistent/Ascend/nnal/atb/set_env.sh \
  --fla-wheel /path/to/wheels/flash_linear_attention_npu_a2-26.9.1+deva4a7958-py3-none-manylinux_2_34_aarch64.whl
bash scripts/set_env.sh check b1
```

未设置 `KVCOMPRESS_BASE_PYTHON` 时，新目标使用安装器当前运行的解释器，
它必须是匹配的 Python 3.11 HUST 基座。目标已安装时，会自动选择
`kvcompress-host.json` 记录的基座，不再按机器专用 Conda 路径猜测。

SDK 默认脚本为工作区 `.envs/kvcompress-toolchain/Ascend/cann-9.1.0/set_env.sh`
和 `.envs/kvcompress-toolchain/Ascend/nnal/atb/set_env.sh`；已有这些持久安装时可省略 SDK 参数。
保留脚本旁的厂商版本元数据：CANN `share/info/asc-devkit/version.info` 与 ATB
`latest/version.info`。检查读取元数据内容，并要求 SDK 提供
`tools/bishengir/bin/bishengir-compile` 和 `bishengir-opt` 可执行文件。

脚本通过 `--system-site-packages` 创建 venv 继承 HUST 基座，安装锁定的
PyTorch/Numba/Manager 覆盖包，再用 `--no-deps --no-build-isolation` editable 安装插件，
不重新解析宿主栈。基座解释器、源码、SDK 和环境都应保留在持久路径，覆盖环境依赖这些位置。

FLA wheel 必须匹配锁文件中的 SHA256。通过 `--fla-wheel` 或 `KVCOMPRESS_FLA_WHEEL`
提供精确本地文件，也可按锁文件中的文件名缓存到工作区 `.cache/fla-npu/`。
历史开发版已从上游撤下，需要从自己的制品存储取得同一文件，不能用当前正式版替代。
已有匹配 FLA 包时无需再次提供 wheel。

TP 自动校准还需要插件 calibration extra 声明的 `accelerate>=1.10,<2`。
匹配宿主已提供其余依赖时，可单独安装，避免重新解析宿主栈：

```bash
bash scripts/set_env.sh run b0 -- python -m pip install --no-deps 'accelerate>=1.10,<2'
```

`--prefix` 或 `KVCOMPRESS_HOST_ENV` 可覆盖目标路径；安装、检查和运行需选择同一路径。
重复安装会复用已匹配的包并刷新 editable 插件。
`--seed-env /path/to/overlay` 可将匹配的已有覆盖环境复制到新目标并重建启动脚本，无需下载包。
要求相同持久 Python 基座和全部锁定版本，原环境保持不变。

`--check` 不安装内容，检查覆盖环境的 Python 基座、继承 site-packages 的设置、
editable 插件来源、HUST 源码来源、包版本、SDK 元数据及启动必需文件。
可与 `--seed-env` 配合，指定新目标后预检种子环境。安装不会启用压缩或启动模型服务。

## 验证与启动

```bash
scripts/set_env.sh check b1
scripts/set_env.sh run b0 -- python - <<'PYTHON'
import sys
import torch, torch_npu, triton, vllm, vllm_ascend
import fla_npu.ops.ascendc
import vllm_ascend_kvcompress
print(sys.executable)
print(torch.__version__)
print(triton.__file__)
PYTHON
scripts/set_env.sh run b0 -- vllm-hust-ext extension validate org.vllm-hust.ascend-kvcompress
# 确认驱动和设备可用后：
scripts/set_env.sh run b0 -- python tests/run_npu_kernel_smoke.py
```

目标内的 `kvcompress-host.json` 保存包位置、源码提交、已跟踪改动哈希、工作树状态、SDK 版本身份和启动路径。
`dependency-check.txt` 保存完整 `pip check` 输出。
继承基座存在已知可选包缺口（如 torchaudio、KV transfer 依赖、FLA 测试工具）和历史 Transformers
元数据不一致；脚本保留并报告它们，打印审计退出码。
安装成功仅表示固定插件环境配置完成，不表示宿主全部 Python 依赖或任意 NPU 组合均通过验收。
请根据启用功能检查该报告。

启动器清理继承的 Python/动态库路径及 Shell 启动钩子（`BASH_ENV`/`ENV`），
按实际 PyTorch CXX11 ABI 加载 CANN/ATB 与 Ascend vendor 配置，再恢复宿主命令和
Triton 源码的优先级，并将所选 SDK 的 BishengIR 工具加入 PATH。支持 `KVCOMPRESS_CANN_ENV`、`KVCOMPRESS_ATB_ENV`、
`KVCOMPRESS_TRITON_SOURCE`、`KVCOMPRESS_HOST_VENDOR_ENV` 覆盖配置。
驱动由分配到的机器提供；元数据、导入和动态库依赖解析检查不能证明 kernel/模型兼容。
原生算子库初始化也需要可用的驱动和设备，应与硬件检查一起执行。
benchmark 前还需完成插件完整测试、NPU smoke 和模型检查。

使用自定义 `--prefix` 时，启动器需设置相同路径的 `KVCOMPRESS_HOST_ENV`。

## 插件开关对照与评测客户端

同基座插件开关对照、外部客户端及统一入口见[浦江评测指南](../benchmark/pujiang.zh.md)。两侧共用本页宿主环境、工具链及兼容桥。
