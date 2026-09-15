# 当前宿主环境安装说明

[English](environment-installation.md) | 简体中文

## 宿主栈依赖边界

插件安装到预先部署好的宿主环境中，不负责解析宿主的 Python 依赖图。因此插件
wheel 的核心 `Requires-Dist` 为空；支持的 `vllm-ascend` 范围仍由 Extension
Manager 清单声明。

不要把当前支持的 vLLM 0.28 / vLLM-Ascend 0.25 快照与 Triton-Ascend 3.2.2
混装。这里并不是缺少 NumPy 分发包：Triton-Ascend 3.2.2 要求
`numpy==1.26.4`，而 vLLM 0.28 要求 `opencv-python-headless>=4.13`，这些可用
OpenCV wheel 又要求 `numpy>=2`，不存在同时满足二者的 NumPy 版本。固定 OpenCV
4.9 也不成立，因为它不满足 vLLM 的 `>=4.13` 约束。应先部署配套的
Triton-Ascend 3.6 / NumPy 2 宿主锁，再用 `--no-deps` 安装本插件。

如果 pip 同时列出不带版本和精确版本的 `vllm` / `vllm-ascend`，不带版本的条目
来自旧插件元数据，并非有意重复安装宿主。请构建并测试核心依赖列表为空的当前
wheel。

本文记录 2026-09-11 源码快照执行 dev-hub `quickstart.sh` 选项五后实际遇到的
安装问题。处理过程不修改 dev-hub、vLLM-HUST、vLLM-Ascend-HUST 或
Triton-Ascend 仓库源码。

## 现象与原因

发行元数据显示 `triton-ascend 3.6.0+git8f0a4de8`，但导入 `triton` 报错：

```text
ModuleNotFoundError: No module named 'triton._C.libtriton.ascend'
```

已安装 editable 包走的是社区 `setup.py` 路径，其 CMake cache 中
`TRITON_PLUGIN_DIRS` 为空；生成的 `libtriton.so` 有通用 binding，但没有
Ascend Python 模块。对于这一代源码，执行
`TRITON_CODEGEN_BACKENDS=ascend pip install -e .` 并不等价于仓库提供的
`setup_ascend.py` 构建入口。

## 不改共享仓库的恢复方法

`setup_ascend.py` 会给其工作树应用 Ascend patch，因此应在临时副本中构建，
确保共享 checkout 不变：

```bash
BUILD_DIR="$(mktemp -d /tmp/triton-ascend-build.XXXXXX)"
cp -a /workspace/zjw/triton-ascend-hust/. "$BUILD_DIR/"
rm -rf "$BUILD_DIR/build"

cd "$BUILD_DIR"
env \
  PATH=/root/miniconda3/envs/vllm-hust-dev/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  TRITON_BUILD_WITH_CLANG_LLD=0 \
  TRITON_BUILD_PROTON=OFF \
  TRITON_BUILD_TD=OFF \
  TRITON_BUILD_NPUIR=OFF \
  TRITON_WHEEL_NAME=triton-ascend \
  TRITON_APPEND_CMAKE_ARGS=-DTRITON_BUILD_UT=OFF \
  MAX_JOBS=16 \
  /root/miniconda3/envs/vllm-hust-dev/bin/python setup_ascend.py bdist_wheel

/root/miniconda3/envs/vllm-hust-dev/bin/python -m pip install \
  --no-deps --force-reinstall dist/triton_ascend-*.whl
```

本机 Clang 不接受 `-fuse-ld=lld`，所以本次使用
`TRITON_BUILD_WITH_CLANG_LLD=0`；这不是所有环境都必须设置的参数。构建过程
可能会把锁定版本的 Ascend LLVM 下载到标准 Triton cache。

运行插件测试前先验证：

```bash
/root/miniconda3/envs/vllm-hust-dev/bin/python - <<'PY'
import triton
from triton._C.libtriton.ascend import ir
print(triton.__version__)
print(ir)
PY
```

然后不解析依赖地重装本插件：

```bash
/root/miniconda3/envs/vllm-hust-dev/bin/python -m pip install \
  --no-deps --no-build-isolation -e /workspace/zjw/vllm-ascend-kvcompress-hust
```

正式验收需记录 Triton 源码 commit 和所构建 wheel 的 hash。本机已有的 3.2.2
wheel 只能用于降级诊断，不能作为当前 3.6.0 宿主栈的正式证据。
