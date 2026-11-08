# 安装与兼容性

[English](installation.md) | 简体中文 | [用户指南](README.zh.md)

## HUST 基座与插件环境

使用固定 HUST 源码基座和长期目录名 `kvcompress-host`。
准备好匹配的持久 Python 3.11 基座及 CANN/ATB 后，一键安装：

```bash
export KVCOMPRESS_BASE_PYTHON=/path/to/persistent/hust-base/bin/python
bash scripts/set_env.sh install b1 --fla-wheel /path/to/the-locked-fla.whl
scripts/set_env.sh check b1
```

脚本核对源码提交和版本，补齐插件独立环境，再安装当前插件。
基座前提、seed 复用、配置覆盖和依赖审计见[环境指南](../development/environment-installation.zh.md)。
Qwen3.5 BF16 还需要两张匹配的 NPU。

## 安装 Manager 和插件

从 [0.3.0 发布 wheel](https://github.com/vLLM-HUST/extension-manager/releases/tag/v0.3.0)
将 Extension Manager 安装到与宿主**相同的 Python 环境**。
将 `KVCOMPRESS_PLUGIN_VERSION` 设为与宿主兼容的已发布版本，然后安装，
不重新解析硬件栈：

```bash
python -m pip install --no-deps "vllm-ascend-kvcompress-hust==${KVCOMPRESS_PLUGIN_VERSION}"
vllm-hust-ext extension validate org.vllm-hust.ascend-kvcompress
```

`validate` 检查已安装扩展的 manifest。完成配置和启用后再运行更严格的
`extension check`。若出现版本错误，应先安装匹配的宿主栈，不能仅跳过检查。
源码可编辑安装需先准备宿主环境，再在仓库内执行
`python -m pip install --no-deps -e .`。

下一步是[配置](configuration.zh.md)。
