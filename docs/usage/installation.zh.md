# 安装与兼容性

[English](installation.md) | 简体中文 | [用户指南](README.zh.md)

## 前置条件

准备 CANN、ATB、torch-npu、Triton-Ascend、vLLM-HUST 与
vLLM-Ascend-HUST 相互匹配的 Ascend 宿主栈，并将其作为整体部署。
本仓库所述的匹配源码版本线为 vLLM-HUST `>=0.29.1.post1.dev0,<0.30`、
vLLM-Ascend-HUST `>=0.25.1rc2.dev0,<0.26`，配合
Triton-Ascend 3.6。精确的旧版 Frontier 源码组合另有窄范围例外；
不能推及任意旧版或新版宿主。详见[兼容性表](../../README.zh.md#兼容范围)和
[环境说明](../development/environment-installation.zh.md)。

Python 需为 `>=3.10,<3.15`。Qwen3.5-35B-A3B 的 BF16 部署需要两张
Ascend NPU，以及足以容纳模型和 KV cache 的显存。服务账号必须能读取模型与
tokenizer。安装前检查宿主：

```bash
python -m pip show vllm vllm-ascend torch torch-npu triton-ascend
python -m pip check
npu-smi info
```

Qwen3.5 的 Gated-DeltaNet 路径还会导入 `fla_npu`。请安装所用
vLLM-Ascend-HUST 宿主版本固定、与 NPU 架构匹配的 FLA NPU 产物；
其导入名并不对应 PyPI 上独立的 `fla-npu` 包。在服务环境中运行
`python -c 'import fla_npu.ops.ascendc'` 检查。

## 安装 Manager 和插件

从认可的软件源或[源码仓库](https://github.com/vLLM-HUST/extension-manager)
将 Extension Manager `>=0.3,<0.4` 安装到与宿主**相同的 Python 环境**。
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
