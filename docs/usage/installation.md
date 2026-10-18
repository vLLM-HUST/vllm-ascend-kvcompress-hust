# Installation and compatibility

English | [简体中文](installation.zh.md) | [User guide](README.md)

## Prerequisites

Prepare an Ascend host with its matching CANN, ATB, torch-npu, Triton-Ascend,
vLLM-HUST, and vLLM-Ascend-HUST versions. Provision that stack as a tested
unit. The matched source line documented by this repository uses vLLM-HUST
`>=0.29.1.post1.dev0,<0.30` and vLLM-Ascend-HUST
`>=0.25.1rc2.dev0,<0.26`, with Triton-Ascend 3.6. The exact legacy
Frontier pair has a separate narrow allowance; arbitrary older or newer
hosts are not supported. See the [compatibility table](../../README.md#compatibility)
and [environment details](../development/environment-installation.md).

Use Python `>=3.10,<3.15`. For Qwen3.5-35B-A3B in BF16, provide two
Ascend NPUs and enough memory for both model and KV cache. Make the model and
tokenizer accessible to the service account. Check the host before installing:

```bash
python -m pip show vllm vllm-ascend torch torch-npu triton-ascend
python -m pip check
npu-smi info
```

The Qwen3.5 Gated-DeltaNet path also imports `fla_npu`. Install the
architecture-matched FLA NPU artifact pinned by your vLLM-Ascend-HUST host
release; its import name is not a standalone `fla-npu` package on PyPI.
Verify `python -c 'import fla_npu.ops.ascendc'` in the serving environment.

## Install the manager and plugin

Install Extension Manager `>=0.3,<0.4` from the approved package source or
its [source repository](https://github.com/vLLM-HUST/extension-manager)
into the **same Python environment** as the host. Set
`KVCOMPRESS_PLUGIN_VERSION` to a published version compatible with that host;
then install it without re-resolving the hardware stack:

```bash
python -m pip install --no-deps "vllm-ascend-kvcompress-hust==${KVCOMPRESS_PLUGIN_VERSION}"
vllm-hust-ext extension validate org.vllm-hust.ascend-kvcompress
```

`validate` checks the installed extension manifest. Run the stricter
`extension check` after configuring and enabling the extension. A version
error means the host and plugin do not form a supported combination; install
a matched host stack before retrying. For an editable source install, use
`python -m pip install --no-deps -e .` in this repository after the host
environment is ready.

Continue with [configuration](configuration.md).
