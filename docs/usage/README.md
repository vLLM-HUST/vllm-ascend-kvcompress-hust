# User guide

English | [简体中文](README.zh.md)

Follow these guides in order for a first deployment. Choose a package version
and configuration compatible with the host; the checked-in examples describe
this source tree and may differ from an older published package. The host
stack and model weights are separate installations; the plugin package does
not supply either one.

1. [Install and check compatibility](installation.md).
2. [Choose and enable a configuration](configuration.md).
3. [Generate or reuse calibration statistics](calibration.md).
4. [Start a service and verify compression](serving.md).
5. [Diagnose startup and runtime problems](troubleshooting.md).

The [Qwen3.5-35B-A3B example](../../examples/qwen3.5-35b-a3b-triattention.json)
targets the validated 1,152-token attention-page layout. Read the page
alignment note in the configuration guide before using another host layout.
For measured performance claims, choose the applicable
[benchmark protocol](../benchmark/README.md) and record evidence in
[results](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/results/README.md); a successful request only proves that
the service works.
