# Installation and compatibility

English | [简体中文](installation.zh.md) | [User guide](README.md)

## HUST base and plugin environment

Use the pinned HUST source stack and the persistent `kvcompress-host` directory.
After provisioning the matched durable Python 3.11 base and CANN/ATB, run:

```bash
export KVCOMPRESS_BASE_PYTHON=/path/to/persistent/hust-base/bin/python
bash scripts/set_env.sh install b1 --fla-wheel /path/to/the-locked-fla.whl
scripts/set_env.sh check b1
```

The installer checks source commits and package versions, configures the
plugin overlay and installs this plugin. Prerequisites, seed reuse, overrides
and dependency audit are in the [environment guide](../development/environment-installation.md).
Qwen3.5 BF16 additionally needs two matching NPUs.

## Install the manager and plugin

Install Extension Manager `>=0.3,<0.4` from its [published wheel](https://github.com/vLLM-HUST/extension-manager/releases/tag/v0.3.0)
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
