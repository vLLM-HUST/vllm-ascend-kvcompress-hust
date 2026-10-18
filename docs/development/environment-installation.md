# Ascend host installation

English | [简体中文](environment-installation.zh.md)

This guide applies to any supported Ascend machine. Personal paths and
restart behavior belong in the ignored `AGENTS.dev.md`, when present. The
[2026-09-27 installation record](../archive/development/environment-installation-20260927.md)
preserves diagnosis from an older host.

## Host contract

The plugin wheel deliberately has no core `Requires-Dist`. Provision vLLM-HUST,
vLLM-Ascend, Triton-Ascend, PyTorch, torch-npu, CANN, ATB, and the device driver
as a mutually compatible stack before installing the plugin. The Extension
Manager manifest records the supported host range. Match native extensions to
the chosen Python ABI, and use the exact tested stack recorded with each
[result](../results/README.md). An installed package version alone does not
prove that its native module or kernels can run.

Use an isolated environment and choose model, cache, toolchain, and evidence
locations appropriate to the machine. On an ephemeral workspace, keep the
environment, model, compiler cache, and raw results on durable storage. The
optional launcher accepts `KVCOMPRESS_HOST_ENV`, `KVCOMPRESS_CANN_ENV`,
`KVCOMPRESS_ATB_ENV`, and `KVCOMPRESS_TRITON_SOURCE` for local paths.
It clears inherited Python/CANN paths before sourcing the selected toolchain.

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

After the host stack is complete, install source without re-resolving its
accelerator dependencies. Install a compatible Extension Manager separately
before running the manifest lifecycle tests:

```bash
scripts/kvcompress_ascend_env.sh python -m pip install \
  --no-deps --no-build-isolation -e .
scripts/kvcompress_ascend_env.sh python -m pytest -q
scripts/kvcompress_ascend_env.sh python tests/run_npu_kernel_smoke.py
```

Confirm the host's native extension imports and run one model request before
claiming service compatibility. Stop any periodic NPU activity before
performance measurements, and use the service lifecycle specified by the
applicable [benchmark protocol](../benchmark/README.md).
