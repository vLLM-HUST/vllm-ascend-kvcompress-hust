# Persistent HUST host environment

English | [简体中文](environment-installation.zh.md)

Plugin development uses one persistent HUST source stack across plugin releases.
The default plugin environment is `.envs/kvcompress-host` beside the repository;
the durable vendor toolchain lives in `.envs/kvcompress-toolchain/Ascend`.
Neither directory name tracks the plugin release number.

## Locked dependencies

The machine-readable contract is [`requirements/ascend-host.json`](../../requirements/ascend-host.json).

| Component | Pinned identity |
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
| A2 FLA NPU | `26.9.1+deva4a7958` (SHA256 in the lock) |

HUST's Triton build declares `numpy>=1.26.4,<2.3`, so NumPy 2.2.6 is valid.
Public Triton-Ascend wheels are not equivalent substitutes for this source build. The reason is that it requires `numpy==1.26.4`, which conflicts with vLLM and vLLM-Ascend.
The launcher prepends the matching Triton source directory to avoid the generic
Triton package shadowing the Ascend backend.

## One-command plugin environment installation

Prerequisites: Linux aarch64, an **already built persistent Python 3.11 HUST
base** with the pinned vLLM/Ascend/Triton sources and NumPy/Transformers/SciPy,
and matching CANN/ATB. The installer provisions the plugin overlay, not the
HUST compilers or vendor SDK from scratch. It checks package metadata and all
three source HEADs before changing the target. It never resets source worktrees.
Existing tracked changes are fingerprinted and Git status is recorded.

On the current workspace, the launcher finds the durable HUST base automatically:

```bash
scripts/kvcompress_install_env.sh
```

On another machine, provide its persistent base and vendor locations:

```bash
export KVCOMPRESS_BASE_PYTHON=/path/to/python/hust-base/bin/python
scripts/kvcompress_install_env.sh \
  --cann-env /path/to/Ascend/cann-9.1.0/set_env.sh \
  --atb-env /path/to/Ascend/nnal/atb/set_env.sh \
  --fla-wheel /path/to/wheels/flash_linear_attention_npu_a2-26.9.1+deva4a7958-py3-none-manylinux_2_34_aarch64.whl
```

The script creates a Python virtual environment with `--system-site-packages`
to inherit the HUST base, pins the PyTorch/Numba/Manager overlay, verifies the
A2 FLA wheel hash if installation is needed, and installs this plugin editable
without re-resolving host dependencies. Keep the base, source repositories,
vendor SDK, wheel cache and environment on durable storage. This is the
existing A2/aarch64 stack; other architectures and FLA variants need a reviewed lock.

For a matched existing plugin environment, use `--seed-env /path/to/environment`
with a new target. This copies its overlay without network downloads and
recreates launchers for the new prefix. It requires the same Python base and
all pinned versions; the seed is preserved. Use `--prefix /path/to/envs/kvcompress-host`
or `KVCOMPRESS_HOST_ENV` to change the target. Existing targets can be rerun
without `--seed-env`; matching packages are reused.

`--check` checks versions/source identities/vendor paths without installing;
combine with `--seed-env` to check a seed before creating the target.
The installer does not enable the compression MOD or start a model server.

## Validation and runtime

```bash
scripts/kvcompress_install_env.sh --check
scripts/kvcompress_ascend_env.sh python - <<'PYTHON'
from importlib.metadata import version
import triton
print(version("vllm"), version("vllm-ascend"), version("triton-ascend"))
print(triton.__file__)
PYTHON
# After NPU allocation:
scripts/kvcompress_ascend_env.sh python tests/run_npu_kernel_smoke.py
```

`kvcompress-host.json` in the target records package locations, source commits,
tracked diff hashes, dirty status and launcher paths. `dependency-check.txt`
contains the full `pip check` output. The inherited host has known optional
package gaps (such as torchaudio, KV-transfer dependencies and FLA test tools)
and a historical Transformers metadata mismatch. The installer preserves and
reports them; success means the pinned overlay was configured, not that every
host Python requirement or NPU combination was qualified. It prints the audit
exit status. Review the report for the features you enable.

The launcher clears inherited Python/CANN paths, restores the pinned Triton
source, sources CANN/ATB and the Ascend vendor setup, and uses the installed
PyTorch CXX11 ABI. Overrides: `KVCOMPRESS_CANN_ENV`, `KVCOMPRESS_ATB_ENV`,
`KVCOMPRESS_TRITON_SOURCE`, `KVCOMPRESS_HOST_VENDOR_ENV`.
Hardware drivers come from the allocated machine. Without NPU allocation,
metadata and source import checks do not establish kernel/model compatibility.
Run the full plugin tests and NPU smoke/model checks before benchmarking.

For a custom `--prefix`, export `KVCOMPRESS_HOST_ENV` to that same path when using the launcher.
