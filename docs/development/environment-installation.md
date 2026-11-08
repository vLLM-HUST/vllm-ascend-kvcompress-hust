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
Public Triton-Ascend wheels can carry different dependency constraints; use the
pinned HUST source build specified by the lock.
The launcher prepends the matching Triton source directory to avoid the generic
Triton package shadowing the Ascend backend.

## Plugin environment installation

Prerequisites: Linux aarch64 with glibc >= 2.34, an **already-built persistent
Python 3.11 HUST base** containing the pinned editable vLLM/Ascend/Triton packages
and NumPy/Transformers/SciPy, and provisioned CANN/ATB 9.1.0.
The installer creates the plugin overlay; it does not download Python, fetch or
build HUST source repositories, or install vendor SDKs. It checks the base
package metadata and all three source HEADs before changing the target. Existing
source changes are preserved and their diff hashes and status are recorded.

Select the base explicitly for the initial installation:

```bash
export KVCOMPRESS_BASE_PYTHON=/path/to/persistent/hust-base/bin/python
bash scripts/set_env.sh install b1 \
  --cann-env /path/to/persistent/Ascend/cann-9.1.0/set_env.sh \
  --atb-env /path/to/persistent/Ascend/nnal/atb/set_env.sh \
  --fla-wheel /path/to/wheels/flash_linear_attention_npu_a2-26.9.1+deva4a7958-py3-none-manylinux_2_34_aarch64.whl
bash scripts/set_env.sh check b1
```

Without `KVCOMPRESS_BASE_PYTHON`, the installer uses its running interpreter for
a new target; that interpreter must be the matched Python 3.11 HUST base.
For an installed target, it selects the base recorded in `kvcompress-host.json`.
There is no machine-specific Conda path fallback.

The default SDK paths are workspace `.envs/kvcompress-toolchain/Ascend/cann-9.1.0/set_env.sh`
and `.envs/kvcompress-toolchain/Ascend/nnal/atb/set_env.sh`; omit the SDK arguments
when those persistent installations already exist. Keep the vendor version
metadata: CANN `share/info/asc-devkit/version.info` and ATB `latest/version.info`
beside their setup scripts. The checks read their contents and require the SDK's
`tools/bishengir/bin/bishengir-compile` and `bishengir-opt` executables.

The installer creates a venv with `--system-site-packages` to inherit the HUST
base, installs the locked PyTorch/Numba/Manager overlay, and installs this plugin
editable with `--no-deps --no-build-isolation`. It does not re-resolve the host
stack. Keep the base interpreter, source checkouts, SDK and environment at their
persistent paths; the overlay depends on them.

FLA must match the lock's SHA256. Supply the exact local wheel through
`--fla-wheel` or `KVCOMPRESS_FLA_WHEEL`, or cache it under workspace
`.cache/fla-npu/` using the filename in the lock. The historical development
release is no longer available upstream; obtain the same artifact from your
artifact store. A current upstream release is not a substitute. A wheel is not
needed when the matching FLA package is already installed.

For TP auto-calibration, the plugin's calibration extra also requires
`accelerate>=1.10,<2`. With its other dependencies provided by the matched host,
install it without re-resolving that stack:

```bash
bash scripts/set_env.sh run b0 -- python -m pip install --no-deps 'accelerate>=1.10,<2'
```

`--prefix` or `KVCOMPRESS_HOST_ENV` changes the overlay target. Repeat the same
path for installation, checks and runtime. Matching installed packages are
reused on subsequent installs, while the editable plugin is refreshed.
`--seed-env /path/to/overlay` copies a matched existing overlay to a new target
without downloading packages and recreates its launchers. The seed must use the
same persistent Python base and locked package versions; it remains unchanged.

`--check` performs identity and path checks without installing. It verifies the
overlay's Python base, inherited site-packages setting, editable plugin origin,
HUST source origins, package versions, SDK metadata and launcher prerequisites.
Use `--check --seed-env` with a new target to preflight a seed. Installation does
not enable compression or start a model service.

## Validation and runtime

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
# Only after driver/device availability is confirmed:
scripts/set_env.sh run b0 -- python tests/run_npu_kernel_smoke.py
```

`kvcompress-host.json` in the target records package locations, source commits,
tracked diff hashes, dirty status, SDK version identities and launcher paths. `dependency-check.txt`
contains the full `pip check` output. The inherited host has known optional
package gaps (such as torchaudio, KV-transfer dependencies and FLA test tools)
and a historical Transformers metadata mismatch. The installer preserves and
reports them; success means the pinned overlay was configured, not that every
host Python requirement or NPU combination was qualified. It prints the audit
exit status. Review the report for the features you enable.

The launcher clears inherited Python/library paths and shell startup hooks
(`BASH_ENV`/`ENV`), loads CANN/ATB and the Ascend vendor setup using the installed
PyTorch CXX11 ABI, and restores the selected host and Triton source precedence.
It also places the selected SDK's BishengIR tools on PATH. Overrides: `KVCOMPRESS_CANN_ENV`, `KVCOMPRESS_ATB_ENV`,
`KVCOMPRESS_TRITON_SOURCE`, `KVCOMPRESS_HOST_VENDOR_ENV`.
Hardware drivers come from the allocated machine. Without NPU allocation,
metadata, imports and dynamic-library dependency resolution do not establish
kernel/model compatibility. Native operator library initialization also needs
a working driver/device; run it with the hardware checks.
Run the full plugin tests and NPU smoke/model checks before benchmarking.

For a custom `--prefix`, export `KVCOMPRESS_HOST_ENV` to that same path when using the launcher.

## Plugin-off/on evaluation and clients

See the [Pujiang guide](../benchmark/pujiang.md) for same-base plugin-off/on comparisons, external clients and unified entry points. Both arms share this host environment, toolchain and compatibility bridge.
