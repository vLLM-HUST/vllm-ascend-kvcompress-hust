# Candidate baseline: CPU preparation, 2026-10-26

English | [简体中文](README.zh.md)

The isolated candidate stack built and installed successfully. Final plugin
tests reported **223 passed, 2 failed, 1 deselected**. Both failures stopped at
NPU architecture discovery on a machine without an allocated NPU. They remain
failures pending a hardware retry. This record does not qualify the stack for
serving and does not extend the plugin's validated compatibility range.

## Pinned combination

| Component | Source / runtime |
| --- | --- |
| vLLM-HUST core | `ebfcfba6507501c3a359a75eb4911ecc442bc2ff` |
| vLLM-Ascend-HUST | `7c8ec865a12f7b6a14d53b61e7f61b5e811dd483`, with the dependency patch below |
| Triton-Ascend-HUST | `b90ab001238f8dde175aebfea23de36bd2cf06a6` |
| KVCompress | `3ddff1c2afee1eccc3d1b65def61167752298f5f`, package `0.10.0` |
| Extension Manager | `36cd420b28c0569d94d5205b9e9e4cc1c8bc0e31`, package `0.3.0` |
| Platform | AArch64, Python `3.12.13`, CANN / ATB `9.1.0` |
| PyTorch | `torch==2.13.0+cpu`, `torch-npu==2.13.0rc1` |
| Transformers / Hub | `5.17.0` / `1.33.0` |
| Target, not yet exercised | Qwen3.5-35B-A3B, BF16, two 910B2 devices, TP=2 |

[manifest.json](manifest.json) records installed versions, build exit codes,
wheel provenance, unresolved checks and SHA256 values. This is an adapted
candidate, not a reconstruction of a historical benchmark environment. Python
3.12.13 also differs from the plugin README's validated Python 3.11.16 snapshot.

## Dependency adaptation

The pinned core requires `transformers>=5.16.1,<5.19.0`, while the original
Ascend runtime and build declarations pin `5.14.1`; they cannot be satisfied
together. The [patch](patches/ascend-transformers-5.17.patch) changes both Ascend
declarations to `5.17.0` in the isolated copy. No plugin algorithm was changed.
The Ascend wheel's `.d20261026` suffix reflects the modified build tree.

The Triton build applies its own upstream patch to 17 tracked files. Its patch
hash is recorded in the manifest; this was not a manual kernel adaptation.
The core was built with `VLLM_TARGET_DEVICE=empty`, Ascend with the explicit
`SOC_VERSION=ascend910b2`, and Triton with the Ascend backend enabled. Installed
wheel identities are in [wheels-installed.json](evidence/wheels-installed.json).

## Completed checks and limits

| Check | Evidence and interpretation |
| --- | --- |
| Python dependencies | [200 installed packages checked](evidence/dependency-check.log); no declared dependency conflict |
| Transformers APIs | [38 import checks](evidence/transformers-api-check.json) and Qwen3.5 Dense/MoE config construction; no device forward pass |
| Model inputs | [Offline config/tokenizer and 14 shard presence checks](evidence/offline-model-check.json); no weight loading or full weight checksum |
| Core input path | [Sample token IDs and chat-template output agree](evidence/core-model-inputs-check.json) with Transformers |
| Host source contracts | [AST/text checks](evidence/source-contract-check.json): expected cache helpers and worker methods exist; selected Triton source does not emit `enable-optimized-metaop` |
| Native packages | [Triton Ascend binding import, compiler version and three library linkage checks](evidence/native-package-check.json); not Ascend native initialization or kernel execution |
| Manager lifecycle | [Validate](evidence/manager-validate.log), [configure](evidence/manager-configure.log), [enable](evidence/manager-enable.log), [enabled status](evidence/manager-enabled-status.log), [disable](evidence/manager-disable.log), [disabled status](evidence/manager-disabled-status.log), using isolated test configuration |
| Plugin tests | [Original final log](evidence/plugin-cpu-final-tests.log) and [JUnit XML](evidence/plugin-cpu-final-tests.xml): 223 passed, 2 failed, 1 deselected |

The three linkage checks cover `libvllm_ascend_kernels.so`, `libtriton.so` and
`libcust_opapi.so` only. Manager enablement does not start a model or prove
runtime compatibility. The package [freeze](evidence/installed-package-freeze.log)
is an inventory, not a portable installation command; locally built wheels must
be supplied separately. Absolute paths in raw files identify the original run.

## Unresolved tests

`test_grouped_topk_router_scaling_compatibility` and
`test_runner_generates_calibration_before_loading_model` import Ascend MoE,
then core FLA, then Triton's `get_current_target/get_arch`. They fail with
`cann_get_soc_name failed` and `get_arch returned NULL without setting an
exception`. Device discovery failed before their intended assertions. An NPU
retry may expose additional incompatibilities; neither test is counted as passed.

`test_ascend_v1_worker_seams_match_current_host` was deliberately deselected
because it loads the native worker runtime. The final run used the repository's
unchanged tests:

```bash
export VLLM_HUST_BENCHMARK_ORIGINALS=/absolute/path/to/verified-originals
VLLM_PLUGINS='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  python -m pytest -q \
  -k 'not test_ascend_v1_worker_seams_match_current_host' \
  --junitxml=/absolute/path/to/plugin-cpu-final-tests.xml
```

Provision the matched environment and the [verified original test inputs](../../../usage/benchmark-originals.md)
first. An earlier preparatory run reported 216 passed / 6 failed: five failures
were missing original test inputs and one occurred before Ascend installation.
The final run supersedes that attempt; its original log/XML hashes remain in
the manifest. No failed outcome was relabeled as passed.

## Hardware gate

After NPU allocation, follow the [environment guide](../../environment-installation.md)
to verify native initialization and BF16 execution. Retry all three unresolved
tests, check TP2/HCCL, and run the plugin kernel and Qwen3.5 GDN output smoke
tests. Then issue actual Qwen3.5 BF16/TP2 requests with the plugin disabled and
enabled; the enabled case must exercise compression, not merely plugin loading.
Only after these gates should B0/B1 or TTFT measurement begin.

No NPU kernel, model service, B0/B1 or TTFT measurement was run in this phase.
The published evidence is a byte-preserving selection with [checksums](SHA256SUMS).
The full local archive, including build logs and earlier attempts, is identified
by SHA256 in the manifest; it is not bundled here. Wheels, weights, environments,
build caches, SSH settings and duplicate historical benchmark inputs are omitted.
