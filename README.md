# vLLM Ascend KV Compression

English | [简体中文](README.zh.md)

An independently packaged TriAttention KV-cache compression plugin for the
upstream-aligned vLLM-HUST and vLLM-Ascend-HUST stacks. Version 0.3 no longer
depends on the removed fork-only KV-compression lifecycle and does not modify
either host repository.

> Status: experimental. Package and Extension Manager lifecycle validation,
> Ascend 910B2 kernel smoke, manager-wrapped service startup, and repeated
> compression all pass on the declared source snapshots. The default 2048-token
> budget failed the recorded long-context quality guardrail, and the exact
> dependency stack plus the full performance/HBM matrix remain release gates.
> See [Validation](docs/validation.md).

## Ownership and maintenance

- School: Huazhong University of Science and Technology (HUST)
- Group: CGCL
- Advisor: Prof. Yao Wan (万瑶)
- Project lead: Sichen Liu (刘思辰), [@Seas0](https://github.com/Seas0)
- Maintainers: Jiawan Zhang (张家万),
  [@Jiawan23](https://github.com/Jiawan23); Ruohao Wei (韦若皓),
  [@kotoriqaq0](https://github.com/kotoriqaq0); and
  [@Seas0](https://github.com/Seas0)

The team agrees to maintain compatibility with vLLM-HUST and
vLLM-Ascend-HUST and to distribute this work through the vLLM-HUST Extension
Manager. This repository is a CGCL-maintained plugin; it is not code built into
the upstream-aligned host repositories.

## Algorithm source and license

The scoring method is an Ascend adaptation of
[TriAttention](https://github.com/WeianMao/triattention), snapshot
[`a4bc3c8f709db60f016ef42c3feb290fd0c00c1b`](https://github.com/WeianMao/triattention/tree/a4bc3c8f709db60f016ef42c3feb290fd0c00c1b),
described in [*TriAttention: Efficient Long Reasoning with Trigonometric KV
Compression*](https://arxiv.org/abs/2604.04921). The implementation was
reworked for Ascend paged K/V storage and does not copy the upstream CUDA
runtime kernels. See [NOTICE](NOTICE) and the
[adaptation notes](docs/ascend-adaptation-vs-triattention-vllm.md).

Repository code and committed documentation are Apache-2.0. Committed
calibration statistics are aggregate model-derived artifacts and have
incomplete generation provenance; do not assume the repository license alone
grants redistribution of the originating model or dataset. Raw benchmark
inputs, model weights, service logs, and unpublished datasets are not part of
the distributable package. The precise scope is recorded in
[Ownership and licensing](docs/ownership-and-licensing.md).

## Compatibility

| Component | Supported line | Validated snapshot |
| --- | --- | --- |
| vLLM-HUST / `vllm` | `>=0.17.2rc1.dev0,<0.18` | `5b343ed52` (`0.17.2rc1.dev5941+g5b343ed52.empty`) |
| vLLM-Ascend-HUST / `vllm-ascend` | `>=0.25.1rc1,<0.26` | `4e57439` (`0.25.1rc1+hust.20260903.4`) |
| Extension Manager | `>=0.2.0.dev0,<0.3` | `9fb467e` |
| Python | `>=3.10,<3.15` | 3.11.16 |

The supported launch is one Ascend NPU, the standard v1 scheduler (including
the current Ascend `BalanceScheduler` wrapper with balancing disabled) and
`NPUModelRunner`, one plain full-attention KV group, block size 128, and dense
BF16/FP16 K/V. Unsupported combinations fail during startup.

## Install and manage

Install the current host stack first, then install the plugin and Extension
Manager. From a source checkout:

```bash
python -m pip install ./extension-manager
python -m pip install ./vllm-ascend-kvcompress-hust
```

After a release is uploaded to PyPI, the equivalent plugin command is:

```bash
python -m pip install 'vllm-ascend-kvcompress-hust[manager]==0.3.0'
```

Copy [examples/triattention.json](examples/triattention.json), replace
`stats_path` with an absolute path to statistics for the exact model revision,
then configure and enable the extension:

```bash
vllm-hust-ext extension validate org.vllm-hust.ascend-kvcompress
vllm-hust-ext extension configure org.vllm-hust.ascend-kvcompress \
  --file /absolute/path/triattention.json
vllm-hust-ext extension enable org.vllm-hust.ascend-kvcompress
vllm-hust-ext extension status org.vllm-hust.ascend-kvcompress
```

Launch through the manager. The plugin must be present in `VLLM_PLUGINS` when
that variable is already used as an allowlist:

```bash
export VLLM_PLUGINS=ascend,ascend_kvcompress
vllm-hust-ext run -- vllm serve /path/to/model \
  --block-size 128 \
  --no-enable-prefix-caching \
  --no-async-scheduling
```

If `VLLM_PLUGINS` is unset, vLLM discovers all installed general plugins. The
plugin still remains inert until Extension Manager or the direct enable flag
activates it.

To disable safely, stop the host process, disable the extension, and restart:

```bash
vllm-hust-ext extension disable org.vllm-hust.ascend-kvcompress
```

To uninstall, first stop every host process, then run:

```bash
vllm-hust-ext extension disable org.vllm-hust.ascend-kvcompress
vllm-hust-ext extension forget org.vllm-hust.ascend-kvcompress
python -m pip uninstall vllm-ascend-kvcompress-hust
```

These operations only change package and Extension Manager state; they do not
edit vLLM-HUST or vLLM-Ascend-HUST.

### Direct activation without Extension Manager

For development only, supply the same JSON as a path or inline object and
restart the host:

```bash
export VLLM_PLUGINS=ascend,ascend_kvcompress
export VLLM_ASCEND_KVCOMPRESS_ENABLED=1
export VLLM_ASCEND_KVCOMPRESS_CONFIG=/absolute/path/triattention.json
vllm serve /path/to/model --block-size 128 --no-enable-prefix-caching
```

Unset both `VLLM_ASCEND_KVCOMPRESS_*` variables and restart to disable direct
activation.

## Runtime design

The package uses the public `vllm.general_plugins` entry point and a static
Extension Manager manifest. Once explicitly enabled, its adapter checks and
hooks these current host symbols:

- `vllm.v1.core.sched.scheduler.Scheduler`
- `vllm.v1.core.kv_cache_manager.KVCacheManager.allocate_slots`
- `vllm.v1.worker.block_table.BlockTable.compute_slot_mapping`
- `vllm_ascend.worker.model_runner_v1.NPUModelRunner` methods
  `initialize_kv_cache`, `_update_states`, `_build_attention_metadata`, and
  `sample_tokens`

The runner hook is installed lazily when the Ascend worker module is actually
loaded, so API and manager processes do not import the NPU runner. These host
symbols are internal and not frozen; the narrow supported version range and
startup checks are intentional. Every host-line update requires the acceptance
suite before widening the range.

Compression runs after a synchronous model step. The scheduler frees the old
tail only at the next scheduling barrier, while semantic RoPE positions remain
unchanged and physical slot/attention lengths use a per-request offset. Prefix
caching is disabled because compressed blocks no longer represent a hashable
semantic prefix.

## Long-context optimization

Version 0.3 keeps repeated compression at a block-aligned physical budget and
adds the following hot-path work:

- direct scoring from paged Ascend K cache, without materializing every key;
- persistent full-length score, K/V copy, aggregate, and dense-index buffers;
- fused NPU normalization, query-head maximum, and cross-layer accumulation;
- JIT parameters that remain dynamic across compression rounds and request
  lengths, avoiding a new score/aggregate compilation for each length;
- one score workspace reused by both specialized and generic paths;
- sampled scoring layers (`score_layer_stride`) while materializing every
  layer; and
- device-resident semantic-to-physical offsets with one slot-mapping pass.

These changes reduce temporary allocation, compilation, and launch pressure.
On an Ascend 910B2 kernel microbenchmark, paged copy, direct scoring, and fused
aggregation were 1.70x, 2.47x, and 1.04x faster than their generic references.
Making round/length arguments dynamic eliminated the observed 5--6 second
per-length recompilations: after one cold compile, alternating 2,176- and
6,311-token transactions completed in 22--26 ms. In a service workload with a
6,311-token prompt and 300 generated tokens, the optimized plugin averaged
25.71 seconds versus 35.97 seconds before this change, but it remained 2.0%
slower than the matched baseline. At four concurrent requests it delivered
897.6 total tok/s versus 923.1 baseline (-2.8%), while logged KV-cache usage
was about 20% versus 60%. These are limited acceptance measurements, not a
general throughput claim; see the [validation record](docs/validation.md).
Historical results from the removed host lifecycle are kept separately in
[results](docs/resuts.md) and must not be compared as 0.3 acceptance data.

## Conflict matrix

| Feature | 0.3 status | Behavior |
| --- | --- | --- |
| Prefix cache | Conflict | Startup rejection; must be disabled |
| Speculative decoding | Conflict | Startup rejection |
| KV transfer / disaggregated P/D | Conflict | Startup rejection |
| Quantized KV | Conflict | Startup rejection; only dense BF16/FP16 |
| Hybrid, MLA, sliding/local attention | Conflict | Startup rejection |
| Async scheduling | Conflict | Startup rejection |
| TP, PP, DP, DCP, PCP > 1 | Conflict | Startup rejection |
| BidKV or another scheduler class | Conflict | Upstream v1 `Scheduler`, or the current inert Ascend `BalanceScheduler` wrapper, required; active balance scheduling is rejected |
| Former Prefix Router, KV Tiering, KNorm, PyramidKV Ascend, SliceGPT code | Not integrated | No assumptions or imports; these implementations are absent from current hosts |
| Other `vllm.general_plugins` | Unverified | Use an explicit allowlist and validate independently |

## Configuration and calibration

The shipped example selects a 2048-token physical budget, a 128-token
recompute window, and one scoring layer in every four. `kv_budget`,
`recompute_window`, and `score_chunk_size` must be positive multiples of 128.
Use model-matched statistics and read the
[calibration artifact guide](docs/calibration-artifacts.md) before deployment.

## Validation and development

- [Current acceptance record](docs/validation.md)
- [Benchmark protocol](docs/benchmarking.md)
- [Packaging and release guide](docs/packaging-and-release.md)
- [Method extension API](docs/methods.md)
- [Calibration artifacts](docs/calibration-artifacts.md)

Run CPU/package checks with:

```bash
VLLM_PLUGINS='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -q
python -m ruff check src tests
```

An NPU release candidate must additionally pass kernel numerical smoke,
long-context quality, matched baseline/compression throughput and latency, and
block/HBM acceptance on the exact declared host snapshots.
